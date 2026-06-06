# Water Sort Bot — Continuation Prompt

## Project Overview

Automated Water Sort / Ball Sort bot for Android. Reads the board via ADB screenshots, solves with A*, plays moves via scrcpy + pyautogui. Handles levels with hidden "?" slots through a reveal-restart-remember loop. The bot restarts levels to accumulate knowledge of hidden slots across attempts, using a persistent `LevelMemory` that stores discovered RGB values keyed by board signature.

## Architecture (7 modules)

- **`main.py`** — Orchestrator. Outer loop: screenshot → read → overlay memory → plan → execute → repeat. Contains `solve_one_level`, the reveal chain, `run_late_game`, `select_reveal_prefix`, and reconciliation logic.
- **`solver.py`** — A* solver, reveal planners (info-gain, deep-reveal, consolidation, maximize, determinization), `is_late_game`, `plan_late_game_solve`, `find_guaranteed_safe_moves`, completion pool utilities.
- **`screen_reader.py`** — Reads pixel colours at calibrated sample points, assigns labels, detects hidden "?" via bimodality test, detects overlays.
- **`auto_calibrate.py`** — Finds tubes by grey glass borders, calculates sample points, detects UI buttons.
- **`automator.py`** — Taps scrcpy window via pyautogui, maps device→screen coords, applies jitter, calculates pour-wait times.
- **`level_memory.py`** — Persists discovered hidden-slot RGBs, stores attempt logs (moves, reveals, outcomes), provides `AttemptSim` for origin tracking.
- **`capture.py`** — ADB screenshot capture, scrcpy lifecycle management.

## Reveal Chain Order

```
late-game (≤5 unknowns) → safe → info-gain → deep-reveal → consolidate → maximize → determinization → reveal_round
```

Late-game bypasses the entire chain when it triggers. If it returns "clean_fail" (couldn't plan), the chain runs as fallback.

## What We Built In This Conversation

### 1. Late-Game Full-Solve Strategy (Version 2)
**The problem**: After learning ~28/31 hidden slots, the remaining unknowns are buried at tube bottoms (depth 0). The reveal chain can't reach them — deep-reveal needs 3 empty tubes for a 3-layer peel but only 2 exist. The bot would grind through 12 failing attempts.

**The solution**: When ≤5 unknowns remain (at any depth), skip the reveal chain. Sample a random completion for the unknowns, A*-solve the fully-specified board for a complete solution (30–50 moves), and execute the full solution. Since unknowns are buried, the solution executes correctly for many moves before any unknown becomes a tube's top. When a pour exposes an unknown (its tube's top becomes UNKNOWN), pause, screenshot, read the real colour, record it to memory, then **always re-plan** from the corrected state (~0.5s per A* call). Repeat until solved.

**Key implementation details**:
- `is_late_game(state, max_unknowns=5)` in `solver.py` — counts total UNKNOWNs across all tubes, returns True when 0 < total ≤ 5. No depth restriction.
- `plan_late_game_solve(state, capacity, max_samples=5)` in `solver.py` — builds completion pool, samples completions, A*-solves each, returns first solvable `(moves, completed_state)`.
- `run_late_game(...)` in `main.py` — executes solution move-by-move, tracking `sim_state`. After each move, checks `sim_state[src] and sim_state[src][-1] == UNKNOWN`. If exposed: multi-read confirmation (3 screenshots, majority vote), record to memory, re-plan. Returns "solved", "dirty" (moves executed but couldn't finish), or "clean_fail" (couldn't even plan).
- Integration in `solve_one_level`: checked after memory overlay but before the reveal chain. On "solved" → level done. On "dirty" → `continue` (re-screenshot, never fall through to reveal chain with stale state). On "clean_fail" → fall through to reveal chain (board unchanged).

**Design decisions**:
- **Always re-plan after every exposure** (not just on mismatch). Costs ~0.5s × ≤5 exposures = negligible. Avoids fragile RGB-vs-label comparison logic (labels are reassigned per screenshot read, making match detection unreliable).
- **Exposure detection uses `tube[-1] == UNKNOWN`** (not depth-0 check), so it works at any depth. The unknown's original position is `len(sim_state[src]) - 1` because apply_move never pours UNKNOWNs and nothing below an unknown can be removed.
- **On failure ("dirty"), use `continue` not fall-through.** If late-game executed moves before failing, the physical board has changed. The reveal chain's state would be stale. `continue` re-screenshots.

### 2. Safe Gating
**The problem**: `find_guaranteed_safe_moves` ran 75 times at ~3s each (205s total) with 0% hit rate. It enumerates multiset permutations of the completion pool and A*-solves each. On hard levels, it never finds anything because the completion space is too large for any single move to be safe across all worlds.

**The fix**: Lowered `SAFE_MAX_UNKNOWNS` from 10 to 5 in `solver.py` (line 22). The function already checks `len(unknown_pos) > max_unknowns` and bails immediately. By lowering the threshold, the expensive 6–10 unknown cases (which never produced results) bail instantly instead of doing full enumeration.

**Result**: 114.77s → 0.09s on the 14-tube level.

### 3. Score_prefix Gating
**The problem**: `select_reveal_prefix` ran every round (~6s each, 233s total) regardless of which planner produced the reveal batch. For fallback planners (consolidate, maximize, determinization, reveal_round), this was wasteful AND actively harmful — it trimmed consolidation moves, breaking their purpose (e.g., trimming a 5-move consolidation sequence to 3 moves that don't actually free a tube).

**The fix**: Track `reveal_source` through the reveal chain. Only run `select_reveal_prefix` when `reveal_source in {"info_gain", "deep"}`. Skip for safe (already solvability-checked), consolidate (sorts known tubes, not reveals), maximize, determinization, and reveal_round.

**Critical discovery**: Without this gating, the bot couldn't solve the level. Score_prefix was trimming consolidation moves into uselessness, preventing the bot from ever freeing empty tubes for deep peels. **This gating isn't just a performance optimization — it's a correctness fix.**

**Result**: Score_prefix calls dropped from 37 to 26 (info_gain + deep only), and consolidation actually works now.

### 4. Broadened Late-Game Trigger
**The problem**: Late-game originally only fired when ALL unknowns were at depth 0 (tube bottoms). This meant waiting until ~31/36 known on hard levels (9+ attempts).

**The fix**: Removed the depth-0 restriction from `is_late_game`. Now triggers when total unknowns ≤ 5 regardless of depth. Updated exposure detection from `len(tube) == 1 and tube[0] == UNKNOWN` to `tube and tube[-1] == UNKNOWN`. Updated memory recording from hardcoded depth `0` to `len(sim_state[src]) - 1`.

**Why it's safe**: Unknowns at higher depths get exposed earlier in the solution (fewer peel moves), so re-planning happens sooner with a less-shuffled board — actually easier for A*.

### 5. Adaptive `needed_empties` in Consolidation
**Pre-existing fix from before this conversation**: Changed `plan_consolidate`'s default `needed_empties` from fixed `3` to `current_empties + 1`. This made 0→1, 1→2, 2→3 each individually achievable, letting the bot incrementally build up empty tubes.

## Current Performance (Latest Run — New Level)

```
=== Reveal planner stats ===
  safe             reached   29  used    0  (0% hit)
  info_gain        reached   29  used   15  (52% hit)
  deep             reached   14  used    2  (14% hit)
  consolidate      reached   11  used    0  (0% hit)
  maximize         reached   12  used    0  (0% hit)
  reveal_round     reached   12  used    1  (8% hit)
  determinization  reached   12  used    0  (0% hit)

=== Reveal planner timings ===
  safe                42.97s     29   1481.7ms
  info_gain            0.02s     29      0.6ms
  deep                 0.00s     14      0.1ms
  consolidate          0.12s     11     10.6ms
  maximize             0.00s     12      0.0ms
  determinization      6.58s     12    548.0ms
  reveal_round         0.00s     12      0.1ms
  score_prefix        57.16s     17   3362.1ms
  full_solve           0.00s      0      0.0ms
```

Level solved. 5 attempts, late-game triggered in attempt 3 (after 24/29 known).

## What Needs To Be Done Next

### Problem: Late-game fails frequently, causing expensive safe fallthrough

On the latest test level (14 tubes, 29 hidden), late-game triggered 10 times but couldn't plan 8 of those times (all 5 sampled completions unsolvable by A*). Each "clean_fail" falls through to the reveal chain where `safe` runs with ≤5 unknowns, does full permutation enumeration (~5s), finds nothing. Those 8 fallthrough rounds account for nearly all 43s of safe time.

### Fix 1: Increase `max_samples` in `plan_late_game_solve`

Change `max_samples` from 5 to 15–20. The solver CAN solve this board (attempt 5 proves it) — it just needs more attempts to find a solvable completion by random sampling. Each failed A* sample costs ~0.5s, so 20 samples = ~10s worst case. Much cheaper than one round of safe fallthrough (~5s) happening 8 times.

**File**: `solver.py`, `plan_late_game_solve` function. Change the default parameter:
```python
def plan_late_game_solve(state, capacity, max_samples=20):
```

### Fix 2: Skip `safe` after late-game clean_fail

If late-game couldn't find a solvable completion across N samples, `safe` won't find guaranteed-safe moves either — it uses the same solvability machinery on the same board. Add a flag so that when late-game falls through to the reveal chain, `safe` is skipped in that round.

**Implementation**: In `main.py`'s `solve_one_level`, after the late-game block:
```python
skip_safe = False
if is_late_game(state) and not dry_run:
    # ... late-game logic ...
    if result == "clean_fail":
        skip_safe = True  # A* couldn't solve any completion; safe won't either
        # fall through to reveal chain

# In the reveal chain:
if not skip_safe:
    REVEAL_STATS["safe_reached"] += 1
    with time_stage("safe"):
        reveal = find_guaranteed_safe_moves(state_mid, capacity, prev_state=prev_state)
```

### Expected Impact

Fix 1 should eliminate most late-game failures (higher chance of finding a solvable completion). Fix 2 is the safety net — when late-game still fails, safe is skipped, saving ~5s per fallthrough round. Together, safe time should drop from 43s to near-zero, and the bot should need fewer attempts overall.

## Key Gotchas and Lessons Learned

1. **Score_prefix trimming consolidation moves is a correctness bug, not just a performance issue.** Without the score_prefix gating, the bot can't solve hard levels because consolidation sequences get truncated into uselessness.

2. **`state` vs `state_mid`**: In `main.py`, `state` is the board after memory overlay and deduction. `state_mid` is `state` after reclaim moves. They have the same unknown count but different tube arrangements. Late-game uses `state`; the reveal chain uses `state_mid`.

3. **Labels are per-read, not stable.** Colour labels (`colour_1`, etc.) are reassigned every screenshot read. Memory stores raw RGB tuples. Never compare labels across reads — compare RGB with tolerance (Euclidean distance < 5). This is why the late-game always re-plans instead of comparing guessed vs real colours.

4. **Multi-read confirmation is essential.** When a hidden slot is exposed mid-animation, a single screenshot can capture transitional frames with wrong colours. The bot takes 3 ADB screenshots and majority-votes per slot before recording to memory.

5. **`apply_move` never pours UNKNOWNs.** This is by design — the solver refuses to move unknown balls. This property is what makes late-game exposure detection work: an unknown stays at its original position until everything above it is poured off, so `len(tube) - 1` after exposure equals the original depth.

6. **Physical board divergence after "dirty" late-game.** If late-game executes moves then fails (re-plan fails), the physical phone board has changed but `state` in the main loop is stale. Must `continue` to re-screenshot, never fall through to the reveal chain.

7. **The log line still says "all unknowns at depth 0"** (line 795) — this is outdated since we broadened the trigger. The print should be updated to "≤5 unknowns remain" or similar. Minor but confusing when reading logs.

## Board Representation

- State: tuple of tuples (hashable for A* seen-sets). Bottom=index 0, top=index -1.
- `UNKNOWN` sentinel marks hidden slots.
- Capacity is typically 4 (balls per tube).
- Empty tube = `()`.
- Solved tube = all 4 slots same colour, no UNKNOWN.

## Testing

The bot has been tested on multiple levels:
- **Level 98**: 13 tubes, 31 hidden, 2 empties. Solved in 8 attempts (was failing at 12 before late-game).
- **14-tube level** (from initial.png): 14 tubes, 36 hidden, 2 empties. Solved in ~10 attempts.
- **Latest level**: 14 tubes, 29 hidden, 2 empties. Solved in 5 attempts.

To test: run `python main.py` with phone connected via ADB, scrcpy running. The bot auto-calibrates tube positions and solves continuously. `rounds.txt` logs all rounds. `level_memory.json` persists across runs.
