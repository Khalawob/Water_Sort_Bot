# Architectural Patterns

Cross-cutting patterns that appear across multiple modules.

## Pipeline Architecture

The bot follows a per-round pipeline, re-run from scratch each iteration:

```
screenshot() -> auto_calibrate() -> read_tubes() -> [overlay learned] -> solve/reveal -> execute_move_list()
```

`_solve_one_level_impl` (`main.py:726`) orchestrates this with up to `max_rounds=40` iterations per attempt and up to 12 restarts per level (`main.py:743`). Each round re-screenshots and re-calibrates to reflect the board state after the previous round's pours. The overlay step merges `LevelMemory`'s cross-restart knowledge onto the freshly-read board before solving.

## Separation of Concerns

Each module owns one pipeline stage. Imports are one-directional -- only `main.py` imports from the others:

| Module | Responsibility | Project imports |
|---|---|---|
| `capture.py` | ADB I/O, scrcpy process management, window detection | none |
| `auto_calibrate.py` | Tube geometry detection, button/win/popup detection | none |
| `screen_reader.py` | Pixel colour reading, hidden-slot detection | `capture` |
| `solver.py` | Pure search algorithms (A*, DFS, 10 reveal planners) | none |
| `level_memory.py` | Cross-restart learning, attempt tracking, disk persistence | none |
| `automator.py` | Device-to-screen coordinate mapping, tap execution | `capture` |
| `main.py` | CLI entry point, pipeline orchestration, multi-attempt retry | all above |

`solver.py` and `level_memory.py` have zero project imports, making them independently testable.

## Config as Shared State Contract

`config.json` is the single source of truth for calibration data. Key consumers of `tubes[].sample_points`:

- `auto_calibrate.py:276` (`auto_calibrate`) produces it per-frame
- `screen_reader.py:113` (`read_tubes`) reads colours from those pixel coordinates
- `automator.py:103` (`get_tube_tap_zones`) derives tap targets from them

Additional keys: `tube_capacity`, `empty_colour`, `next_button`, `restart_button`. Config is regenerated from the screenshot each round but preserves button locations across rounds (`main.py:215`, `get_config_for_level`).

## UNKNOWN Sentinel (Mirrored, Not Imported)

`UNKNOWN = "unknown"` is defined independently in three files:
- `solver.py:20`
- `screen_reader.py:32`
- `level_memory.py:23`

This is deliberate: it keeps the dependency graph one-directional (no cross-imports between library modules). Any change to the sentinel value must be mirrored in all three locations.

The sentinel threads through the entire pipeline: `screen_reader` marks hidden slots as UNKNOWN, `solver` never pours them and penalises them in heuristics, `level_memory` records revealed colours keyed by (tube, depth), and `main.py`'s overlay replaces UNKNOWN with learned colours before solving.

## Fallback Chains

Three independent fallback chains handle failure gracefully:

**Screenshot capture** (`capture.py`):
`screenshot()` at line 148 (ADB fast pipe `exec-out screencap -p`) falls back to `_screenshot_legacy()` at line 178 (two-step `screencap` + `adb pull`).

**Solver** (`solver.py:1452`):
`solve()` tries A* with `max_states=1_000_000`, then DFS with restricted empties (`max_states=2_000_000`), then DFS with relaxed empties (`max_states=2_000_000`).

**Auto-calibration** (`main.py:215`, `get_config_for_level`):
Fresh per-frame detection, falling back to saved config from the previous level.

## Module-Level Lazy Caches

| Cache | Location | Purpose |
|---|---|---|
| `_cached_mapping` | `automator.py:25` | scrcpy window bounds + scale factors |
| `_device_resolution` | `capture.py:196` | device screen dimensions from `adb shell wm size` |
| `REVEAL_STATS` | `main.py:65` | per-planner reached/used counts (dict) |
| `REVEAL_TIMES` | `main.py:80` | per-planner cumulative wall-clock timing (dict) |

The first two use `None` as the uninitialized sentinel. The latter two are zero-initialised dicts reset between levels.

## Device Coordinate Abstraction

All game coordinates (tube positions, button locations) are stored in **device space** (native Android pixels). Conversion to screen coordinates happens only at tap time via `_device_to_screen()` (`automator.py:62`), using cached scale factors. This makes `config.json` independent of scrcpy window size or position.

## Human-Like Input Randomisation

Tap positions and timing are randomised to avoid detection:

- **Position jitter** (`automator.py:129`, `jittered_tap`): random offset around tube center, scaled to tube spread
- **Timing jitter** (`automator.py:136`, `human_delay`): `base +/- 0.15s`
- **Pour wait scaling** (`main.py:202`): `0.83 + 0.52 * num_poured` -- linear with balls poured

## Immutable State in Solver

Puzzle state is `tuple(tuple(t) for t in state)` -- a tuple of tuples. This makes states hashable for the `seen` set in A* and `visited` in DFS, enabling O(1) revisit detection. All state transitions create new tuples via `apply_move()` (`solver.py:690`).

## Cross-Restart Learning (LevelMemory)

The game is deterministic on restart: the same level always has the same hidden-slot layout. `LevelMemory` (`level_memory.py:35`) exploits this:

- **Persistence**: JSON file keyed by sha1 signature of (tube labels, RGB values, capacity)
- **Records**: maps `(tube, depth)` to the RGB colour of originally-hidden slots
- **Overlay**: `main.py` merges learned slots onto freshly-read boards before solving
- **AttemptSim** (`level_memory.py:196`): tracks ball origins during an attempt so newly-read colours can be attributed back to their original (tube, depth) positions
- **Corruption guard**: capacity violations trigger `memory.delete()` + retry (once per level)

## UI Element Detection (auto_calibrate.py)

Beyond tube geometry, `auto_calibrate.py` detects UI overlays:

- `detect_buttons()` (`auto_calibrate.py:338`): finds purple/lilac UI buttons (menu, restart, undo, add_tube)
- `detect_win_screen()` (`auto_calibrate.py:379`): red banner + yellow NEXT / green CLAIM button
- `detect_popup()` (`auto_calibrate.py:476`): theme-unlock or special-level popup overlays

These are used by `main.py` in `tap_next_level` (`main.py:1361`) to navigate between levels in loop mode.

## Debug Instrumentation

- `debug_screenshots/level_NNN/` -- per-level directories with round-end screenshots
- `rounds.txt` per level -- stdout captured via `_Tee` class, includes planner decisions and board state
- `REVEAL_STATS` / `REVEAL_TIMES` dicts (`main.py:65,80`) track planner hit rates and latencies
- `format_reveal_stats()` / `print_reveal_stats()` produce human-readable summaries

## Reveal Chain Architecture

The reveal chain (`main.py:1034-1168`) is a prioritised cascade of 10 planners. Each is tried only if all higher-priority planners produced no moves. The chain balances information-theoretic optimality (info_gain, deep) against exhaustive safety (guaranteed_safe) and heuristic fallbacks (maximize, determinization, reveal_round).