#!/usr/bin/env python3
"""
Water Sort Bot — ADB screenshots + scrcpy taps + auto-calibration.

Auto-detects tube positions from each screenshot. No manual calibration
needed except for the 'Next Level' button (one-time).

Supports hidden/unknown slots with iterative solve-reveal-solve.
"""

import argparse
import hashlib
import io
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2

from solver import (
    solve, plan_reveal_round, plan_reveal_maximize, plan_reveal_info_gain,
    plan_reveal_deep, plan_consolidate, _exposes_unknown,
    find_reclaim_moves, find_safe_moves, apply_move, UNKNOWN, deduce_hidden_slots,
    pick_best_move_by_determinization, validate_move_sequence,
    find_guaranteed_safe_moves, score_reveal_batch,
    sample_solvable_completions, is_late_game, plan_late_game_solve,
    find_path_to_unknown,
)
from screen_reader import (
    detect_no_more_moves,
    read_tubes, has_unknowns, load_config, save_config, calibrate,
    is_game_screen, wait_for_game_screen, colour_distance, CONFIG_PATH,
)
from level_memory import LevelMemory, AttemptSim, MEMORY_PATH
from capture import screenshot, launch_scrcpy, stop_scrcpy
from automator import (
    adb_tap, human_delay, jittered_tap,
    get_tube_tap_zones, refresh_mapping,
)
from auto_calibrate import auto_calibrate, visualise_detection, detect_buttons


# Consecutive barren (learned-nothing) attempts tolerated before giving up on a
# level — retries only help once the maximizer's randomized tie-breaking makes
# them diverge (see plan_reveal_maximize).
PATIENCE = 3

# Only score reveal batches for solvability when empties are scarce (dead-end
# risk is real); with more empties than this the batch can't strand the board.
REVEAL_SOLVABILITY_EMPTY_GATE = 1

# Skip solvability scoring when too many unknowns remain: each sample fills all
# unknowns and A*-solves (~15-20s), and most random completions are unsolvable,
# so the estimate is noise. Late-game fires at <=5 unknowns, so scoring only
# meaningfully helps in the 6-8 range; above this the cost is wasted.
SCORE_PREFIX_MAX_UNKNOWNS = 8

# Instrumentation: how often each reveal planner is reached vs. actually used.
REVEAL_STATS = {
    "safe_reached": 0, "safe_used": 0,
    "info_gain_reached": 0, "info_gain_used": 0,
    "deep_reached": 0, "deep_used": 0,
    "consolidate_reached": 0, "consolidate_used": 0,
    "maximize_reached": 0, "maximize_used": 0,
    "reveal_round_reached": 0, "reveal_round_used": 0,
    "determinization_reached": 0, "determinization_used": 0,
    "path_to_unknown_reached": 0, "path_to_unknown_used": 0,
    "heal": 0, "patience_retry": 0,
}

# Instrumentation: cumulative wall-clock per planner stage as [seconds, calls],
# so the dominant cost is provable from rounds.txt / the end-of-run table. See
# `time_stage` and `print_reveal_stats`.
REVEAL_TIMES = {
    "safe": [0.0, 0], "info_gain": [0.0, 0], "deep": [0.0, 0],
    "consolidate": [0.0, 0],
    "maximize": [0.0, 0], "determinization": [0.0, 0], "reveal_round": [0.0, 0],
    "path_to_unknown": [0.0, 0],
    "score_prefix": [0.0, 0], "full_solve": [0.0, 0],
}


@contextmanager
def time_stage(stage):
    """Accumulate elapsed time + a call into REVEAL_TIMES[stage] and print it
    inline so each round's planning cost is visible in the log."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        REVEAL_TIMES[stage][0] += elapsed
        REVEAL_TIMES[stage][1] += 1
        print(f"  ⏱ {stage} {elapsed:.2f}s")


def format_reveal_stats():
    lines = []
    s = REVEAL_STATS
    lines.append("\n=== Reveal planner stats ===")
    for stage in ("safe", "info_gain", "deep", "consolidate", "maximize",
                  "reveal_round", "determinization", "path_to_unknown"):
        reached = s[f"{stage}_reached"]
        used = s[f"{stage}_used"]
        pct = (100 * used / reached) if reached else 0.0
        lines.append(f"  {stage:16s} reached {reached:4d}  used {used:4d}  ({pct:.0f}% hit)")
    lines.append(f"  {'heal':16s} {s['heal']:4d}   {'patience_retry':16s} {s['patience_retry']:4d}")

    t = REVEAL_TIMES
    lines.append("\n=== Reveal planner timings ===")
    lines.append(f"  {'stage':16s} {'total':>9s} {'calls':>6s} {'mean':>10s}")
    for stage in ("safe", "info_gain", "deep", "consolidate", "maximize",
                  "determinization", "reveal_round", "path_to_unknown",
                  "score_prefix", "full_solve"):
        total, calls = t[stage]
        mean_ms = (1000 * total / calls) if calls else 0.0
        lines.append(f"  {stage:16s} {total:8.2f}s {calls:6d} {mean_ms:8.1f}ms")
    return "\n".join(lines)


def print_reveal_stats():
    print(format_reveal_stats())


def _rgb_signature(state, label_to_rgb):
    """Label-independent board fingerprint: map each colour label to its RGB so
    a relabeled-but-identical board compares equal. Tube order is geometry-stable
    so it's kept. Returns a hashable tuple."""
    return tuple(
        tuple("?" if slot == UNKNOWN else tuple(label_to_rgb.get(slot, (slot,)))
              for slot in tube)
        for tube in state
    )


def _hash_board(state):
    """Stable sha1 of a board's sorted tube contents (UNKNOWN kept verbatim).

    Order-independent across tubes so two reads of the same stuck layout hash
    equal regardless of tube enumeration. Returns None for a falsy/empty state.
    """
    if not state:
        return None
    canon = sorted(tuple(tube) for tube in state)
    return hashlib.sha1(repr(canon).encode()).hexdigest()


def read_and_display(image, config, tube_capacity, return_colours=False):
    """Read tubes, print them, return (tubes, is_valid).

    When ``return_colours`` is True, also returns the ``seen_colours`` map
    ({rgb_tuple: label}) as a third element so callers can recover RGB values.
    """
    if return_colours:
        tubes, seen_colours = read_tubes(image, config, return_colours=True)
    else:
        tubes = read_tubes(image, config)

    print(f"\nDetected {len(tubes)} tubes (capacity {tube_capacity}):")
    for i, tube in enumerate(tubes):
        label = " | ".join(tube) if tube else "(empty)"
        print(f"  Tube {i+1}: [{label}]")

    all_colours = [c for t in tubes for c in t if c != UNKNOWN]
    colour_counts = {}
    for c in all_colours:
        colour_counts[c] = colour_counts.get(c, 0) + 1

    unknown_count = sum(t.count(UNKNOWN) for t in tubes)

    print(f"\nKnown colours: {len(colour_counts)}")
    valid = True
    for colour, count in sorted(colour_counts.items()):
        if count != tube_capacity:
            valid = False
        status = "✓" if count == tube_capacity else f"({count})"
        print(f"  {colour}: {count}  {status}")

    if unknown_count > 0:
        print(f"  Hidden slots: {unknown_count}")

    if return_colours:
        return tubes, valid, seen_colours
    return tubes, valid


def execute_move_list(moves, config, tube_capacity):
    """Execute a list of moves with smart timing."""
    zones = get_tube_tap_zones(config)
    refresh_mapping()

    for i, (src, dst, num_poured) in enumerate(moves, 1):
        src_x, src_y = jittered_tap(zones[src])
        dst_x, dst_y = jittered_tap(zones[dst])

        pour_wait = 0.83 + (0.52 * num_poured)

        print(f"    Move {i}/{len(moves)}: Tube {src+1} → Tube {dst+1} "
              f"({num_poured} poured, wait {pour_wait:.1f}s)")

        adb_tap(src_x, src_y)
        time.sleep(0.3)
        adb_tap(dst_x, dst_y)
        time.sleep(pour_wait)

    return True


def get_config_for_level(image, existing_config=None, fallback_to_existing=True):
    """
    Auto-detect tubes from the screenshot.
    Preserves next_button from existing config if available.
    """
    print("🔎 Auto-detecting tubes...")
    new_config = auto_calibrate(image)

    if new_config is None:
        if existing_config and fallback_to_existing:
            print("  ⚠ Auto-detection failed, using saved config.")
            return existing_config
        if not existing_config:
            print("  ✗ Auto-detection failed and no saved config.")
        return None

    # Preserve next_button and restart_button from existing config
    for key in ("next_button", "restart_button"):
        if existing_config and key in existing_config:
            new_config[key] = existing_config[key]
            print(f"  ↻ Preserved {key} from existing config.")

    # Auto-detect buttons for any still missing
    buttons = detect_buttons(image)
    if "restart_button" not in new_config and "restart" in buttons:
        x, y = buttons["restart"]
        new_config["restart_button"] = {"x": x, "y": y}
        print(f"  ✓ Auto-detected restart_button at ({x}, {y}).")
    if "next_button" not in new_config and "next" in buttons:
        x, y = buttons["next"]
        new_config["next_button"] = {"x": x, "y": y}
        print(f"  ✓ Auto-detected next_button at ({x}, {y}).")

    # Save for debugging
    save_config(new_config)

    return new_config


class _Tee:
    def __init__(self, real, buf):
        self._real = real
        self._buf = buf

    def write(self, data):
        self._real.write(data)
        self._buf.write(data)

    def flush(self):
        self._real.flush()
        self._buf.flush()

    def fileno(self):
        return self._real.fileno()


def _overlay_learned_colours(tubes, learned_slots, seen_colours, label_to_rgb):
    """Replace UNKNOWN slots with labels for learned RGBs, in place.

    For each learned ``(tube, depth) → rgb`` whose slot is still UNKNOWN, reuse
    an existing label if that RGB is already known (within the same tolerance
    read_tubes uses); otherwise mint a new label for a fully-hidden colour and
    register it in both ``seen_colours`` and ``label_to_rgb``.
    """
    if not learned_slots:
        return 0

    existing_nums = [
        int(name.split("_")[1])
        for name in seen_colours.values()
        if name.startswith("colour_") and name.split("_")[1].isdigit()
    ]
    counter = max(existing_nums) if existing_nums else 0

    filled = 0
    for (ti, depth), rgb in learned_slots.items():
        if ti >= len(tubes):
            continue
        tube = tubes[ti]
        if depth >= len(tube) or tube[depth] != UNKNOWN:
            continue

        label = None
        for known_rgb, name in seen_colours.items():
            if colour_distance(rgb, known_rgb) < 5:
                label = name
                break
        if label is None:
            counter += 1
            label = f"colour_{counter}"
            seen_colours[rgb] = label
            label_to_rgb[label] = rgb

        tube[depth] = label
        filled += 1

    if filled:
        print(f"  🧠 Recalled {filled} hidden slot(s) from memory.")
    return filled


def _batch_exposed_unknown(state, moves, capacity):
    """True if replaying ``moves`` on ``state`` exposes a freshly-hidden top.

    A reveal move pours a tube down to an UNKNOWN slot, so its source tube ends
    with an UNKNOWN top in our model. Only then is a confirming post-move read
    worth the extra screenshot.
    """
    sim = tuple(tuple(t) for t in state)
    for src, dst, _ in moves:
        sim, _ = apply_move(sim, src, dst, capacity)
        if sim[src] and sim[src][-1] == UNKNOWN:
            return True
    return False


def _majority_rgb(votes, tol=5):
    """Return an RGB that ``>=2`` of ``votes`` agree on (within ``tol``), else None.

    ADB screenshots are pixel-perfect, so a settled board produces identical RGB
    across reads and the votes are equal; mid-animation artifacts are inconsistent
    and fail to reach a majority. ``tol`` matches read_tubes' matching tolerance.
    """
    for candidate in votes:
        if sum(1 for other in votes if colour_distance(candidate, other) < tol) >= 2:
            return candidate
    return None


def _majority_read(reads, tol=5):
    """Build a consensus ``(tubes_labels, label_to_rgb)`` from several reads.

    ``reads`` is a list of ``(tubes_labels, label_to_rgb)`` post-move reads. For
    each slot position, only the RGB agreed on by a 2-of-3 majority survives;
    disputed slots (likely still mid-animation) become UNKNOWN so reconcile skips
    them. Slots stay positioned (UNKNOWN placeholders) so tube lengths are
    preserved for reconcile's shape check. Labels are synthetic (``vote_N``).
    """
    # The read with the most tubes sets the structure; shorter reads just abstain.
    ref_tubes = max(reads, key=lambda r: len(r[0]))[0]
    consensus_tubes = []
    consensus_l2r = {}
    seen = {}                                  # rgb -> label
    counter = 0
    confirmed = 0
    skipped = []                               # (ti, d) candidate slots w/o consensus
    for ti in range(len(ref_tubes)):
        depth = max((len(t[ti]) for t, _ in reads if ti < len(t)), default=0)
        layers = []
        for d in range(depth):
            votes = []
            for tubes, l2r in reads:
                if ti < len(tubes) and d < len(tubes[ti]):
                    label = tubes[ti][d]
                    if label != UNKNOWN:
                        votes.append(tuple(l2r[label]))
            winner = _majority_rgb(votes, tol)
            if winner is None:
                layers.append(UNKNOWN)
                # A slot with colour votes but no majority is a dropped candidate;
                # one UNKNOWN in every read is just still-hidden, not a dispute.
                if votes:
                    skipped.append((ti, d))
                    print(f"  ⚠ No consensus for tube {ti + 1} slot {d + 1} "
                          "— skipping record")
                continue
            confirmed += 1
            match = next((lbl for rgb, lbl in seen.items()
                          if colour_distance(rgb, winner) < tol), None)
            if match is None:
                counter += 1
                match = f"vote_{counter}"
                seen[winner] = match
                consensus_l2r[match] = winner
            layers.append(match)
        consensus_tubes.append(layers)
    print(f"  🗳 Majority vote: {confirmed}/{confirmed + len(skipped)} slots "
          f"confirmed, {len(skipped)} skipped (no consensus)")
    return consensus_tubes, consensus_l2r


def _read_exposed_slot(config, src, depth=0, settle=1.0):
    """Read the colour of a freshly-exposed slot at ``depth`` in tube ``src``.

    Used by the late-game full-solve path after a peel leaves the unknown as the
    tube's top (index ``depth``). Waits ``settle`` for the pour animation, then
    majority-votes 3 ADB reads (same machinery as the round reconciliation) so a
    mid-animation frame can't poison the read. Returns the slot's RGB tuple, or
    ``None`` when the slot still reads UNKNOWN / has no 2-of-3 consensus.
    """
    time.sleep(settle)
    reads = []
    for _ in range(3):
        img = screenshot()
        if img is None:
            continue
        rt, rseen = read_tubes(img, config, return_colours=True)
        reads.append((rt, {label: rgb for rgb, label in rseen.items()}))
        time.sleep(0.15)
    if not reads:
        return None

    reconcile_tubes, reconcile_l2r = _majority_read(reads)
    if src >= len(reconcile_tubes) or depth >= len(reconcile_tubes[src]):
        return None
    label = reconcile_tubes[src][depth]
    if label == UNKNOWN:
        return None
    return reconcile_l2r[label]


def run_late_game(state, capacity, memory, signature, attempt_moves,
                  attempt_reveals, config, label_to_rgb):
    """Late-game full-solve: execute a sampled complete solution, pausing to read
    each unknown as it becomes the tube's top (at any depth), then re-solving from
    the corrected board.

    Returns one of:
      "solved"     — the board was driven to completion.
      "dirty"      — moves were executed but the run couldn't finish (re-plan
                     failed or a read failed); exposed slots are already recorded,
                     caller should re-screenshot next round.
      "clean_fail" — nothing was executed (couldn't even plan); caller may fall
                     through to the reveal chain on the still-valid board.
    """
    sim_state = tuple(tuple(t) for t in state)
    executed_any = False

    # Label minting for a re-seen / brand-new colour, mirroring _overlay_learned_colours.
    existing_nums = [
        int(name.split("_")[1])
        for name in label_to_rgb
        if name.startswith("late_") and name.split("_")[1].isdigit()
    ]
    late_counter = max(existing_nums) if existing_nums else 0

    while True:
        planned = plan_late_game_solve(sim_state, capacity)
        if planned is None:
            return "dirty" if executed_any else "clean_fail"
        solution, _filled = planned
        print(f"  🧩 Late-game plan: {len(solution)} move(s) on a sampled completion")

        # Fresh scrcpy window before each execution batch
        stop_scrcpy()
        launch_scrcpy()
        refresh_mapping()
        zones = get_tube_tap_zones(config)  # re-fetch zones with fresh mapping

        replanned = False
        for i, (src, dst, n) in enumerate(solution, 1):
            # Execute one move — same tap sequence + wait as execute_move_list.
            src_x, src_y = jittered_tap(zones[src])
            dst_x, dst_y = jittered_tap(zones[dst])
            pour_wait = 0.83 + (0.52 * n)
            print(f"    Move {i}/{len(solution)}: Tube {src+1} → Tube {dst+1} "
                  f"({n} poured, wait {pour_wait:.1f}s)")
            adb_tap(src_x, src_y)
            time.sleep(0.3)
            adb_tap(dst_x, dst_y)
            time.sleep(pour_wait)
            executed_any = True

            sim_state, _ = apply_move(sim_state, src, dst, capacity)
            attempt_moves.append((src, dst, n))

            # Exposure: pouring stripped the last known ball above an unknown,
            # making it the tube's new top.  Works at any depth.
            if sim_state[src] and sim_state[src][-1] == UNKNOWN:
                attempt_reveals.append(1)
                depth = len(sim_state[src]) - 1
                rgb = _read_exposed_slot(config, src, depth)
                if rgb is None:
                    print(f"  ⚠ Couldn't read exposed slot in tube {src+1} — "
                          "stopping late-game (will re-read next round).")
                    return "dirty"

                # Origin == current position: apply_move never pours UNKNOWN
                # and nothing below an unknown can be removed, so it sits at its
                # original index throughout execution.
                memory.record_slot(signature, src, depth, rgb, capacity)

                # Map RGB → label so the re-solve treats a re-seen visible colour
                # as that colour (keeps the completion pool consistent).
                label = None
                for name, known_rgb in label_to_rgb.items():
                    if colour_distance(rgb, known_rgb) < 5:
                        label = name
                        break
                if label is None:
                    late_counter += 1
                    label = f"late_{late_counter}"
                    label_to_rgb[label] = rgb
                print(f"  🧠 Tube {src+1} depth {depth} revealed → {label}; recorded to memory.")

                tubes = [list(t) for t in sim_state]
                tubes[src][-1] = label
                sim_state = tuple(tuple(t) for t in tubes)

                replanned = True
                break                              # re-plan from corrected board
            else:
                attempt_reveals.append(0)

        if replanned:
            continue
        # Solution ran to the end without exposing an unknown → board is solved.
        return "solved"


def run_reveal_intervention(config, capacity, memory, signature, n_reveals=2):
    """Restart-and-reveal loop: for each reveal cycle, restart the level to get
    a clean board with empties, run find_path_to_unknown, execute the path,
    read the exposed slot, and record it to memory.

    Returns the number of slots successfully revealed.
    """
    revealed = 0
    for i in range(n_reveals):
        print(f"\n  🔬 Intervention reveal {i + 1}/{n_reveals}...")

        # Restart to get a clean board with maximum empties
        if not tap_restart_level(config):
            print("  ⚠ Restart failed — aborting intervention")
            break

        # Screenshot + read
        image = screenshot()
        if image is None:
            print("  ⚠ Screenshot failed — aborting intervention")
            break
        if not is_game_screen(image, config):
            image = wait_for_game_screen(config, timeout=10)
            if image is None:
                print("  ⚠ Game not visible — aborting intervention")
                break

        tubes, valid, seen_colours = read_and_display(
            image, config, capacity, return_colours=True)
        label_to_rgb = {label: rgb for rgb, label in seen_colours.items()}

        # Overlay known slots from memory
        _overlay_learned_colours(
            tubes, memory.get_initial_slots(signature),
            seen_colours, label_to_rgb)

        # Deduce forced slots
        state = tuple(tuple(t) for t in tubes)
        state = deduce_hidden_slots(state, capacity)

        # Find a path to a buried unknown
        path = find_path_to_unknown(state, capacity)
        if not path:
            print(f"  ℹ No path to unknown found — ending intervention")
            break

        print(f"  🧭 Found {len(path)} move(s) to expose a buried slot")

        # Fresh scrcpy window before executing taps
        stop_scrcpy()
        launch_scrcpy()
        refresh_mapping()

        # Execute the path
        execute_move_list(path, config, capacity)

        # Simulate to find which tube has the exposed unknown
        sim_state = state
        exposed_src = None
        exposed_depth = None
        for src, dst, n in path:
            sim_state, _ = apply_move(sim_state, src, dst, capacity)
            if sim_state[src] and sim_state[src][-1] == UNKNOWN:
                exposed_src = src
                exposed_depth = len(sim_state[src]) - 1

        if exposed_src is None:
            print("  ⚠ No unknown exposed after executing path")
            break

        # Read the exposed slot's colour
        rgb = _read_exposed_slot(config, exposed_src, exposed_depth)
        if rgb is None:
            print("  ⚠ Couldn't read exposed slot — ending intervention")
            break

        # Record to memory
        memory.record_slot(signature, exposed_src, exposed_depth, rgb, capacity)

        # Identify the colour label for logging
        label = None
        for name, known_rgb in label_to_rgb.items():
            if colour_distance(rgb, known_rgb) < 5:
                label = name
                break
        if label is None:
            label = f"intervention_{i + 1}"

        print(f"  🧠 Revealed tube {exposed_src + 1} depth {exposed_depth} → {label}")
        revealed += 1

    return revealed


def select_reveal_prefix(state, reveal, capacity, empties):
    """Choose the reveal prefix that best preserves solvability.

    Returns ``reveal`` unchanged (and runs no solvability scoring) when empties
    exceed ``REVEAL_SOLVABILITY_EMPTY_GATE`` — dead-end risk is low there, so the
    behaviour stays identical to before. In the scarce-empty regime, returns the
    shortest prefix that keeps every solvable sampled world solvable; if none is
    fully safe, the highest-scoring prefix (longer wins ties).
    """
    if not reveal or empties > REVEAL_SOLVABILITY_EMPTY_GATE:
        return reveal
    # With many unknowns, sampled solvability is too noisy to be useful
    unknown_count = sum(1 for t in state for s in t if s == UNKNOWN)
    if unknown_count > SCORE_PREFIX_MAX_UNKNOWNS:
        return reveal

    # Sample the solvable base ONCE and score every prefix against the same set
    # (shared common random numbers): each prefix still gets a full sample, but
    # the expensive sampling+solve is done once per round instead of per prefix.
    base = sample_solvable_completions(state, capacity)
    if base is None:                               # unsamplable → no trimming
        return reveal

    scores = {}                                    # memoize per prefix length k
    def score(k):
        if k not in scores:
            scores[k] = score_reveal_batch(state, reveal[:k], capacity, solvable=base)
        return scores[k]

    for k in range(1, len(reveal)):                # prefixes, shortest first
        if score(k) >= 1.0:
            return reveal[:k]                      # shortest fully-safe prefix
    # None fully safe: keep the highest-scoring prefix (longer wins ties via k).
    best_k = max(range(1, len(reveal) + 1), key=lambda k: (score(k), k))
    return reveal[:best_k]


def decide_retry(status, learned, fully_mapped, barren_attempts,
                 attempt, max_attempts, give_up, patience=PATIENCE):
    """Pure end-of-attempt decision: should the level be retried?

    Returns ``(retry, new_barren_attempts, reason)``. ``reason`` is a short tag
    ("heal" / "learned" / "fully_mapped" / "patience" / "give_up") for logging
    and stats. Kept side-effect-free so it can be unit-tested without a device.
    """
    new_barren = 0 if learned > 0 else barren_attempts + 1
    if give_up:
        return False, new_barren, "give_up"

    if status == "heal":
        retry = True            # corruption recovery always retries (once, guarded)
        reason = "heal"
    elif learned > 0:
        retry = True
        reason = "learned"
    elif fully_mapped:
        retry = True
        reason = "fully_mapped"
    elif new_barren < patience:
        retry = True
        reason = "patience"
    else:
        retry = False
        reason = "give_up"

    if retry and attempt >= max_attempts:
        retry = False
        reason = "give_up"
    return retry, new_barren, reason


def wait_for_detectable_tubes(image, existing_config, timeout=15,
                              poll_interval=1.5, min_tubes=3):
    """Detection-based 'game visible' gate. Treats successful tube
    auto-detection as proof the game is on screen. Returns (image, config)
    on success, (None, None) on timeout."""
    elapsed = 0
    while elapsed <= timeout:
        if image is not None:
            config = get_config_for_level(image, existing_config,
                                          fallback_to_existing=False)
            if config and len(config.get("tubes", [])) >= min_tubes:
                return image, config
        print(f"    No tubes detected — retrying in {poll_interval}s...")
        time.sleep(poll_interval)
        elapsed += poll_interval
        image = screenshot()
    return None, None


def solve_one_level(config, tube_capacity, dry_run=False, max_rounds=40, level_num=1):
    """Solve a level, appending the reveal-planner stats to its rounds.txt on exit.

    Thin wrapper around :func:`_solve_one_level_impl` so the stats snapshot is
    written on every exit path (success, failure, or exception) without
    threading a write through the impl's many return points.
    """
    screenshots_dir = Path(__file__).parent / "debug_screenshots" / f"level_{level_num:03d}"
    log_file = screenshots_dir / "rounds.txt"
    try:
        return _solve_one_level_impl(config, tube_capacity, dry_run, max_rounds, level_num)
    finally:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(format_reveal_stats() + "\n")


def _solve_one_level_impl(config, tube_capacity, dry_run=False, max_rounds=40, level_num=1):
    """
    Solve a level with auto-calibration and iterative reveal strategy.

    Learns originally-hidden slot colours across restart attempts (the game is
    deterministic on restart) via a persistent LevelMemory, restarting and
    retrying only while each attempt teaches us something new.
    """
    screenshots_dir = Path(__file__).parent / "debug_screenshots" / f"level_{level_num:03d}"
    log_file = screenshots_dir / "rounds.txt"
    log_file.unlink(missing_ok=True)   # start each level with a fresh log
    initial_saved = False

    memory = LevelMemory()
    signature = None
    capacity = tube_capacity
    config_detected = False
    MAX_ATTEMPTS = 12

    # Per-level state that must survive across attempts.
    healed_once = False          # corruption recovery is allowed at most once
    total_hidden_slots = None    # set from the round-1 raw read
    barren_attempts = 0          # consecutive attempts that learned nothing new

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # TEMP: reproducible measurement runs — remove for prod non-determinism.
        random.seed(level_num * 100 + attempt)
        slots_before = memory.count(signature)
        sim = None
        prev_state = None
        force_park = False
        status = "stuck"
        give_up = False          # set by second-corruption recovery (terminal)
        restarted_for_late_game = False   # restart-for-clean-slate allowed once per attempt
        late_game_intervention_done = False
        seen_signatures = set()
        attempt_moves = []       # flat (src, dst, count) executed this attempt
        attempt_reveals = []     # parallel per-move hidden-slot reveal counts
        last_state = None        # most recent model board, for stuck board_hash
        is_barren_retry = barren_attempts > 0
        barren_path_tried = False

        for round_num in range(1, max_rounds + 1):
            _buf = io.StringIO()
            _orig = sys.stdout
            sys.stdout = _Tee(_orig, _buf)
            try:
                print(f"\n{'─' * 40}")
                print(f"  Attempt {attempt}/{MAX_ATTEMPTS} · Round {round_num}")
                print(f"{'─' * 40}")

                print("📸 Capturing screenshot (ADB)...")
                image = screenshot()

                if image is None:
                    print("  ✗ Screenshot failed.")
                    return False

                if detect_no_more_moves(image):
                    print("  🚫 'No more moves!' detected.")
                    status = "no_moves"
                    break

                # Auto-detect tubes once, on the very first round overall;
                # the level is the same across attempts so reuse thereafter.
                if not config_detected:
                    # First round: successful tube auto-detection is the "game
                    # visible" signal. is_game_screen can't be trusted yet —
                    # config still holds the previous level's geometry, so its
                    # sample points hit tube borders and look like a solid overlay.
                    image, new_config = wait_for_detectable_tubes(image, config)
                    if new_config is None:
                        print("⚠ Game not visible (no tubes detected) — aborting level.")
                        return False
                    config = new_config
                    capacity = config.get("tube_capacity", 4)
                    config_detected = True
                else:
                    # Later rounds: config is correct, so the cheap pixel gate works.
                    if not is_game_screen(image, config):
                        print("⚠ Game not visible — waiting...")
                        image = wait_for_game_screen(config, timeout=15)
                        if image is None:
                            return False

                print("🔍 Reading tubes...")
                tubes, valid, seen_colours = read_and_display(
                    image, config, capacity, return_colours=True)
                label_to_rgb = {label: rgb for rgb, label in seen_colours.items()}

                # Compute the level signature once, from the first raw read.
                if signature is None:
                    signature = LevelMemory.compute_signature(tubes, label_to_rgb, capacity)
                    slots_before = memory.count(signature)
                    # Total hidden slots in the level (from the raw read), so we can
                    # tell when the board is fully mapped and a full-info solve is due.
                    total_hidden_slots = sum(t.count(UNKNOWN) for t in tubes)
                    print(f"  🔑 Level signature {signature[:12]}… "
                          f"({slots_before}/{total_hidden_slots} hidden slot(s) known)")

                # Seed the sim on the first round of the attempt. Reveals are
                # reconciled at the END of each round (against end_image) so the
                # final round's reveals are recorded even when the attempt ends
                # next round before another read (No more moves / stuck).
                if sim is None:
                    sim = AttemptSim(tubes, label_to_rgb,
                                     memory.get_initial_slots(signature))

                # Fill UNKNOWN slots from memory before deducing/solving.
                _overlay_learned_colours(
                    tubes, memory.get_initial_slots(signature),
                    seen_colours, label_to_rgb)

                # Corruption guard: a colour can occupy at most `capacity` slots.
                # If the overlay pushes one past that, recorded memory is provably
                # wrong — heal (delete + relearn) once, then give up if it recurs.
                if not dry_run:
                    overlay_counts = {}
                    for t in tubes:
                        for slot in t:
                            if slot != UNKNOWN:
                                overlay_counts[slot] = overlay_counts.get(slot, 0) + 1
                    overflow = [c for c, n in overlay_counts.items() if n > capacity]
                    if overflow:
                        if not healed_once:
                            print(f"  🧠 Memory looks corrupt ({overflow[0]} exceeds "
                                  f"capacity {capacity}) — deleted, relearning from scratch.")
                            memory.delete(signature)
                            healed_once = True
                            status = "heal"
                            REVEAL_STATS["heal"] += 1
                            break
                        print("  ✗ Corruption persists after one heal — giving up.")
                        status = "stuck"
                        give_up = True
                        break

                # Check if already solved
                all_done = all(
                    len(t) == 0 or (len(t) == capacity and len(set(t)) == 1
                                    and UNKNOWN not in t)
                    for t in tubes
                )
                if all_done:
                    print("\n🎉 Level already complete!")
                    if signature:
                        memory.delete(signature)
                    return True

                has_hidden = has_unknowns(tubes)

                if has_hidden and not initial_saved:
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(screenshots_dir / "initial.png"), image)
                    print(f"  📷 Saved initial screenshot → debug_screenshots/level_{level_num:03d}/initial.png")
                    initial_saved = True

                state = tuple(tuple(t) for t in tubes)

                # Deduce forced unknown slots from colour-count constraints
                state = deduce_hidden_slots(state, capacity)
                has_hidden = any(UNKNOWN in tube for tube in state)

                force_park = prev_state is not None and state == prev_state
                if force_park:
                    print("  ⚠ State unchanged from previous round — forcing park strategy.")
                prev_state = state
                last_state = state       # snapshot for a stuck-attempt board_hash

                if not has_hidden:
                    print("\n🧠 Solving (full information)...")
                    with time_stage("full_solve"):
                        moves = solve(state, tube_capacity=capacity)

                    if moves is None:
                        # Real levels are solvable, so a fully-known board A* can't
                        # solve almost certainly means corrupt overlaid memory.
                        if not dry_run and memory.count(signature) > 0:
                            if not healed_once:
                                print("\n✗ No solution found on a fully-known board — "
                                      "memory likely corrupt; deleted, relearning.")
                                memory.delete(signature)
                                healed_once = True
                                status = "heal"
                                REVEAL_STATS["heal"] += 1
                                break
                            print("\n✗ No solution — corruption persists after one "
                                  "heal — giving up.")
                            give_up = True
                        else:
                            print("\n✗ No solution found!")
                        status = "stuck"
                        break

                    print(f"\n✓ Solution: {len(moves)} moves")
                    if dry_run:
                        for i, (s, d, n) in enumerate(moves, 1):
                            print(f"  {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                        print("\n(Dry run — no taps sent)")
                        return True

                    # Fresh scrcpy window before long solution execution
                    stop_scrcpy()
                    launch_scrcpy()
                    refresh_mapping()

                    execute_move_list(moves, config, capacity)
                    print("\n🎉 Level complete!")
                    if signature:
                        memory.delete(signature)
                    return True

                # Late-game: all remaining unknowns sit at depth 0 (tube bottoms),
                # unreachable by the reveal chain. Sample a completion, A*-solve the
                # full board, and execute it — pausing to read each unknown as a peel
                # exposes it, re-solving from the corrected board each time.
                skip_safe = False
                path_to_unknown = None
                if is_late_game(state) and not dry_run:
                    # If the board has been modified this attempt and we haven't
                    # already restarted for late-game, restart to give A* a clean
                    # board with maximum empty tubes.
                    if not restarted_for_late_game and len(attempt_moves) > 0:
                        print("  🎯 Late-game detected on modified board — "
                              "restarting for clean slate")
                        if not tap_restart_level(config):
                            return False
                        restarted_for_late_game = True
                        seen_signatures.clear()   # fresh board, reset cycle detection
                        prev_state = None          # no "previous state" after restart
                        continue                   # next round: screenshot → overlay → late-game on fresh board

                    print("  🎯 Late-game detected — ≤5 unknowns remain, "
                          "running full-solve strategy")
                    result = run_late_game(state, capacity, memory, signature,
                                           attempt_moves, attempt_reveals, config,
                                           label_to_rgb)
                    if result == "solved":
                        print("\n🎉 Level complete (late-game)!")
                        if signature:
                            memory.delete(signature)
                        return True
                    if result == "dirty":
                        print("  ↻ Late-game stopped early — re-reading board next round.")
                        continue          # board changed; re-screenshot, don't tap on stale state
                    if result == "clean_fail" and not late_game_intervention_done:
                        late_game_intervention_done = True
                        print("  🔬 Late-game failed — running reveal intervention "
                              "to reduce unknowns")
                        n_revealed = run_reveal_intervention(
                            config, capacity, memory, signature)
                        if n_revealed > 0:
                            print(f"  🔬 Intervention revealed {n_revealed} slot(s) "
                                  "— restarting for late-game retry")
                            # Final restart so late-game gets a clean board
                            if not tap_restart_level(config):
                                return False
                            # Reset round state since the board is now fresh
                            sim = None
                            prev_state = None
                            seen_signatures.clear()
                            continue
                        print("  ℹ Intervention revealed nothing — falling through")
                    # "clean_fail": nothing executed, `state` still valid → fall through.
                    print("  ⚠ Late-game couldn't plan — trying path-to-unknown BFS")
                    skip_safe = True                  # A* proved no completion solvable; safe won't either
                    REVEAL_STATS["path_to_unknown_reached"] += 1
                    with time_stage("path_to_unknown"):
                        path_to_unknown = find_path_to_unknown(state, capacity)
                    if path_to_unknown:
                        REVEAL_STATS["path_to_unknown_used"] += 1
                        print(f"  🧭 Path-to-unknown: {len(path_to_unknown)} move(s) "
                              "to expose a buried slot")
                    else:
                        print("  ℹ Path-to-unknown found nothing — falling through (safe skipped)")
                    # fall through to reveal chain (state unchanged); skip_safe gates `safe`

                # Unknowns remain: reclaim parking tubes first, then plan reveal round
                print("\n🔄 Hidden slots remain — planning reveal round...")
                board_sig = _rgb_signature(state, label_to_rgb)
                if board_sig in seen_signatures:
                    print("  ♻ Reveal state already seen this attempt — cycle detected, ending attempt.")
                    status = "stuck"
                    break
                seen_signatures.add(board_sig)
                reclaim = find_reclaim_moves(state, capacity)
                state_mid = state
                for src, dst, _ in reclaim:
                    state_mid, _ = apply_move(state_mid, src, dst, capacity)

                # Reveal chain: guaranteed-safe moves lead (exhaustive endgame);
                # info-gain ranks reveals by uncertainty reduction + outcome memory;
                # maximizer is the info-gain fallback; determinization handles buried
                # unknowns; heuristic park is the genuine last resort.
                reveal = None
                reveal_source = None

                if is_barren_retry and not barren_path_tried and not path_to_unknown:
                    barren_path_tried = True
                    REVEAL_STATS["path_to_unknown_reached"] += 1
                    with time_stage("path_to_unknown"):
                        path_to_unknown = find_path_to_unknown(state_mid, capacity)
                    if path_to_unknown:
                        REVEAL_STATS["path_to_unknown_used"] += 1
                        print(f"  🧭 Barren-retry path-to-unknown: {len(path_to_unknown)} move(s) "
                              "to expose a buried slot")
                    else:
                        print("  ℹ Barren-retry path-to-unknown found nothing — "
                              "falling through to normal cascade")

                if path_to_unknown:
                    # Late-game clean_fail produced a multi-tube peel path; use it
                    # directly and skip the whole planner cascade below.
                    reveal = path_to_unknown
                    reveal_source = "path_to_unknown"
                elif not skip_safe:
                    REVEAL_STATS["safe_reached"] += 1
                    with time_stage("safe"):
                        reveal = find_guaranteed_safe_moves(state_mid, capacity, prev_state=prev_state)
                    if reveal:
                        REVEAL_STATS["safe_used"] += 1
                        reveal_source = "safe"
                        print(f"  🛡 {len(reveal)} guaranteed-safe move(s).")

                if not reveal:
                    REVEAL_STATS["info_gain_reached"] += 1
                    with time_stage("info_gain"):
                        ig_moves, ig_score = plan_reveal_info_gain(
                            state_mid, capacity,
                            attempt_log=memory.get_attempt_log(signature))
                    if ig_moves:
                        REVEAL_STATS["info_gain_used"] += 1
                        reveal = ig_moves
                        reveal_source = "info_gain"
                        print(f"  🧠 Info-gain selected {len(reveal)} reveal "
                              f"move(s) (score: {ig_score:.2f})")
                    else:
                        # Info-gain saw no directly-exposable unknown; try digging
                        # 2–3 layers into one tube to surface a buried unknown.
                        print("  ℹ Info-gain found no candidates — trying deep-reveal")
                        REVEAL_STATS["deep_reached"] += 1
                        with time_stage("deep"):
                            deep_moves, deep_score = plan_reveal_deep(
                                state_mid, capacity,
                                attempt_log=memory.get_attempt_log(signature))
                        if deep_moves:
                            REVEAL_STATS["deep_used"] += 1
                            reveal = deep_moves
                            reveal_source = "deep"
                            print(f"  ⛏ Deep-reveal: {len(deep_moves)}-move dig into "
                                  f"tube {deep_moves[0][0] + 1} (score: {deep_score:.2f})")
                        else:
                            print("  ℹ Deep-reveal found no buried targets — falling through")
                            reveal = []
                            # Consolidation: deep-reveal couldn't dig. If buried
                            # unknowns remain but empties are the bottleneck, sort
                            # known-only tubes to free a third empty so next
                            # round's deep-reveal can reach depth-0 unknowns.
                            empties_mid = sum(1 for t in state_mid if len(t) == 0)
                            buried = any(UNKNOWN in t and not _exposes_unknown(t)
                                         for t in state_mid)
                            if buried and empties_mid < 3:
                                REVEAL_STATS["consolidate_reached"] += 1
                                with time_stage("consolidate"):
                                    cons_moves, freed = plan_consolidate(
                                        state_mid, capacity)
                                if cons_moves:
                                    REVEAL_STATS["consolidate_used"] += 1
                                    reveal = cons_moves
                                    reveal_source = "consolidate"
                                    print(f"  🔧 Consolidate: {len(cons_moves)} moves "
                                          f"to free {freed} tube(s) for deep dig")
                                else:
                                    print("  ℹ Consolidation found no path to free "
                                          "tubes — falling through")
                            if not reveal:
                                REVEAL_STATS["maximize_reached"] += 1
                                with time_stage("maximize"):
                                    reveal = plan_reveal_maximize(state_mid, capacity, prev_state=prev_state)
                                if reveal:
                                    REVEAL_STATS["maximize_used"] += 1
                                    reveal_source = "maximize"
                                else:
                                    REVEAL_STATS["determinization_reached"] += 1
                                    with time_stage("determinization"):
                                        reveal = pick_best_move_by_determinization(state_mid, capacity)
                                    if reveal:
                                        REVEAL_STATS["determinization_used"] += 1
                                        reveal_source = "determinization"
                                    else:
                                        print("  ℹ Maximizer + determinization empty — heuristic park (last resort)...")
                                        REVEAL_STATS["reveal_round_reached"] += 1
                                        with time_stage("reveal_round"):
                                            reveal = plan_reveal_round(state_mid, capacity, force_park=force_park, prev_state=prev_state)
                                        if reveal:
                                            REVEAL_STATS["reveal_round_used"] += 1
                                            reveal_source = "reveal_round"

                # When empties are scarce a reveal batch can be executable yet
                # still strand the board (e.g. spend the last empty). Keep the
                # prefix that best preserves solvability. No-op (and no extra
                # solves) when empties are plentiful — see select_reveal_prefix.
                # Only score prefixes for strategic planners (info-gain, deep-reveal).
                # Safe is already solvability-checked; consolidate sorts known tubes
                # (not a reveal); maximize/determinization/reveal_round are heuristic
                # fallbacks where the 6s scoring cost exceeds the value; path_to_unknown
                # is a shortest-path peel we want executed whole, not trimmed.
                SCORE_WORTHY = {"info_gain", "deep"}
                empties_now = sum(1 for t in state_mid if len(t) == 0)
                if reveal_source in SCORE_WORTHY:
                    with time_stage("score_prefix"):
                        trimmed = select_reveal_prefix(state_mid, reveal, capacity, empties_now)
                    if len(trimmed) < len(reveal):
                        print(f"  🔒 Reveal trimmed {len(reveal)}→{len(trimmed)} move(s) "
                              f"to preserve solvability ({empties_now} empt"
                              f"{'y' if empties_now == 1 else 'ies'}).")
                    reveal = trimmed
                elif reveal_source:
                    print(f"  ⏭ Skipping solvability scoring for {reveal_source} moves")

                all_moves = reclaim + reveal

                if not all_moves:
                    print("  ⚠ New strategy found no moves — trying legacy fallback...")
                    all_moves = find_safe_moves(tubes, capacity, prev_state=prev_state)
                    if all_moves:
                        print(f"  {len(all_moves)} fallback moves")

                # Validate the planned batch against the board before executing:
                # the game rejects a pour onto a freshly-revealed (UNKNOWN) top,
                # so truncate at the first move it couldn't execute. The same
                # validated prefix is what we execute AND mirror onto the sim,
                # keeping origin-tracking in lock-step with the device.
                executed_moves = validate_move_sequence(state, all_moves, capacity)

                if not executed_moves:
                    empties_now = [i for i, t in enumerate(state) if len(t) == 0]
                    if not empties_now and prev_state is not None:
                        print("  ✗ Deadlocked: 0 empty tubes and all candidate moves would reverse the prior state.")
                    else:
                        print(f"  ✗ Round {round_num}: stuck — no moves available.")
                    status = "stuck"
                    break

                print(f"  {len(reclaim)} reclaim + {len(reveal)} reveal moves")
                if len(executed_moves) < len(all_moves):
                    print(f"  ✂ Truncated batch to {len(executed_moves)}/{len(all_moves)} "
                          f"executable move(s) — later move would pour onto a hidden top.")
                if dry_run:
                    for i, (s, d, n) in enumerate(executed_moves, 1):
                        print(f"    {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                    print("  (Dry run — continuing to next round)")
                else:
                    execute_move_list(executed_moves, config, capacity)

                # Mirror the executed moves onto the sim so the next round's
                # read can be reconciled against it.
                for src, dst, n in executed_moves:
                    sim.apply_move(src, dst, n)

                print("\n  ⏳ Waiting for animations...")
                time.sleep(1.5)

                end_image = screenshot()
                if end_image is not None:
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(screenshots_dir / f"round_{round_num:02d}_end.png"), end_image)
                    print(f"  📷 Saved round {round_num} end screenshot → debug_screenshots/level_{level_num:03d}/round_{round_num:02d}_end.png")

                # Reconcile against the post-move board now, so reveals are
                # recorded the same round they happen — even if the attempt ends
                # next round (No more moves / stuck) before another read.
                revealed_count = 0
                if not dry_run and end_image is not None and sim is not None and sim.valid:
                    end_tubes, end_seen = read_tubes(end_image, config, return_colours=True)
                    end_label_to_rgb = {label: rgb for rgb, label in end_seen.items()}

                    # Poisoned-read guard: when this batch freshly exposed a hidden
                    # slot, majority-vote 3 reads instead of trusting one. ADB shots
                    # are deterministic, so a settled board yields identical RGB
                    # across reads; inconsistent mid-animation pixels lose the
                    # per-slot 2-of-3 vote and are dropped (that slot only). Pay the
                    # extra screenshots only when a hidden slot was actually exposed.
                    if _batch_exposed_unknown(state, executed_moves, capacity):
                        print("  🗳 Confirm read 1/3...")
                        reads = [(end_tubes, end_label_to_rgb)]
                        for i in range(2):
                            time.sleep(0.15)
                            vote_image = screenshot()
                            if vote_image is not None:
                                print(f"  🗳 Confirm read {2 + i}/3...")
                                vote_tubes_n, vote_seen = read_tubes(
                                    vote_image, config, return_colours=True)
                                reads.append((vote_tubes_n,
                                              {label: rgb for rgb, label in vote_seen.items()}))
                        reconcile_tubes, reconcile_l2r = _majority_read(reads)
                    else:
                        reconcile_tubes, reconcile_l2r = end_tubes, end_label_to_rgb

                    revealed = sim.reconcile(reconcile_tubes, reconcile_l2r)
                    for origin, rgb in revealed:
                        memory.record_slot(signature, origin[0], origin[1], rgb, capacity)
                        print(f"  🆕 Recorded origin ({origin[0]},{origin[1]}) → RGB{rgb}")
                    revealed_count = len(revealed)
                    if not sim.valid:
                        print("  ⚠ Sim desynced from board — stopped attributing reveals.")

                # Accumulate this round's executed moves into the attempt log,
                # attributing the round's revealed-slot count across the moves that
                # actually uncovered a hidden top (others get 0). Even integer split
                # keeps sum(attempt_reveals) == total slots revealed this attempt.
                if not dry_run and executed_moves:
                    sim_state = state
                    reveal_idxs = []
                    for idx, (src, dst, n) in enumerate(executed_moves):
                        sim_state, _ = apply_move(sim_state, src, dst, capacity)
                        if sim_state[src] and sim_state[src][-1] == UNKNOWN:
                            reveal_idxs.append(idx)
                    per_move = [0] * len(executed_moves)
                    if reveal_idxs and revealed_count:
                        base, extra = divmod(revealed_count, len(reveal_idxs))
                        for j, idx in enumerate(reveal_idxs):
                            per_move[idx] = base + (1 if j < extra else 0)
                    elif revealed_count:
                        per_move[-1] = revealed_count   # fallback: credit last move
                    attempt_moves.extend(executed_moves)
                    attempt_reveals.extend(per_move)

                if not dry_run:
                    print("  🔄 Restarting scrcpy...")
                    stop_scrcpy()
                    launch_scrcpy()
                    refresh_mapping()

            finally:
                sys.stdout = _orig
                if initial_saved:
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*40}\n  ATTEMPT {attempt} · ROUND {round_num}\n{'='*40}\n")
                        f.write(_buf.getvalue())
        else:
            print(f"\n✗ Couldn't solve within {max_rounds} rounds this attempt.")

        # ── Attempt finished without solving — decide whether to retry ──
        # Tee this summary into rounds.txt too: the round loop's finally above
        # already restored sys.stdout, so without this the 📝/📚/↩/✗ verdicts
        # would print to the terminal but never reach the log.
        _buf = io.StringIO()
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, _buf)
        try:
            slots_after = memory.count(signature)
            learned = slots_after - slots_before
            if status == "no_moves":
                print("  🚫 Attempt ended on 'No more moves!'.")
            print(f"  📚 Learned {learned} new hidden slot(s) this attempt "
                  f"({slots_after} known total).")

            # Record dead-end outcomes so the info-gain scorer can steer away from
            # paths that previously stranded the board (advisory; penalty decays as
            # more slots become known). Solved/heal attempts are not recorded.
            if not dry_run and signature and status in ("stuck", "no_moves"):
                board_hash = _hash_board(last_state)
                memory.record_attempt(signature, attempt_moves, attempt_reveals,
                                       status, board_hash)
                total_str = (f"{slots_after}/{total_hidden_slots}"
                             if total_hidden_slots is not None else str(slots_after))
                print(f"  📝 Recorded attempt {attempt} (outcome: {status}, "
                      f"reveals: {sum(attempt_reveals)}, total_known: {total_str})")

            if dry_run:
                return False

            # A fully-mapped board's next attempt becomes a full-information solve, so
            # always grant one more attempt (let #1's self-heal handle a still-unsolved
            # board). Patience tolerates a few barren attempts because the maximizer's
            # randomized tie-breaking makes each retry explore a divergent path.
            fully_mapped = (total_hidden_slots is not None
                            and slots_after >= total_hidden_slots)
            retry, barren_attempts, reason = decide_retry(
                status, learned, fully_mapped, barren_attempts,
                attempt, MAX_ATTEMPTS, give_up)

            if retry:
                reason_msg = {
                    "heal": "healing corrupt memory",
                    "learned": f"learned {learned} new slot(s)",
                    "fully_mapped": "board fully mapped",
                    "patience": f"patience {barren_attempts}/{PATIENCE}",
                }[reason]
                if reason == "patience":
                    REVEAL_STATS["patience_retry"] += 1
                print(f"  ↩ Restarting level to retry ({reason_msg}) "
                      f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...")
                if not tap_restart_level(config):
                    return False
                continue

            if give_up:
                print("  ✗ Giving up (memory corruption unrecoverable).")
            elif attempt >= MAX_ATTEMPTS:
                print(f"  ✗ Reached max attempts ({MAX_ATTEMPTS}) — giving up.")
            else:
                print(f"  ✗ No new slots learned in {PATIENCE} attempt(s) — giving up.")
            return False
        finally:
            sys.stdout = _orig
            if initial_saved:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*40}\n  ATTEMPT {attempt} · SUMMARY\n{'='*40}\n")
                    f.write(_buf.getvalue())

    return False


def tap_next_level(config, wait_time=3.0):
    """Tap Next Level button."""
    btn = config.get("next_button")
    if not btn:
        print("\n⚠ No 'next level' button configured — trying auto-detection...")
        fallback_img = screenshot()
        buttons = detect_buttons(fallback_img) if fallback_img is not None else {}
        if "next" in buttons:
            bx, by = buttons["next"]
            btn = {"x": bx, "y": by}
            config["next_button"] = btn
            save_config(config)
            print(f"  ✓ Auto-detected next_button at ({bx}, {by}).")
        else:
            print("  Run: python main.py --set-next-button")
            return False

    x, y = btn["x"], btn["y"]
    print(f"\n⏳ Waiting {wait_time}s for win animation...")
    time.sleep(wait_time)

    for attempt in range(3):
        print(f"👆 Tapping 'Next Level' at ({x}, {y})...")
        adb_tap(x, y)
        time.sleep(wait_time)

        image = screenshot()
        if image is not None and is_game_screen(image, config):
            print("  ✓ Game screen detected!")
            return True

        print(f"  Not visible (attempt {attempt+1}/3) — retrying...")
        time.sleep(1.5)

    image = wait_for_game_screen(config, timeout=20)
    if image is not None:
        return True

    print("  ⚠ Could not confirm next level loaded.")
    return False


def set_next_button():
    """Quick one-time setup: just set the Next Level button position."""
    print("Taking screenshot...")
    img = screenshot()
    if img is None:
        print("  ✗ Screenshot failed.")
        return

    print("Click the 'Next Level' button, then press any key.")
    btn_pos = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn_pos.append((x, y))
            cv2.circle(img, (x, y), 12, (255, 0, 255), 3)
            cv2.imshow("Set Next Button", img)
            print(f"  ✓ Button position: ({x}, {y})")

    cv2.namedWindow("Set Next Button", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Set Next Button", 540, 960)
    cv2.imshow("Set Next Button", img)
    cv2.setMouseCallback("Set Next Button", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if btn_pos:
        config = load_config() or {}
        config["next_button"] = {"x": btn_pos[0][0], "y": btn_pos[0][1]}
        save_config(config)
        print(f"  Saved next button at ({btn_pos[0][0]}, {btn_pos[0][1]})")


def tap_restart_level(config):
    """Tap the Restart button on the 'No more moves!' screen."""
    btn = config.get("restart_button")
    if not btn:
        print("  ⚠ No restart_button in config. It should have been auto-detected "
              "during round-1 calibration — the calibration screenshot may have "
              "been bad. Not detecting from the 'No more moves!' screen because "
              "the hand-icon overlay interferes with button detection. Run "
              "`python main.py --set-restart-button` to set it manually.")
        return False
    x, y = btn["x"], btn["y"]
    print(f"  🔄 Tapping restart at ({x}, {y})...")
    adb_tap(x, y)
    time.sleep(2.0)
    return True


def set_restart_button():
    """Quick one-time setup: set the Restart button position."""
    print("Navigate the game to a 'No more moves!' screen, then run this command.")
    print("Taking screenshot...")
    img = screenshot()
    if img is None:
        print("  ✗ Screenshot failed.")
        return

    print("Click the 'Restart' button, then press any key.")
    btn_pos = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn_pos.append((x, y))
            cv2.circle(img, (x, y), 12, (0, 165, 255), 3)
            cv2.imshow("Set Restart Button", img)
            print(f"  ✓ Button position: ({x}, {y})")

    cv2.namedWindow("Set Restart Button", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Set Restart Button", 540, 960)
    cv2.imshow("Set Restart Button", img)
    cv2.setMouseCallback("Set Restart Button", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if btn_pos:
        config = load_config() or {}
        config["restart_button"] = {"x": btn_pos[0][0], "y": btn_pos[0][1]}
        save_config(config)
        print(f"  Saved restart button at ({btn_pos[0][0]}, {btn_pos[0][1]})")


def main():
    parser = argparse.ArgumentParser(description="Water Sort Puzzle Bot")
    parser.add_argument("--calibrate", action="store_true",
                        help="Manual calibration (usually not needed)")
    parser.add_argument("--set-next-button", action="store_true",
                        help="Set the Next Level button position (one-time)")
    parser.add_argument("--set-restart-button", action="store_true",
                        help="Set the Restart button position (shown on 'No more moves' screen)")
    parser.add_argument("--test-detect", action="store_true",
                        help="Test auto-detection and save visualisation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solve without tapping")
    parser.add_argument("--loop", action="store_true",
                        help="Auto-solve levels (Ctrl+C to stop)")
    parser.add_argument("--levels", type=int, default=0,
                        help="Max levels in loop mode (0 = unlimited)")
    parser.add_argument("--win-wait", type=float, default=3.0,
                        help="Seconds for win animation (default: 3.0)")
    args = parser.parse_args()

    # ── Manual calibrate ────────────────────────────────────────────
    if args.calibrate:
        calibrate()
        return

    # ── Set next button ─────────────────────────────────────────────
    if args.set_next_button:
        set_next_button()
        return

    # ── Set restart button ───────────────────────────────────────────
    if args.set_restart_button:
        set_restart_button()
        return

    # ── Test auto-detection ─────────────────────────────────────────
    if args.test_detect:
        print("📸 Taking ADB screenshot...")
        img = screenshot()
        if img is None:
            print("  ✗ Failed.")
            return

        config = auto_calibrate(img)
        if config:
            vis = visualise_detection(img, config)
            vis_path = "detection_test.png"
            cv2.imwrite(vis_path, vis)
            print(f"\n  Saved visualisation to {vis_path}")
            print("  Open it to check if tubes were detected correctly.")

            # Also do a test read
            print("\n🔍 Test reading:")
            tubes = read_tubes(img, config)
            for i, tube in enumerate(tubes):
                label = " | ".join(tube) if tube else "(empty)"
                print(f"  Tube {i+1}: [{label}]")
        return

    # ── Load existing config (for next_button) ──────────────────────
    config = load_config() or {}
    tube_capacity = config.get("tube_capacity", 4)

    # Reset level memory for a fresh start
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()
        print("🧹 Reset level memory (level_memory.json)")

    # ── Launch scrcpy ───────────────────────────────────────────────
    if not args.dry_run:
        print("🖥️  Launching scrcpy...")
        if not launch_scrcpy():
            sys.exit(1)

    # Test ADB screenshot
    print("📸 Testing ADB capture...")
    test_img = screenshot()
    if test_img is not None:
        h, w = test_img.shape[:2]
        print(f"  ✓ ADB screenshot working ({w}×{h})\n")
    else:
        print("  ✗ ADB screenshot failed.\n")
        if not args.dry_run:
            stop_scrcpy()
        sys.exit(1)

    # ── Single level ────────────────────────────────────────────────
    if not args.loop:
        if not args.dry_run:
            input("Press Enter to start...")
        try:
            solve_one_level(config, tube_capacity, args.dry_run)
        finally:
            print_reveal_stats()
            if not args.dry_run:
                stop_scrcpy()
        return

    # ── Loop mode ───────────────────────────────────────────────────
    if "next_button" not in config:
        print("⚠ Loop mode needs the 'Next Level' button position.")
        print("  Run: python main.py --set-next-button\n")
        if not args.dry_run:
            stop_scrcpy()
        sys.exit(1)

    level = 0
    max_levels = args.levels if args.levels > 0 else float("inf")
    failures = 0
    max_failures = 3

    print("═══ LOOP MODE ═══")
    print(f"  Solving {'unlimited' if args.levels == 0 else args.levels} levels")
    print("  Auto-detecting tubes each level")
    print("  Press Ctrl+C to stop\n")

    input("Press Enter to start...")

    try:
        while level < max_levels:
            level += 1
            print(f"\n{'═' * 50}")
            print(f"  LEVEL {level}")
            print(f"{'═' * 50}")

            success = solve_one_level(config, tube_capacity, level_num=level)

            if success:
                failures = 0
                if not tap_next_level(config, wait_time=args.win_wait):
                    break
            else:
                failures += 1
                print(f"\n  ({failures}/{max_failures} consecutive failures)")
                if failures >= max_failures:
                    print("\nToo many failures — stopping.")
                    break
                print("  Retrying in 3s...")
                time.sleep(3)
                level -= 1

    except KeyboardInterrupt:
        print(f"\n\nStopped after {level - 1} levels.")
        print_reveal_stats()

    print(f"\n🏁 Done! Solved {level - failures} levels.")
    print_reveal_stats()
    stop_scrcpy()


if __name__ == "__main__":
    main()
