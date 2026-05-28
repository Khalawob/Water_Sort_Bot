#!/usr/bin/env python3
"""
Water Sort Bot — ADB screenshots + scrcpy taps + auto-calibration.

Auto-detects tube positions from each screenshot. No manual calibration
needed except for the 'Next Level' button (one-time).

Supports hidden/unknown slots with iterative solve-reveal-solve.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2

from solver import (
    solve, plan_reveal_round, find_reclaim_moves, find_safe_moves,
    apply_move, UNKNOWN, deduce_hidden_slots,
    pick_best_move_by_determinization,
)
from screen_reader import (
    detect_no_more_moves,
    read_tubes, has_unknowns, load_config, save_config, calibrate,
    is_game_screen, wait_for_game_screen, CONFIG_PATH,
)
from capture import screenshot, launch_scrcpy, stop_scrcpy
from automator import (
    adb_tap, human_delay, jittered_tap,
    get_tube_tap_zones, refresh_mapping,
)
from auto_calibrate import auto_calibrate, visualise_detection, detect_buttons


def read_and_display(image, config, tube_capacity):
    """Read tubes, print them, return (tubes, is_valid)."""
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


def get_config_for_level(image, existing_config=None):
    """
    Auto-detect tubes from the screenshot.
    Preserves next_button from existing config if available.
    """
    print("🔎 Auto-detecting tubes...")
    new_config = auto_calibrate(image)

    if new_config is None:
        if existing_config:
            print("  ⚠ Auto-detection failed, using saved config.")
            return existing_config
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


def solve_one_level(config, tube_capacity, dry_run=False, max_rounds=10, level_num=1):
    """
    Solve a level with auto-calibration and iterative reveal strategy.
    """
    screenshots_dir = Path(__file__).parent / "debug_screenshots" / f"level_{level_num:03d}"
    log_file = screenshots_dir / "rounds.txt"
    initial_saved = False
    prev_state = None
    force_park = False
    restart_count = 0

    for round_num in range(1, max_rounds + 1):
        _buf = io.StringIO()
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, _buf)
        try:
            print(f"\n{'─' * 40}")
            print(f"  Round {round_num}")
            print(f"{'─' * 40}")

            print("📸 Capturing screenshot (ADB)...")
            image = screenshot()

            if image is None:
                print("  ✗ Screenshot failed.")
                return False

            if detect_no_more_moves(image):
                print("  🚫 'No more moves!' detected.")
                if dry_run:
                    return False
                restart_count += 1
                if restart_count > 3:
                    print(f"  ✗ Too many restarts ({restart_count}) — giving up.")
                    return False
                print(f"  ↩ Restart attempt {restart_count}/3...")
                if not tap_restart_level(config):
                    return False
                prev_state = None
                force_park = False
                continue

            if not is_game_screen(image, config):
                print("⚠ Game not visible — waiting...")
                image = wait_for_game_screen(config, timeout=15)
                if image is None:
                    return False

            # Auto-detect tubes only on round 1; reuse config for subsequent rounds
            if round_num == 1:
                config = get_config_for_level(image, config)
                if config is None:
                    return False
                tube_capacity = config.get("tube_capacity", 4)

            print("🔍 Reading tubes...")
            tubes, valid = read_and_display(image, config, tube_capacity)

            # Check if already solved
            all_done = all(
                len(t) == 0 or (len(t) == tube_capacity and len(set(t)) == 1
                                and UNKNOWN not in t)
                for t in tubes
            )
            if all_done:
                print("\n🎉 Level already complete!")
                return True

            has_hidden = has_unknowns(tubes)

            if has_hidden and not initial_saved:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(screenshots_dir / "initial.png"), image)
                print(f"  📷 Saved initial screenshot → debug_screenshots/level_{level_num:03d}/initial.png")
                initial_saved = True

            state = tuple(tuple(t) for t in tubes)

            # Deduce forced unknown slots from colour-count constraints
            state = deduce_hidden_slots(state, tube_capacity)
            has_hidden = any(UNKNOWN in tube for tube in state)

            force_park = prev_state is not None and state == prev_state
            if force_park:
                print("  ⚠ State unchanged from previous round — forcing park strategy.")
            prev_state = state

            if not has_hidden:
                print("\n🧠 Solving (full information)...")
                moves = solve(state, tube_capacity=tube_capacity)

                if moves is None:
                    print("\n✗ No solution found!")
                    return False

                print(f"\n✓ Solution: {len(moves)} moves")
                if dry_run:
                    for i, (s, d, n) in enumerate(moves, 1):
                        print(f"  {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                    print("\n(Dry run — no taps sent)")
                    return True

                execute_move_list(moves, config, tube_capacity)
                print("\n🎉 Level complete!")
                return True

            # Unknowns remain: reclaim parking tubes first, then plan reveal round
            print("\n🔄 Hidden slots remain — planning reveal round...")
            reclaim = find_reclaim_moves(state, tube_capacity)
            state_mid = state
            for src, dst, _ in reclaim:
                state_mid, _ = apply_move(state_mid, src, dst, tube_capacity)

            reveal = pick_best_move_by_determinization(state_mid, tube_capacity)
            if not reveal:
                print("  ℹ Determinization found no moves — falling back to heuristic reveal...")
                reveal = plan_reveal_round(state_mid, tube_capacity, force_park=force_park, prev_state=prev_state)
            all_moves = reclaim + reveal

            if not all_moves:
                print("  ⚠ New strategy found no moves — trying legacy fallback...")
                fallback = find_safe_moves(tubes, tube_capacity, prev_state=prev_state)
                if not fallback:
                    empties_now = [i for i, t in enumerate(state) if len(t) == 0]
                    if not empties_now and prev_state is not None:
                        print("  ✗ Deadlocked: 0 empty tubes and all candidate moves would reverse the prior state.")
                    else:
                        print(f"  ✗ Round {round_num}: stuck — no moves available.")
                    return False
                print(f"  {len(fallback)} fallback moves")
                if not dry_run:
                    execute_move_list(fallback, config, tube_capacity)
                else:
                    for i, (s, d, n) in enumerate(fallback, 1):
                        print(f"    {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                    print("  (Dry run — continuing to next round)")

            print(f"  {len(reclaim)} reclaim + {len(reveal)} reveal moves")
            if dry_run:
                for i, (s, d, n) in enumerate(all_moves, 1):
                    print(f"    {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                print("  (Dry run — continuing to next round)")
            else:
                execute_move_list(all_moves, config, tube_capacity)

            print("\n  ⏳ Waiting for animations...")
            time.sleep(1.5)

            end_image = screenshot()
            if end_image is not None:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(screenshots_dir / f"round_{round_num:02d}_end.png"), end_image)
                print(f"  📷 Saved round {round_num} end screenshot → debug_screenshots/level_{level_num:03d}/round_{round_num:02d}_end.png")

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
                    f.write(f"\n{'='*40}\n  ROUND {round_num}\n{'='*40}\n")
                    f.write(_buf.getvalue())

    print(f"\n✗ Couldn't solve after {max_rounds} rounds.")
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

    print(f"\n🏁 Done! Solved {level - failures} levels.")
    stop_scrcpy()


if __name__ == "__main__":
    main()
