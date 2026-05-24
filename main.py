#!/usr/bin/env python3
"""
Water Sort Bot — ADB screenshots + scrcpy taps + auto-calibration.

Auto-detects tube positions from each screenshot. No manual calibration
needed except for the 'Next Level' button (one-time).

Supports hidden/unknown slots with iterative solve-reveal-solve.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from solver import solve, find_safe_moves, UNKNOWN
from screen_reader import (
    read_tubes, has_unknowns, load_config, save_config, calibrate,
    is_game_screen, wait_for_game_screen, CONFIG_PATH,
)
from capture import screenshot, launch_scrcpy, stop_scrcpy
from automator import (
    adb_tap, human_delay, jittered_tap,
    get_tube_tap_zones, refresh_mapping,
)
from auto_calibrate import auto_calibrate, visualise_detection


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

    # Preserve next_button from existing config
    if existing_config and "next_button" in existing_config:
        new_config["next_button"] = existing_config["next_button"]

    # Save for debugging
    save_config(new_config)

    return new_config


def solve_one_level(config, tube_capacity, dry_run=False, max_rounds=10):
    """
    Solve a level with auto-calibration and iterative reveal strategy.
    """
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'─' * 40}")
        print(f"  Round {round_num}")
        print(f"{'─' * 40}")

        print("📸 Capturing screenshot (ADB)...")
        image = screenshot()

        if image is None:
            print("  ✗ Screenshot failed.")
            return False

        if not is_game_screen(image, config):
            print("⚠ Game not visible — waiting...")
            image = wait_for_game_screen(config, timeout=15)
            if image is None:
                return False

        # Auto-detect tubes for this frame
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

        if not has_hidden:
            print("\n🧠 Solving (full information)...")
            moves = solve(tubes, tube_capacity=tube_capacity)

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
        else:
            print("\n🧠 Attempting solve with hidden slots...")
            moves = solve(tubes, tube_capacity=tube_capacity)

            if moves is not None:
                print(f"\n✓ Full solution found: {len(moves)} moves")
                if dry_run:
                    for i, (s, d, n) in enumerate(moves, 1):
                        print(f"  {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                    print("\n(Dry run — no taps sent)")
                    return True

                execute_move_list(moves, config, tube_capacity)
                print("\n🎉 Level complete!")
                return True

            print("  Finding safe moves to reveal hidden slots...")
            safe = find_safe_moves(tubes, tube_capacity)

            if not safe:
                print("  ✗ No safe moves found. Stuck.")
                return False

            print(f"\n  Playing {len(safe)} safe moves:")
            if dry_run:
                for i, (s, d, n) in enumerate(safe, 1):
                    print(f"    {i}. Tube {s+1} → Tube {d+1} ({n} poured)")
                print("  (Dry run — would re-screenshot and continue)")
                return True

            execute_move_list(safe, config, tube_capacity)
            print("\n  ⏳ Waiting for animations...")
            time.sleep(1.5)

    print(f"\n✗ Couldn't solve after {max_rounds} rounds.")
    return False


def tap_next_level(config, wait_time=3.0):
    """Tap Next Level button."""
    btn = config.get("next_button")
    if not btn:
        print("\n⚠ No 'next level' button configured.")
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


def main():
    parser = argparse.ArgumentParser(description="Water Sort Puzzle Bot")
    parser.add_argument("--calibrate", action="store_true",
                        help="Manual calibration (usually not needed)")
    parser.add_argument("--set-next-button", action="store_true",
                        help="Set the Next Level button position (one-time)")
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

            success = solve_one_level(config, tube_capacity)

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
