# Water Sort Bot

Automated solver for Water Sort / Ball Sort Android puzzle games. Mirrors the phone screen via scrcpy, captures screenshots via ADB, uses computer vision to detect tube positions and read colours, solves the puzzle with A*/DFS search, then executes tap moves on the scrcpy window.

## Tech Stack

- **Python 3** — no build system; run directly with `python main.py`
- **OpenCV (cv2)** — image processing, border detection, calibration UI
- **NumPy** — mask operations in `auto_calibrate.py`
- **pyautogui + pygetwindow** — clicking on the scrcpy window (Windows)
- **ADB** (external) — pixel-perfect screenshots at native device resolution
- **scrcpy** (external) — live screen mirror; forwards mouse clicks as touch

## Key Directories / Files

```
water_sort_bot/
├── main.py            # CLI entry point; pipeline orchestration and loop mode
├── solver.py          # Pure A*/DFS solver — no I/O, no project imports
├── screen_reader.py   # Colour reading and hidden-slot detection from screenshots
├── auto_calibrate.py  # Vision-based tube geometry detection
├── capture.py         # ADB screenshot capture and scrcpy process management
├── automator.py       # Coordinate mapping and tap execution via pyautogui
└── config.json        # Auto-generated calibration data (tube positions, empty colour)
```

`config.json` is the runtime contract between stages — its `tubes[].sample_points` is consumed by `screen_reader.py:92`, `auto_calibrate.py:255`, and `automator.py:103`.

## Setup

```bash
pip install opencv-python numpy pyautogui pygetwindow mss
# Install ADB and add to PATH; install scrcpy
# Update SCRCPY_PATH in capture.py:20 to your scrcpy.exe location
```

Connect your Android device via USB with USB debugging enabled, then verify with `adb devices`.

## Commands

```bash
# One-time setup: set the Next Level button (needed for --loop)
python main.py --set-next-button

# Solve a single level
python main.py

# Auto-solve unlimited levels
python main.py --loop

# Auto-solve N levels
python main.py --loop --levels 50

# Solve without sending taps (debug)
python main.py --dry-run

# Test tube auto-detection; saves detection_test.png
python main.py --test-detect

# Manual calibration fallback (rarely needed)
python main.py --calibrate
```

Optional flags: `--delay <secs>` (inter-move delay), `--win-wait <secs>` (wait after win animation, default 3.0).

## Hidden Slot Strategy

Unrevealed slots are marked `UNKNOWN` (`solver.py:17`). When any unknowns are present, A* is skipped entirely — partial information makes it unreliable. Instead, `find_safe_moves()` (`solver.py:225`) applies a reveal strategy each round, re-screenshots, and retries until the board is fully visible, then solves once with complete information. Up to `max_rounds=10` iterations are allowed (`main.py:109`).

`find_safe_moves()` branches on empty-tube count:

- **2+ empties** — park the least-occurring visible colour into one empty (maximises hidden-slot reclaim), then consolidate the most-occurring colour into a second empty (maximises reveals per empty spent). Re-surveys the board between steps.
- **1 empty** — first look for free matching pours (same top colour, space available) that reveal slots without spending the empty; if none exist, park the least-occurring colour into the empty.
- **0 empties** — matching visible-top pours only (only option available).

Ties in occurrence count are broken randomly.

## Additional Documentation

- [`.claude/docs/architectural_patterns.md`](.claude/docs/architectural_patterns.md) — pipeline architecture, fallback chains, UNKNOWN sentinel, coordinate abstraction, module-level caches, solver state representation
