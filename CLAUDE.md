# Water Sort Bot

Automated solver for Water Sort / Ball Sort Android puzzle games. Mirrors the phone screen via scrcpy, captures screenshots via ADB, uses computer vision to detect tube positions and read colours, solves the puzzle with A*/DFS search, then executes tap moves on the scrcpy window.

## Tech Stack

- **Python 3** -- no build system; run directly with `python main.py`
- **OpenCV (cv2)** -- image processing, border detection, calibration UI
- **NumPy** -- mask operations in `auto_calibrate.py`
- **pyautogui + pygetwindow** -- clicking on the scrcpy window (Windows)
- **ADB** (external) -- pixel-perfect screenshots at native device resolution
- **scrcpy** (external) -- live screen mirror; forwards mouse clicks as touch

## Key Directories / Files

```
water_sort_bot/
├── main.py            # CLI entry point; pipeline orchestration, multi-attempt retry, reveal chain
├── solver.py          # Pure A*/DFS solver + 10 reveal planners -- no I/O, no project imports
├── screen_reader.py   # Colour reading, hidden-slot detection, popup detection
├── auto_calibrate.py  # Vision-based tube geometry detection, button/win/popup detection
├── capture.py         # ADB screenshot capture and scrcpy process management
├── automator.py       # Coordinate mapping and tap execution via pyautogui
├── level_memory.py    # Cross-restart learning: persists discovered hidden-slot colours
├── config.json        # Auto-generated calibration data (see below)
└── tests/             # Unit tests (pytest)
```

`config.json` is the runtime contract between stages. Key consumers of `tubes[].sample_points`:
- Produced by `auto_calibrate.py:276`
- Read by `screen_reader.py:113` and `automator.py:103`

Additional keys: `tube_capacity`, `empty_colour`, `next_button`, `restart_button`.

## Setup

```bash
pip install opencv-python numpy pyautogui pygetwindow mss
# Install ADB and add to PATH; install scrcpy
# Update SCRCPY_PATH in capture.py:20 to your scrcpy.exe location
```

Connect your Android device via USB with USB debugging enabled, then verify with `adb devices`.

## Commands

```bash
python main.py                          # Solve a single level
python main.py --loop                   # Auto-solve unlimited levels
python main.py --loop --levels 50       # Auto-solve N levels
python main.py --dry-run                # Solve without sending taps
python main.py --set-next-button        # One-time: set Next Level button (needed for --loop)
python main.py --set-restart-button     # One-time: set Restart button (needed for retries)
python main.py --calibrate              # Manual calibration fallback (rarely needed)
python main.py --test-detect            # Test tube auto-detection; saves detection_test.png
python main.py --test-popup             # Test popup/win-screen detection
python main.py --win-wait 5.0           # Wait after win animation (default 3.0)
```

## Tests

```bash
python -m pytest tests/                 # All unit tests
python test_popup_detection.py          # Manual popup detection test (root level)
```

## Hidden Slot Strategy

Unrevealed slots are marked `UNKNOWN` (`solver.py:20`). When unknowns are present, `_solve_one_level_impl` (`main.py:726`) runs a prioritised reveal chain each round (up to `max_rounds=40`), re-screenshots, and retries until the board is fully visible, then solves with complete information.

The reveal chain (`main.py:1034-1168`) tries planners in priority order:
1. `plan_late_game_solve` (`solver.py:894`) -- sample completions when <=5 unknowns remain
2. `find_path_to_unknown` (`solver.py:925`) -- BFS shortest peel to expose a buried slot
3. `find_guaranteed_safe_moves` (`solver.py:1148`) -- exhaustive safe-move enumeration
4. `plan_reveal_info_gain` (`solver.py:393`) -- ranks reveals by uncertainty reduction
5. `plan_reveal_deep` (`solver.py:482`) -- multi-layer peel into a tube
6. `plan_consolidate` (`solver.py:583`) -- frees empties for deep-reveal
7. `plan_reveal_maximize` (`solver.py:291`) -- heuristic fallback
8. `pick_best_move_by_determinization` (`solver.py:966`) -- Monte Carlo sampling
9. `plan_reveal_round` (`solver.py:136`) -- heuristic park (last resort)
10. `find_safe_moves` (`solver.py:1303`) -- legacy fallback

## Multi-Attempt System

Each level gets up to 12 attempts (`main.py:743`). `decide_retry` (`main.py:656`) determines next action after each attempt. `LevelMemory` (`level_memory.py:35`) persists discovered hidden-slot colours across restarts (the game is deterministic on restart). `AttemptSim` (`level_memory.py:196`) tracks ball origins so newly-revealed colours can be attributed to their original positions.

## Solver

`solve()` at `solver.py:1452`: A* (`max_states=1_000_000`) then DFS restricted empties (`max_states=2_000_000`) then DFS relaxed (`max_states=2_000_000`). State is `tuple(tuple(t))` -- immutable and hashable for visited-set dedup.

## Additional Documentation

- [`.claude/docs/architectural_patterns.md`](.claude/docs/architectural_patterns.md) -- cross-module patterns: pipeline architecture, fallback chains, coordinate abstraction, caching, immutable state, human-like randomisation, reveal chain architecture, cross-restart learning, debug instrumentation
