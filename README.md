# Water Sort Bot 🧪🤖

Fully automated solver for Water Sort / Ball Sort puzzle games on Android.
Uses scrcpy to mirror your phone screen, reads the colours, solves the puzzle, and clicks the solution on the scrcpy window — all while you watch it live.

## How It Works

```
scrcpy window capture (~50ms) → CIELAB K-means → BFS/DFS Solver → pyautogui clicks → Next Level → Repeat
```

## Setup

### 1. Install Python packages

```bash
pip install opencv-python numpy pyautogui mss pygetwindow
```

All pre-built wheels — no compilation, installs in seconds.

### 2. Install ADB

- **Windows**: Download [SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools), extract, add to PATH
- **macOS**: `brew install android-platform-tools`
- **Linux**: `sudo apt install adb`

### 3. Install scrcpy

- **Windows**: `scoop install scrcpy` or [download](https://github.com/Genymobile/scrcpy/releases)
- **macOS**: `brew install scrcpy`
- **Linux**: `sudo apt install scrcpy`

### 4. Connect your phone

1. **Settings → Developer Options → USB Debugging → ON**
   (Tap "Build Number" 7 times in About Phone)
2. Connect via USB
3. Verify: `adb devices`

### 5. Calibrate (one-time)

Open your Water Sort game to any level:

```bash
python main.py --calibrate
```

For each tube:
- Click the **center** of each colour slot, **bottom to top**
- Press **`n`** for the next tube
- Press **`e`** and click an empty slot for the background colour
- Press **`b`** and click the **"Next Level" button**
- Press **`q`** when done

### 6. Play

```bash
python main.py                    # solve one level
python main.py --loop             # auto-grind levels
python main.py --loop --levels 50 # solve 50 then stop
python main.py --dry-run          # solve without tapping
python main.py --loop --delay 1.5 # slower for laggy games
python main.py --loop --win-wait 5 # longer wait between levels
```

## Important: Don't touch the scrcpy window

Since the bot clicks on the scrcpy window, keep it visible and don't cover it or click on it while the bot is running. The bot refreshes the window position at the start of each level, so if you accidentally bump it, it self-corrects.

## Features

- **~50ms screenshots** from the scrcpy window via mss
- **Near-instant taps** via pyautogui clicking on scrcpy
- **Live monitoring** — watch the game solve itself on your PC
- **Dual solver** — BFS (≤12 tubes) / DFS with heuristics (13-20+ tubes)
- **CIELAB clustering** — auto-detects colours, no manual palette
- **Region averaging** — 15×15 patch per slot for robust sampling
- **State verification** — re-checks every N moves, re-solves if needed
- **Ad/popup detection** — waits for overlays to clear
- **Human-like input** — jittered positions and varied timing
- **Coordinate scaling** — works if scrcpy window is resized

## Troubleshooting

### scrcpy won't launch
- `scrcpy --version` to check it's installed
- `adb devices` to check your phone is connected

### Screenshots fail / clicks miss
- Don't minimise or cover the scrcpy window
- **Windows**: make sure `pygetwindow` is installed
- Re-calibrate if the scrcpy window was resized

### Colours are misread
- Check `screenshot.png` to see what the bot captured
- Increase `sample_radius` in `config.json` (default: 7)

### Loop mode stops
- Increase `--win-wait` for games with ads between levels
- Re-calibrate the Next Level button if it moved

## File Structure

```
water_sort_bot/
├── main.py           # Entry point
├── solver.py         # BFS + DFS solver
├── screen_reader.py  # Colour extraction + calibration
├── capture.py        # scrcpy window capture + management
├── automator.py      # pyautogui tap injection
├── config.json       # Tube positions (from --calibrate)
└── screenshot.png    # Latest frame
```
