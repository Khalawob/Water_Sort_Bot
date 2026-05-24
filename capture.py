"""
Capture — ADB screenshots for reading, scrcpy for display + tapping.

ADB screenshots are pixel-perfect at native resolution — ideal for
exact colour matching. scrcpy runs alongside for live monitoring
and receives taps via pyautogui.
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCREENSHOT_PATH = Path(__file__).parent / "screenshot.png"
WINDOW_TITLE = "WaterSortBot"
SCRCPY_PATH = r"C:\Users\Brian\Downloads\scrcpy-win64-v4.0\scrcpy-win64-v4.0\scrcpy.exe"

# ── scrcpy (display + tap target only) ───────────────────────────────

_scrcpy_process = None


def launch_scrcpy(stay_awake=True):
    """Launch scrcpy for live display and tap input."""
    global _scrcpy_process

    if _scrcpy_process and _scrcpy_process.poll() is None:
        print("  scrcpy already running.")
        return True

    cmd = [
        SCRCPY_PATH,
        "--window-title", WINDOW_TITLE,
        "--no-audio",
    ]
    if stay_awake:
        cmd.append("--stay-awake")

    try:
        _scrcpy_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

        if _scrcpy_process.poll() is not None:
            print("  ✗ scrcpy exited immediately.")
            _scrcpy_process = None
            return False

        print(f"  ✓ scrcpy launched (window: '{WINDOW_TITLE}')")
        return True
    except FileNotFoundError:
        print(f"  ✗ scrcpy not found at: {SCRCPY_PATH}")
        _scrcpy_process = None
        return False


def stop_scrcpy():
    """Stop scrcpy."""
    global _scrcpy_process
    if _scrcpy_process and _scrcpy_process.poll() is None:
        _scrcpy_process.terminate()
        _scrcpy_process.wait(timeout=5)
        print("  scrcpy stopped.")
    _scrcpy_process = None


# ── Window detection (for pyautogui tapping) ─────────────────────────

def find_scrcpy_window():
    """Find the scrcpy window position and size."""
    system = platform.system()
    try:
        if system == "Windows":
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(WINDOW_TITLE)
            if windows:
                w = windows[0]
                if w.isMinimized:
                    w.restore()
                    time.sleep(0.3)
                return (w.left, w.top, w.width, w.height)

        elif system == "Linux":
            result = subprocess.run(
                ["xdotool", "search", "--name", WINDOW_TITLE],
                capture_output=True, text=True,
            )
            if result.stdout.strip():
                wid = result.stdout.strip().split("\n")[0]
                geo = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", wid],
                    capture_output=True, text=True,
                )
                vals = {}
                for line in geo.stdout.strip().split("\n"):
                    if "=" in line:
                        k, v = line.split("=")
                        vals[k] = int(v)
                size = subprocess.run(
                    ["xwininfo", "-id", wid],
                    capture_output=True, text=True,
                )
                for line in size.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("Width:"):
                        vals["WIDTH"] = int(line.split(":")[1].strip())
                    elif line.startswith("Height:"):
                        vals["HEIGHT"] = int(line.split(":")[1].strip())
                if all(k in vals for k in ["X", "Y", "WIDTH", "HEIGHT"]):
                    return (vals["X"], vals["Y"], vals["WIDTH"], vals["HEIGHT"])

        elif system == "Darwin":
            script = f'''
            tell application "System Events"
                set proc to first process whose name contains "scrcpy"
                set win to first window of proc
                set {{x, y}} to position of win
                set {{w, h}} to size of win
                return (x as text) & "," & (y as text) & "," & (w as text) & "," & (h as text)
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True,
            )
            if result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) == 4:
                    return tuple(int(p.strip()) for p in parts)
    except Exception as e:
        print(f"  ⚠ Window detection error: {e}")
    return None


def get_title_bar_height():
    system = platform.system()
    defaults = {"Windows": 31, "Linux": 37, "Darwin": 28}
    return defaults.get(system, 30)


# ── ADB screenshot (primary method) ─────────────────────────────────

def screenshot(save=True):
    """
    Capture via ADB fast pipe. Pixel-perfect at native resolution.
    This is the primary capture method for colour reading.
    """
    try:
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return _screenshot_legacy(save)

        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return _screenshot_legacy(save)

        if save:
            cv2.imwrite(str(SCREENSHOT_PATH), image)
        return image

    except subprocess.TimeoutExpired:
        print("  ⚠ Screenshot timed out.")
        return None
    except FileNotFoundError:
        print("  ✗ 'adb' not found on PATH.")
        sys.exit(1)


def _screenshot_legacy(save=True):
    """Fallback two-step capture for older devices."""
    try:
        subprocess.run(
            ["adb", "shell", "screencap", "-p", "/sdcard/screenshot.png"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["adb", "pull", "/sdcard/screenshot.png", str(SCREENSHOT_PATH)],
            check=True, capture_output=True,
        )
        return cv2.imread(str(SCREENSHOT_PATH))
    except subprocess.CalledProcessError:
        return None


# ── Device resolution ────────────────────────────────────────────────

_device_resolution = None


def get_device_resolution():
    """Get the device screen resolution."""
    global _device_resolution
    if _device_resolution:
        return _device_resolution
    try:
        result = subprocess.run(
            ["adb", "shell", "wm", "size"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().split("\n"):
            if "size" in line.lower():
                size_str = line.split(":")[-1].strip()
                w, h = size_str.split("x")
                _device_resolution = (int(w), int(h))
                return _device_resolution
    except Exception:
        pass
    return None
