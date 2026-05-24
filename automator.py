"""
Automator — executes moves by clicking on the scrcpy window via pyautogui.

scrcpy forwards mouse clicks on its window to the device as touch events.
This is faster than `adb shell input tap` and doesn't spawn a process per tap.

Taps are jittered in position and timing to look more human-like.
"""

import random
import time

import pyautogui

from capture import find_scrcpy_window, get_title_bar_height, get_device_resolution

# Disable pyautogui's built-in pause between actions
pyautogui.PAUSE = 0
# Disable fail-safe (mouse to corner won't abort)
pyautogui.FAILSAFE = False


# ── Coordinate mapping ───────────────────────────────────────────────

_cached_mapping = None


def _get_window_mapping():
    """
    Compute how to map device coordinates to screen coordinates.
    Returns (window_left, content_top, scale_x, scale_y) or None.
    """
    bounds = find_scrcpy_window()
    if bounds is None:
        return None

    left, top, width, height = bounds
    title_bar = get_title_bar_height()

    content_top = top + title_bar
    content_width = width
    content_height = height - title_bar

    device_res = get_device_resolution()
    if device_res:
        dev_w, dev_h = device_res
        scale_x = content_width / dev_w
        scale_y = content_height / dev_h
    else:
        scale_x = 1.0
        scale_y = 1.0

    return left, content_top, scale_x, scale_y


def refresh_mapping():
    """Re-detect the scrcpy window position."""
    global _cached_mapping
    _cached_mapping = _get_window_mapping()


def _device_to_screen(dev_x, dev_y):
    """Convert device coordinates to screen coordinates for clicking."""
    global _cached_mapping
    if _cached_mapping is None:
        _cached_mapping = _get_window_mapping()
    if _cached_mapping is None:
        raise RuntimeError("Can't find scrcpy window")

    win_left, content_top, scale_x, scale_y = _cached_mapping
    screen_x = int(win_left + dev_x * scale_x)
    screen_y = int(content_top + dev_y * scale_y)
    return screen_x, screen_y


# ── Tap functions ────────────────────────────────────────────────────

def _focus_scrcpy():
    """Bring the scrcpy window to focus by clicking its title bar."""
    try:
        from capture import find_scrcpy_window
        bounds = find_scrcpy_window()
        if bounds:
            left, top, width, height = bounds
            # Click center of the title bar — won't trigger a game tap
            title_x = left + width // 2
            title_y = top + 15  # middle of the title bar
            pyautogui.click(title_x, title_y)
            time.sleep(0.05)
    except Exception:
        pass


def adb_tap(dev_x, dev_y):
    """Tap at device coordinates by clicking on the scrcpy window."""
    _focus_scrcpy()
    screen_x, screen_y = _device_to_screen(dev_x, dev_y)
    pyautogui.click(screen_x, screen_y)


# ── Tube zones and jitter ────────────────────────────────────────────

def get_tube_tap_zones(config):
    """
    Compute a tap zone for each tube — a center point plus jitter range.
    All coordinates in device space.
    """
    zones = []
    for tube_info in config["tubes"]:
        points = tube_info["sample_points"]
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]

        center_x = int(sum(all_x) / len(all_x))
        center_y = int(sum(all_y) / len(all_y))

        spread_x = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 20
        spread_y = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 40

        zones.append({
            "center_x": center_x,
            "center_y": center_y,
            "jitter_x": max(8, spread_x // 3),
            "jitter_y": max(12, spread_y // 3),
        })
    return zones


def jittered_tap(zone):
    """Pick a slightly randomised tap point within the tube's zone."""
    x = zone["center_x"] + random.randint(-zone["jitter_x"], zone["jitter_x"])
    y = zone["center_y"] + random.randint(-zone["jitter_y"], zone["jitter_y"])
    return x, y


def human_delay(base, variance=0.15):
    """Sleep for a slightly randomised duration."""
    jitter = random.uniform(-variance, variance)
    time.sleep(max(0.05, base + jitter))


# ── Move execution ───────────────────────────────────────────────────

def execute_moves(moves, config, delay=1.0):
    """Execute a list of (src_idx, dst_idx, num_poured) moves on the device."""
    zones = get_tube_tap_zones(config)
    refresh_mapping()

    print(f"\nExecuting {len(moves)} moves...\n")

    for i, (src, dst, num_poured) in enumerate(moves, 1):
        src_x, src_y = jittered_tap(zones[src])
        dst_x, dst_y = jittered_tap(zones[dst])

        pour_wait = 0.83 + (0.52 * num_poured)

        print(f"  Move {i}/{len(moves)}: Tube {src+1} → Tube {dst+1}  "
              f"({num_poured} poured, wait {pour_wait:.1f}s)")

        adb_tap(src_x, src_y)
        time.sleep(0.3)         # wait for highlight
        adb_tap(dst_x, dst_y)
        time.sleep(pour_wait)   # smart timing based on pour size

    print("\n✓ All moves executed!")
