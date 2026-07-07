"""
Screen Reader — extracts tube colours from ADB screenshots using
exact pixel matching.

ADB screenshots are pixel-perfect, so the same colour always has the
exact same RGBA/BGR value. No clustering or fuzzy matching needed.

Also detects hidden/unknown slots that haven't been revealed yet.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from capture import screenshot as take_screenshot

CONFIG_PATH = Path(__file__).parent / "config.json"

# How close a pixel must be to the empty colour to count as empty
EMPTY_THRESHOLD = 15

# How close a pixel must be to count as "hidden" (grey/dark cover)
# Hidden slots are darker/greyer than game colours but different from empty
HIDDEN_COLOUR_RANGE = {
    "min": (40, 40, 40),    # darker than this = empty
    "max": (100, 100, 100), # lighter than this = actual colour
}

UNKNOWN = "unknown"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return None


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {CONFIG_PATH}")


# ── Pixel reading ────────────────────────────────────────────────────

def read_pixel(image, x, y):
    """Read BGR value at (x, y) and return as a tuple."""
    b, g, r = image[y, x]
    return (int(r), int(g), int(b))


def colour_distance(c1, c2):
    """Euclidean distance between two RGB tuples."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def is_empty(rgb, empty_rgb):
    """Check if a pixel is the empty/background colour."""
    return colour_distance(rgb, empty_rgb) < EMPTY_THRESHOLD


def is_hidden(rgb, empty_rgb):
    """
    Check if a pixel represents a hidden/unrevealed slot.
    Hidden slots are typically grey — darker than game colours
    but distinctly different from the empty background.
    """
    # Must not be empty
    if is_empty(rgb, empty_rgb):
        return False

    r, g, b = rgb
    min_r, min_g, min_b = HIDDEN_COLOUR_RANGE["min"]
    max_r, max_g, max_b = HIDDEN_COLOUR_RANGE["max"]

    # Check if it falls in the grey range
    in_range = (min_r <= r <= max_r and min_g <= g <= max_g and min_b <= b <= max_b)

    # Also check if it's greyish (low saturation — R, G, B are close together)
    spread = max(r, g, b) - min(r, g, b)
    is_grey = spread < 40

    return in_range and is_grey


def _slot_is_hidden(image, x, y, radius):
    """Detect a hidden "?" slot by patch bimodality.

    A hidden slot shows a white "?" glyph on a dark background, so a patch around
    its centre contains BOTH bright (glyph) and dark (background) pixels. Real
    balls/liquid render as a flat, uniform patch — even genuine grey or white
    balls — so they fail this test. This is robust to where the centre sample
    lands (glyph stroke, antialiased edge, or background), unlike a single pixel.
    """
    h, w = image.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius)
    x0, x1 = max(0, x - radius), min(w, x + radius)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    brightness = patch.mean(axis=2)  # channel-order independent
    white_frac = float((brightness > 200).mean())
    dark_frac = float((brightness < 100).mean())
    return white_frac > 0.03 and dark_frac > 0.30


# ── Exact pixel colour matching ──────────────────────────────────────

def read_tubes(image, config, return_colours=False):
    """
    Read tube colours using exact pixel matching.

    Each unique BGR value gets assigned a colour name. Same pixel value
    = same colour, guaranteed, because ADB screenshots are pixel-perfect.

    Hidden/unknown slots are marked as "unknown".

    When ``return_colours`` is True, also returns the ``seen_colours`` map
    ({rgb_tuple: label}) so callers can recover the RGB behind each label
    (labels themselves are not stable across reads).
    """
    empty_rgb = tuple(config.get("empty_colour", [40, 40, 40]))
    seen_colours = {}
    colour_counter = 0

    tubes = []

    for tube_info in config["tubes"]:
        sample_points = tube_info["sample_points"]

        # Patch half-size for hidden-slot detection: large enough to span the
        # "?" glyph and its dark background, small enough to stay inside one slot
        # (slot half-height ≈ slot_spacing / 2).
        if len(sample_points) >= 2:
            slot_spacing = abs(sample_points[1][1] - sample_points[0][1])
        else:
            slot_spacing = 0
        hidden_radius = max(6, slot_spacing // 4)

        layers = []
        for (x, y) in sample_points:
            rgb = read_pixel(image, x, y)

            # Check empty
            if is_empty(rgb, empty_rgb):
                continue

            # Hidden "?" slots are a white glyph on a dark background — a bimodal
            # patch. Real balls/liquid (incl. flat grey or white) are uniform, so
            # this avoids recording the glyph's white centre or grey antialiased
            # edge as a spurious colour.
            if _slot_is_hidden(image, x, y, hidden_radius):
                layers.append(UNKNOWN)
                continue

            # Exact match — use a tolerance of 3 to handle any
            # sub-pixel rounding (shouldn't happen with ADB but safe)
            matched = False
            for known_rgb, name in seen_colours.items():
                if colour_distance(rgb, known_rgb) < 5:
                    layers.append(name)
                    matched = True
                    break

            if not matched:
                colour_counter += 1
                name = f"colour_{colour_counter}"
                seen_colours[rgb] = name
                layers.append(name)

        tubes.append(layers)

    # Print detected colours
    print(f"\n  Colours detected ({len(seen_colours)}):")
    for rgb, name in seen_colours.items():
        print(f"    {name:15s}  RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})")

    has_unknown = any(UNKNOWN in tube for tube in tubes)
    if has_unknown:
        unknown_count = sum(tube.count(UNKNOWN) for tube in tubes)
        print(f"  Hidden slots: {unknown_count}")

    if return_colours:
        return tubes, seen_colours
    return tubes


def has_unknowns(tubes):
    """Check if any tubes have hidden/unknown slots."""
    return any(UNKNOWN in tube for tube in tubes)


# ── Game screen detection ────────────────────────────────────────────

def is_game_screen(image, config):
    """Check if the game is visible (not an ad/popup)."""
    empty_rgb = tuple(config.get("empty_colour", [40, 40, 40]))
    img_h = image.shape[0]

    samples = []
    for tube_info in config["tubes"][:5]:
        sample_points = tube_info["sample_points"]
        x, y = sample_points[0]
        rgb = read_pixel(image, x, y)

        # An empty or hidden ("?") bottom slot is a sure sign the board is
        # visible. The "?" glyph's centre pixel is white, so probe ±¼ slot to
        # catch the dark grey cover around it (mirrors read_tubes' probe).
        if is_empty(rgb, empty_rgb) or is_hidden(rgb, empty_rgb):
            return True
        if len(sample_points) >= 2:
            probe_dy = max(4, abs(sample_points[1][1] - sample_points[0][1]) // 4)
            for dy in (probe_dy, -probe_dy):
                ny = y + dy
                if 0 <= ny < img_h and is_hidden(read_pixel(image, x, ny), empty_rgb):
                    return True

        samples.append(rgb)

    if len(samples) < 2:
        return True

    # Reaching here means every sampled bottom slot is a real colour. Only a
    # solid overlay would make them all identical and far from empty.
    first = samples[0]
    all_same = all(colour_distance(s, first) < 30 for s in samples[1:])
    if all_same and colour_distance(first, empty_rgb) > 80:
        return False
    return True


def wait_for_game_screen(config, timeout=15, poll_interval=1.5):
    """Keep taking screenshots until the game is visible."""
    import time
    elapsed = 0
    while elapsed < timeout:
        image = take_screenshot()
        if image is not None and is_game_screen(image, config):
            return image
        print(f"    Screen blocked — retrying in {poll_interval}s...")
        time.sleep(poll_interval)
        elapsed += poll_interval
    print("    ⚠ Timed out waiting for game screen.")
    return None


# ── "No more moves" overlay detection ───────────────────────────────

_NMM_BRIGHT = 200           # per-channel minimum to count as white/bright
_NMM_BOTTOM_DENSITY = 0.02  # 2% bright pixels in bottom 10% → "No more moves!" text
_NMM_TOP_DENSITY = 0.08     # 8% bright pixels in top 20% → hand icons


def detect_no_more_moves(image):
    """
    Return True if the "No more moves!" overlay is visible.

    Two independent signals — either is sufficient:
    - Bottom 10%: the "No more moves!" text (bright on dark background).
    - Top 20%: large white hand icons that replace the normal tube buttons.
    """
    h = image.shape[0]

    bottom = image[int(h * 0.90):, :]
    if np.all(bottom > _NMM_BRIGHT, axis=2).mean() > _NMM_BOTTOM_DENSITY:
        return True

    top = image[:int(h * 0.20), :]
    return np.all(top > _NMM_BRIGHT, axis=2).mean() > _NMM_TOP_DENSITY


# ── Calibration ──────────────────────────────────────────────────────

def calibrate():
    """
    Interactive calibration using an ADB screenshot.

    Click tube slots bottom→top, press:
      'n' = next tube
      'e' = set empty colour
      'b' = set Next Level button
      'q' = done
    """
    print("Taking screenshot via ADB...")
    img = take_screenshot()
    if img is None:
        print("  ✗ Can't take screenshot. Check ADB connection.")
        sys.exit(1)

    tubes = []
    current_points = []
    tube_num = 1
    empty_colour = None
    next_button = None

    def on_click(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = img[y, x]
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            current_points.append((x, y))
            cv2.circle(img, (x, y), 8, (0, 255, 0), 2)
            cv2.imshow("Calibrate", img)
            slot = len(current_points)
            print(f"  Tube {tube_num}, slot {slot}: ({x}, {y}) — RGB {rgb}")

    print("\n═══ CALIBRATION MODE ═══")
    print("Click the CENTER of each colour slot, from BOTTOM to TOP.")
    print("  'n' = next tube")
    print("  'e' = click an EMPTY slot (to set the empty colour)")
    print("  'b' = click the 'Next Level' button (needed for --loop mode)")
    print("  'q' = done\n")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate", 540, 960)
    cv2.imshow("Calibrate", img)
    cv2.setMouseCallback("Calibrate", on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            if current_points:
                tubes.append({"sample_points": current_points})
                print(f"  ✓ Tube {tube_num} saved with {len(current_points)} slots.\n")
                current_points = []
                tube_num += 1
            else:
                print("  (no points clicked yet)")
        elif key == ord("e"):
            print("  Click an EMPTY area of a tube...")
            temp = []

            def on_empty_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    bgr = img[y, x]
                    rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
                    temp.append(rgb)
                    print(f"  Empty colour set to RGB {rgb}")

            cv2.setMouseCallback("Calibrate", on_empty_click)
            cv2.waitKey(0)
            if temp:
                empty_colour = list(temp[0])
            cv2.setMouseCallback("Calibrate", on_click)
        elif key == ord("b"):
            print("  Click the 'Next Level' button...")
            btn_pos = []

            def on_btn_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    btn_pos.append((x, y))
                    cv2.circle(img, (x, y), 12, (255, 0, 255), 3)
                    cv2.imshow("Calibrate", img)
                    print(f"  ✓ 'Next Level' button set to ({x}, {y})")

            cv2.setMouseCallback("Calibrate", on_btn_click)
            cv2.waitKey(0)
            if btn_pos:
                next_button = {"x": btn_pos[0][0], "y": btn_pos[0][1]}
            cv2.setMouseCallback("Calibrate", on_click)
        elif key == ord("q"):
            if current_points:
                tubes.append({"sample_points": current_points})
                print(f"  ✓ Tube {tube_num} saved with {len(current_points)} slots.")
            break

    cv2.destroyAllWindows()

    config = {
        "tubes": tubes,
        "tube_capacity": len(tubes[0]["sample_points"]) if tubes else 4,
        "empty_colour": empty_colour or [40, 40, 40],
    }
    if next_button:
        config["next_button"] = next_button
    save_config(config)

    print(f"\nCalibration complete! {len(tubes)} tubes configured.")
    if next_button:
        print(f"'Next Level' button: ({next_button['x']}, {next_button['y']})")
    else:
        print("Tip: re-run --calibrate and press 'b' to set the 'Next Level' button.")
    return config


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        config = load_config()
        if config is None:
            print("No config.json. Run with --calibrate first.")
            sys.exit(1)
        img = take_screenshot()
        if img is not None:
            tubes = read_tubes(img, config)
            print("\nDetected tubes:")
            for i, tube in enumerate(tubes):
                print(f"  Tube {i+1}: {tube}")
