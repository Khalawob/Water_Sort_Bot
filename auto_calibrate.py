"""
Auto-calibration — detects tube positions from an ADB screenshot.

Uses a border-scanline approach:
1. Creates a mask of the tube glass border colour (grey ~185-190 RGB)
2. Projects horizontally to find row Y-bands
3. Scans each band horizontally to find border-pair tube positions
4. Calculates 4 evenly-spaced sample points per tube

This detects ALL tubes regardless of contents — filled, empty, or hidden.
"""

import cv2
import numpy as np


# ── Border detection ─────────────────────────────────────────────────

def find_border_mask(image, game_top, game_bottom):
    """
    Create a boolean mask of tube glass border pixels.

    The tube borders are a consistent light grey (~RGB 183-192)
    with very low channel spread. This is distinct from:
      - Game background: dark teal (~24, 39, 44)
      - Colours: high saturation, large channel spread
      - Hidden slot grey: darker (40-130), also low spread but lower range
      - White UI text/icons: above 205
    """
    game = image[game_top:game_bottom, :]
    b = game[:, :, 0].astype(np.int16)
    g = game[:, :, 1].astype(np.int16)
    r = game[:, :, 2].astype(np.int16)

    min_ch = np.minimum(np.minimum(r, g), b)
    max_ch = np.maximum(np.maximum(r, g), b)

    mask = (min_ch >= 170) & (max_ch <= 205) & ((max_ch - min_ch) <= 20)
    return mask


def find_row_bands(border_mask, min_band_height=80, min_gap=30):
    """
    Find Y-bands containing tube rows by horizontal border density.

    Each row of tubes produces a tall vertical band (200-300px) where
    every scanline crosses tube borders. Gaps between rows have zero
    border pixels.
    """
    raw_density = border_mask.sum(axis=1).astype(float)

    # Smooth to prevent single-row dips from fragmenting bands
    kernel_size = 25
    kernel = np.ones(kernel_size) / kernel_size
    row_density = np.convolve(raw_density, kernel, mode='same')

    # Low threshold — even 3 tubes with thin borders produce ~20 pixels/row.
    # Gaps between rows have 0 density, so anything above ~8 is a real band.
    threshold = max(row_density.max() * 0.04, 8)

    bands = []
    in_band = False
    band_start = 0

    for y in range(len(row_density)):
        if row_density[y] > threshold and not in_band:
            band_start = y
            in_band = True
        elif row_density[y] <= threshold and in_band:
            if y - band_start >= min_band_height:
                bands.append((band_start, y))
            in_band = False

    if in_band and len(row_density) - band_start >= min_band_height:
        bands.append((band_start, len(row_density)))

    # Merge bands that are very close (could be split by a gap in borders)
    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    return merged


def find_border_runs(border_mask, scan_y, min_run_width=2):
    """
    Find runs of border pixels on a single horizontal scanline.
    Returns list of (start_x, end_x) for each run.
    """
    line = border_mask[scan_y, :]
    runs = []
    in_run = False
    run_start = 0

    for x in range(len(line)):
        if line[x] and not in_run:
            run_start = x
            in_run = True
        elif not line[x] and in_run:
            if x - run_start >= min_run_width:
                runs.append((run_start, x))
            in_run = False

    if in_run and len(line) - run_start >= min_run_width:
        runs.append((run_start, len(line)))

    return runs


def find_tubes_in_band(border_mask, band_top, band_bottom):
    """
    Find tube positions within a row band by scanning multiple lines
    and pairing border runs sequentially.

    Each tube produces exactly 2 border runs (left edge, right edge)
    on any horizontal scanline through its body, so sequential pairing
    of runs (1,2), (3,4), (5,6)... gives the tube positions.

    Uses majority voting across scanlines: anti-aliased '?' glyph edges
    fall in the border colour range (grey 170-205, low spread) and create
    extra runs that split one real tube into two narrow fakes. With 4-slot
    tubes the '?' characters occupy at most 2 of the 5 sampled Y positions,
    so the 3 clean scanlines outvote the 2 contaminated ones.
    """
    band_height = band_bottom - band_top

    # Scan several lines through the middle portion of the band
    # (avoid the very top/bottom where borders curve)
    all_results = []
    for offset_pct in [0.30, 0.40, 0.50, 0.60, 0.70]:
        scan_y = band_top + int(band_height * offset_pct)
        runs = find_border_runs(border_mask, scan_y)

        if len(runs) < 2:
            continue

        # Merge runs that are very close (border split by a 1-2px gap)
        merged = [runs[0]]
        for s, e in runs[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= 3:  # gap of 3px or less = same border
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        runs = merged

        # Need an even number of runs for pairing
        if len(runs) < 2 or len(runs) % 2 != 0:
            continue

        # Sequential pairing: (left, right), (left, right), ...
        tubes = []
        valid = True
        for i in range(0, len(runs) - 1, 2):
            left_start, left_end = runs[i]
            right_start, right_end = runs[i + 1]
            width = right_end - left_start

            # Sanity check: tube width should be reasonable (30-200px)
            if width < 30 or width > 200:
                valid = False
                break

            tubes.append((left_start, width))

        if not valid:
            continue

        # Verify consistent widths
        widths = [w for _, w in tubes]
        if max(widths) - min(widths) <= 15:
            all_results.append(tubes)

    if not all_results:
        return []

    # Vote on tube count across all valid scanlines. Contaminated scanlines
    # (those crossing '?' glyphs) report 2× the real tube count. The majority
    # count is correct; ties are broken by width consistency.
    count_votes = {}
    for result in all_results:
        n = len(result)
        count_votes[n] = count_votes.get(n, 0) + 1
    winner_count = max(count_votes, key=count_votes.get)
    candidates = [r for r in all_results if len(r) == winner_count]

    return min(candidates, key=lambda r: max(w for _, w in r) - min(w for _, w in r))


# ── Main detection ───────────────────────────────────────────────────

def detect_tubes(image, tube_capacity=4):
    """
    Auto-detect tube positions from an ADB screenshot.
    Returns a list of tube dicts with sample_points.
    """
    h, w = image.shape[:2]

    # Define game area — skip UI at top and ad banner at bottom
    game_top = int(h * 0.20)
    game_bottom = int(h * 0.85)

    # Step 1: Create border mask
    border_mask = find_border_mask(image, game_top, game_bottom)

    # Step 2: Find row bands
    bands = find_row_bands(border_mask)

    if not bands:
        print("  ✗ No tube rows found via border scan.")
        return []

    # Step 3: Find tubes in each band
    all_rects = []
    for band_top, band_bottom in bands:
        tubes_in_band = find_tubes_in_band(border_mask, band_top, band_bottom)

        for tube_x, tube_width in tubes_in_band:
            abs_y = band_top + game_top
            abs_h = band_bottom - band_top
            all_rects.append((tube_x, abs_y, tube_width, abs_h))

    if not all_rects:
        print("  ✗ No tubes found in any row band.")
        return []

    # Step 4: Calculate sample points per tube
    tubes = []
    for (x, y, bw, bh) in all_rects:
        center_x = x + bw // 2

        # 4 evenly spaced points from bottom to top
        margin_top = int(bh * 0.10)
        margin_bottom = int(bh * 0.10)
        usable_top = y + margin_top
        usable_bottom = y + bh - margin_bottom
        slot_height = (usable_bottom - usable_top) / tube_capacity

        sample_points = []
        for i in range(tube_capacity):
            slot_y = int(usable_bottom - slot_height * (i + 0.5))
            sample_points.append((center_x, slot_y))

        tubes.append({"sample_points": sample_points})

    return tubes


def detect_empty_colour(image, tubes):
    """Auto-detect the empty/background colour."""
    h, w = image.shape[:2]

    if not tubes:
        return (30, 30, 30)

    # Sample from tubes that look empty (darkest readings)
    darkest = None
    darkest_brightness = 999

    for tube in tubes:
        for x, y in tube["sample_points"]:
            if 0 <= y < h and 0 <= x < w:
                bgr = image[y, x]
                rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
                brightness = sum(rgb)
                if brightness < darkest_brightness:
                    darkest_brightness = brightness
                    darkest = rgb

    return darkest or (30, 30, 30)


def auto_calibrate(image, tube_capacity=4):
    """
    Full auto-calibration. Returns a config dict or None.
    """
    tubes = detect_tubes(image, tube_capacity=tube_capacity)

    if not tubes:
        print("  ✗ Could not detect any tubes.")
        return None

    empty_colour = detect_empty_colour(image, tubes)

    # Count filled vs empty
    filled = 0
    for tube in tubes:
        x, y = tube["sample_points"][0]
        h, w = image.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            bgr = image[y, x]
            brightness = int(bgr[0]) + int(bgr[1]) + int(bgr[2])
            if brightness > 150:
                filled += 1

    print(f"  ✓ Detected {len(tubes)} tubes ({filled} filled, {len(tubes) - filled} empty)")
    print(f"  Empty colour: RGB{empty_colour}")

    config = {
        "tubes": tubes,
        "tube_capacity": tube_capacity,
        "empty_colour": list(empty_colour),
    }
    return config


def visualise_detection(image, config):
    """Draw detected tubes on the image for debugging."""
    vis = image.copy()

    for i, tube in enumerate(config["tubes"]):
        points = tube["sample_points"]

        # Draw bounding area
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs) - 20, max(xs) + 20
        y_min, y_max = min(ys) - 15, max(ys) + 15
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 200, 200), 1)

        # Draw sample points
        for j, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
            cv2.putText(vis, str(j + 1), (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw tube label
        top_x, top_y = points[-1]
        cv2.putText(vis, f"T{i+1}", (top_x - 10, top_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return vis


def detect_buttons(image):
    """
    Detect the 4 UI buttons (menu, restart, undo, add_tube) from the top 15%
    of the screenshot by finding purple/lilac coloured regions.

    Returns {"menu": (x,y), "restart": (x,y), "undo": (x,y), "add_tube": (x,y)}
    or {} if fewer than 4 buttons are found.
    """
    h, w = image.shape[:2]
    roi = image[:int(h * 0.15), :]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    # Purple/lilac: blue-dominant, enough blue-minus-green spread to exclude grey
    mask = (
        (b >= 140) & (b <= 220) &
        (g >= 60)  & (g <= 145) &
        (r >= 80)  & (r <= 165) &
        (b > r) &
        ((b - g) >= 50)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 60 <= bw <= 200 and bh >= 40:
            candidates.append((x + bw // 2, y + bh // 2))

    candidates.sort(key=lambda p: p[0])

    if len(candidates) < 4:
        return {}

    names = ["menu", "restart", "undo", "add_tube"]
    return {name: coord for name, coord in zip(names, candidates[:4])}


def detect_win_screen(image):
    """Detect the win screen by its red banner and optionally locate a yellow NEXT button.

    Returns {"detected": bool, "next_button_position": (x, y) | None}.
    """
    h, w = image.shape[:2]
    roi = image[int(h * 0.15):int(h * 0.32), :]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    mask = (
        (r >= 170) & (r <= 255) &
        (g >= 10) & (g <= 130) &
        (b >= 30) & (b <= 130) &
        (r > g + 60) &
        (r > b + 60)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    banner_found = False
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw >= w * 0.4 and bh >= 30:
            banner_found = True
            break

    if not banner_found:
        return {"detected": False, "next_button_position": None}

    y_top = int(h * 0.70)
    y_bot = int(h * 0.90)
    btn_roi = image[y_top:y_bot, :]

    bb = btn_roi[:, :, 0].astype(np.int16)
    bg = btn_roi[:, :, 1].astype(np.int16)
    br = btn_roi[:, :, 2].astype(np.int16)

    yellow_mask = (
        (bb >= 0) & (bb <= 50) &
        (bg >= 150) & (bg <= 240) &
        (br >= 200) & (br <= 255) &
        (br > bg)
    ).astype(np.uint8) * 255

    btn_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in btn_contours:
        bx, by, bbw, bbh = cv2.boundingRect(cnt)
        if bbw >= 200 and bbh >= 60:
            area = bbw * bbh
            if area > best_area:
                best = (bx, by, bbw, bbh)
                best_area = area

    if best is not None:
        bx, by, bbw, bbh = best
        cx = bx + bbw // 2
        cy = y_top + by + bbh // 2
        return {"detected": True, "next_button_position": (cx, cy)}

    return {"detected": True, "next_button_position": None}


def detect_popup(image):
    """
    Detect theme-unlock or special-level popups by their coloured buttons.

    Returns {"popup": "theme"|"special"|None, "skip_position": (x,y)|None}.
    Coordinates are absolute device pixels (ADB screenshot space).
    """
    win_info = detect_win_screen(image)
    if win_info["detected"]:
        return {"popup": None, "skip_position": None}

    h, w = image.shape[:2]
    roi_top = int(h * 0.50)
    roi_bottom = int(h * 0.90)
    roi = image[roi_top:roi_bottom, :]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    green_mask = (
        (b >= 40) & (b <= 120) &
        (g >= 100) & (g <= 230) &
        (r >= 0) & (r <= 40) &
        (g > r + 60) &
        (g > b + 20)
    ).astype(np.uint8) * 255

    yellow_mask = (
        (b >= 0) & (b <= 50) &
        (g >= 150) & (g <= 240) &
        (r >= 200) & (r <= 255) &
        (r > g)
    ).astype(np.uint8) * 255

    blue_mask = (
        (b >= 170) & (b <= 250) &
        (g >= 130) & (g <= 220) &
        (r >= 60) & (r <= 150) &
        (b > r + 50)
    ).astype(np.uint8) * 255

    def _find_button(mask, min_w=200, min_h=60):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw >= min_w and bh >= min_h:
                area = bw * bh
                if area > best_area:
                    best = (x, y, bw, bh)
                    best_area = area
        return best

    green = _find_button(green_mask)
    if green is not None:
        gx, gy, gw, gh = green
        skip_x = gx + gw // 2
        skip_y = roi_top + gy + gh + 60
        return {"popup": "theme", "skip_position": (skip_x, skip_y)}

    yellow = _find_button(yellow_mask)
    if yellow is not None:
        yx, yy, yw, yh = yellow
        blue = _find_button(blue_mask)
        if blue is not None:
            bx, by, bbw, bbh = blue
            skip_x = bx + bbw // 2
            skip_y = roi_top + by + bbh // 2
            return {"popup": "special", "skip_position": (skip_x, skip_y)}
        skip_x = yx + yw // 2
        skip_y = roi_top + yy + yh + 150
        return {"popup": "special", "skip_position": (skip_x, skip_y)}

    return {"popup": None, "skip_position": None}


if __name__ == "__main__":
    from capture import screenshot, SCREENSHOT_PATH

    print("Taking ADB screenshot...")
    img = screenshot()
    if img is None:
        print("Failed.")
    else:
        print(f"Image: {img.shape[1]}x{img.shape[0]}")
        config = auto_calibrate(img)

        if config:
            print(f"\n{len(config['tubes'])} tubes detected:")
            for i, tube in enumerate(config["tubes"]):
                print(f"  T{i+1}: {tube['sample_points']}")

            vis = visualise_detection(img, config)

            buttons = detect_buttons(img)
            if buttons:
                print("\nButtons detected:")
                for name, (x, y) in buttons.items():
                    print(f"  {name}: ({x}, {y})")
                    cv2.circle(vis, (x, y), 15, (0, 0, 255), 3)
                    cv2.putText(vis, name, (x - 30, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                print("\nNo buttons detected (check colour ranges).")

            vis_path = str(SCREENSHOT_PATH).replace(".png", "_detected.png")
            cv2.imwrite(vis_path, vis)
            print(f"\nVisualisation saved to: {vis_path}")
