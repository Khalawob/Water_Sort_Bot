"""Tests for detect_popup — synthetic images + reference screenshots."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_calibrate import detect_popup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_blank(h=1520, w=720):
    return np.full((h, w, 3), (44, 39, 24), dtype=np.uint8)


def _draw_rect(image, x, y, w, h, bgr):
    image[y:y+h, x:x+w] = bgr


def test_theme_popup():
    img = _make_blank()
    _draw_rect(img, 180, 973, 360, 129, (75, 179, 0))
    result = detect_popup(img)
    assert result["popup"] == "theme", f"Expected 'theme', got {result['popup']}"
    skip_x, skip_y = result["skip_position"]
    assert abs(skip_x - 360) < 10, f"skip_x={skip_x}, expected ~360"
    assert abs(skip_y - 1162) < 15, f"skip_y={skip_y}, expected ~1162"
    print("  PASS: theme popup detected correctly")


def test_theme_popup_edge_colour():
    img = _make_blank()
    _draw_rect(img, 180, 973, 360, 129, (56, 126, 0))
    result = detect_popup(img)
    assert result["popup"] == "theme", f"Expected 'theme', got {result['popup']}"
    print("  PASS: green edge colour detected")


def test_special_popup_with_blue():
    img = _make_blank()
    _draw_rect(img, 194, 894, 332, 122, (0, 207, 255))
    _draw_rect(img, 193, 1106, 334, 125, (226, 180, 103))
    result = detect_popup(img)
    assert result["popup"] == "special", f"Expected 'special', got {result['popup']}"
    skip_x, skip_y = result["skip_position"]
    assert abs(skip_x - 360) < 10, f"skip_x={skip_x}, expected ~360"
    assert abs(skip_y - 1168) < 15, f"skip_y={skip_y}, expected ~1168"
    print("  PASS: special popup (with blue SKIP) detected correctly")


def test_special_popup_no_blue():
    img = _make_blank()
    _draw_rect(img, 194, 894, 332, 122, (0, 207, 255))
    result = detect_popup(img)
    assert result["popup"] == "special", f"Expected 'special', got {result['popup']}"
    assert result["skip_position"] is not None
    print("  PASS: special popup (no blue) fallback works")


def test_no_popup_on_blank():
    img = _make_blank()
    result = detect_popup(img)
    assert result["popup"] is None, f"Expected None, got {result['popup']}"
    assert result["skip_position"] is None
    print("  PASS: no false positive on blank screen")


def test_reference_screenshots():
    import cv2
    refs = [
        ("Screenshot_20260606-090156_Water Sort Puzzle.jpg", "theme"),
        ("Screenshot_20260606-090158_Water Sort Puzzle.jpg", "theme"),
        ("Screenshot_20260615-181826_Water Sort Puzzle.jpg", "special"),
    ]
    for fname, expected in refs:
        path = os.path.join(PROJECT_ROOT, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname} not found")
            continue
        img = cv2.imread(path)
        result = detect_popup(img)
        assert result["popup"] == expected, f"{fname}: expected '{expected}', got {result['popup']}"
        assert result["skip_position"] is not None
        print(f"  PASS: {fname} -> {result['popup']} skip={result['skip_position']}")


def test_no_popup_on_game_screenshot():
    import cv2
    import glob
    game_imgs = glob.glob(os.path.join(PROJECT_ROOT, "debug_screenshots", "level_001", "*.png"))
    if not game_imgs:
        print("  SKIP: no game screenshots found")
        return
    img = cv2.imread(game_imgs[0])
    result = detect_popup(img)
    assert result["popup"] is None, f"False positive on game screen: {result['popup']}"
    print(f"  PASS: no false positive on {os.path.basename(game_imgs[0])}")


if __name__ == "__main__":
    test_theme_popup()
    test_theme_popup_edge_colour()
    test_special_popup_with_blue()
    test_special_popup_no_blue()
    test_no_popup_on_blank()
    test_reference_screenshots()
    test_no_popup_on_game_screenshot()
    print("\nAll popup detection tests passed.")
