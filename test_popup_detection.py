#!/usr/bin/env python3
"""Test popup and win-screen detection against reference screenshots."""

import cv2
import sys
from auto_calibrate import detect_win_screen, detect_popup

TESTS = [
    {
        "file": "Screenshot_20260616-090527_Water Sort Puzzle.jpg",
        "label": "Win screen (CLAIM x2)",
        "expect_win": True,
        "expect_popup": None,
        "expect_next_button": True,
    },
    {
        "file": "Screenshot_20260616-103439_Water Sort Puzzle.jpg",
        "label": "Win screen (yellow NEXT)",
        "expect_win": True,
        "expect_popup": None,
        "expect_next_button": True,
    },
    {
        "file": "Screenshot_20260606-090156_Water Sort Puzzle.jpg",
        "label": "Theme popup (frame 1)",
        "expect_win": False,
        "expect_popup": "theme",
        "expect_next_button": False,
    },
    {
        "file": "Screenshot_20260606-090158_Water Sort Puzzle.jpg",
        "label": "Theme popup (frame 2)",
        "expect_win": False,
        "expect_popup": "theme",
        "expect_next_button": False,
    },
    {
        "file": "Screenshot_20260615-181826_Water Sort Puzzle.jpg",
        "label": "Special level popup",
        "expect_win": False,
        "expect_popup": "special",
        "expect_next_button": False,
    },
]

def main():
    passed = 0
    failed = 0

    for t in TESTS:
        img = cv2.imread(t["file"])
        if img is None:
            print(f"SKIP  {t['label']} — file not found: {t['file']}")
            continue

        h, w = img.shape[:2]
        win_info = detect_win_screen(img)
        popup = detect_popup(img)

        win_ok = win_info["detected"] == t["expect_win"]
        popup_ok = popup["popup"] == t["expect_popup"]
        btn_ok = (win_info["next_button_position"] is not None) == t["expect_next_button"]
        status = "PASS" if (win_ok and popup_ok and btn_ok) else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"{status}  {t['label']} ({w}x{h})")
        print(f"       win_screen: {win_info['detected']} (expected {t['expect_win']}){'' if win_ok else '  <-- MISMATCH'}")
        print(f"       popup:      {popup['popup']} (expected {t['expect_popup']}){'' if popup_ok else '  <-- MISMATCH'}")
        has_btn = win_info["next_button_position"] is not None
        print(f"       next_btn:   {has_btn} (expected {t['expect_next_button']}){'' if btn_ok else '  <-- MISMATCH'}")
        if win_info["next_button_position"]:
            print(f"       next_pos:   {win_info['next_button_position']}")
        if popup["skip_position"]:
            print(f"       skip_pos:   {popup['skip_position']}")
        print()

    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
