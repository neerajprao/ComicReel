import numpy as np

from comicreel.stages.panel_detection import detect_panel_boxes_cv


def _draw_synthetic_page(panel_rects: list[tuple[int, int, int, int]], size=(400, 400)) -> np.ndarray:
    """White page with mid-gray filled panel_rects = (x, y, w, h), separated by
    white gutters -- exercises the whitespace-gutter-projection algorithm
    without needing drawn border lines."""
    h, w = size
    page = np.full((h, w, 3), 255, dtype=np.uint8)
    for x, y, bw, bh in panel_rects:
        page[y : y + bh, x : x + bw, :] = 120
    return page


def test_detects_a_simple_2x2_grid():
    rects = [
        (10, 10, 180, 180),
        (210, 10, 180, 180),
        (10, 210, 180, 180),
        (210, 210, 180, 180),
    ]
    page = _draw_synthetic_page(rects, size=(400, 400))
    boxes = detect_panel_boxes_cv(page)

    assert len(boxes) == 4
    for bx, by, bw, bh in boxes:
        assert bw > 150 and bh > 150


def test_detects_a_full_width_panel_plus_two_column_row():
    rects = [
        (10, 10, 380, 100),   # full-width top panel
        (10, 130, 180, 180),  # bottom-left
        (210, 130, 180, 180),  # bottom-right
    ]
    page = _draw_synthetic_page(rects, size=(400, 320))
    boxes = detect_panel_boxes_cv(page)
    assert len(boxes) == 3


def test_ignores_small_gutter_gap_below_min_panel_fraction():
    # A 2px sliver of "panel" between two gutters shouldn't survive the
    # min_panel_fraction size filter.
    page = np.full((200, 200, 3), 255, dtype=np.uint8)
    page[10:190, 10:190, :] = 120
    page[10:190, 95:97, :] = 255  # thin 2px gutter splitting it in two
    boxes = detect_panel_boxes_cv(page, min_panel_fraction=0.06)
    assert len(boxes) == 1


def test_empty_white_page_has_no_panels():
    page = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = detect_panel_boxes_cv(page)
    assert boxes == []
