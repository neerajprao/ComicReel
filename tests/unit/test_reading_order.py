from comicreel.reading_order import order_panels

# A hand-labeled 6-panel grid matching tests/fixtures/sample_pages/page_04.jpg's
# layout: two side-by-side panels, then a full-width panel, then two more
# side-by-side rows. bbox = (x, y, w, h).
GRID_LAYOUT = {
    "top_left": (0, 0, 100, 100),
    "top_right": (100, 0, 100, 100),
    "full_width": (0, 100, 200, 80),
    "mid_left": (0, 180, 100, 100),
    "mid_right": (100, 180, 100, 100),
    "bottom_left": (0, 280, 100, 100),
    "bottom_right": (100, 280, 100, 100),
}


def _ordered_names(direction: str = "auto") -> list[str]:
    names = list(GRID_LAYOUT)
    boxes = [GRID_LAYOUT[n] for n in names]
    return [names[i] for i in order_panels(boxes, direction)]


def test_empty_input():
    assert order_panels([]) == []


def test_single_panel():
    assert order_panels([(0, 0, 10, 10)]) == [0]


def test_ltr_grid_reading_order():
    assert _ordered_names("ltr") == [
        "top_left",
        "top_right",
        "full_width",
        "mid_left",
        "mid_right",
        "bottom_left",
        "bottom_right",
    ]


def test_rtl_grid_reading_order():
    assert _ordered_names("rtl") == [
        "top_right",
        "top_left",
        "full_width",
        "mid_right",
        "mid_left",
        "bottom_right",
        "bottom_left",
    ]


def test_auto_defaults_to_ltr():
    assert _ordered_names("auto") == _ordered_names("ltr")


def test_slightly_misaligned_row_still_groups_together():
    # Two panels in the "same" row but with slightly different y (common with
    # imperfect detections) should still be treated as one row, not two.
    boxes = [
        (0, 0, 100, 100),
        (100, 8, 100, 92),
        (0, 100, 200, 50),
    ]
    assert order_panels(boxes, "ltr") == [0, 1, 2]
