import numpy as np

from comicreel.stages.panel_detection import snap_boxes_to_gutters


def test_expands_an_undershot_box_to_the_true_gutter():
    # A 300x300 white page with one 200x200 gray panel at (50,50). A box that
    # undershoots the panel by 20px on every side should snap out to it.
    page = np.full((300, 300, 3), 255, dtype=np.uint8)
    page[50:250, 50:250, :] = 120
    undershot = [(70.0, 70.0, 160.0, 160.0)]

    [snapped] = snap_boxes_to_gutters(page, undershot, min_gutter_px=5, max_search_fraction=0.2)
    x, y, w, h = snapped
    assert abs(x - 50) <= 1
    assert abs(y - 50) <= 1
    assert abs((x + w) - 250) <= 1
    assert abs((y + h) - 250) <= 1


def test_ignores_a_narrow_interior_gap_that_isnt_a_real_gutter():
    # A panel with a thin 3px bright stripe near its top edge (like a small
    # gap inside a caption box) shouldn't be mistaken for the page gutter.
    page = np.full((300, 300, 3), 255, dtype=np.uint8)
    page[50:250, 50:250, :] = 120
    page[70:73, 50:250, :] = 255  # thin false gutter, 3px
    undershot = [(50.0, 75.0, 200.0, 175.0)]

    [snapped] = snap_boxes_to_gutters(page, undershot, min_gutter_px=8, max_search_fraction=0.2)
    _x, y, _w, _h = snapped
    assert abs(y - 50) <= 1  # snapped past the thin false gutter to the real edge


def test_leaves_an_already_correct_box_unchanged():
    page = np.full((200, 200, 3), 255, dtype=np.uint8)
    page[40:160, 40:160, :] = 120
    correct = [(40.0, 40.0, 120.0, 120.0)]

    [snapped] = snap_boxes_to_gutters(page, correct)
    assert snapped == (40.0, 40.0, 120.0, 120.0)


def test_no_gutter_found_within_search_range_leaves_box_unchanged():
    page = np.full((200, 200, 3), 120, dtype=np.uint8)  # no gutter anywhere
    box = [(40.0, 40.0, 120.0, 120.0)]

    [snapped] = snap_boxes_to_gutters(page, box, max_search_fraction=0.05)
    assert snapped == (40.0, 40.0, 120.0, 120.0)
