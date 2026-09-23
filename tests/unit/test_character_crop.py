from comicreel.character_crop import head_crop_bbox

BOX = (10.0, 20.0, 100.0, 200.0)  # x, y, w, h


def test_head_crop_keeps_the_top_of_the_box():
    _x, y, _w, h = head_crop_bbox(BOX, top_fraction=0.5, width_fraction=1.0)
    assert y == 20.0  # top edge unchanged
    assert h == 100.0  # half the original height


def test_head_crop_narrows_width_around_the_center():
    x, _y, w, _h = head_crop_bbox(BOX, top_fraction=1.0, width_fraction=0.5)
    assert w == 50.0
    # centered: original x=10, w=100 -> center x=60; new w=50 -> new x=35
    assert x == 35.0


def test_head_crop_defaults_produce_a_smaller_box():
    _x, y, w, h = head_crop_bbox(BOX)
    assert 0 < w < BOX[2]
    assert 0 < h < BOX[3]
    assert y == BOX[1]
