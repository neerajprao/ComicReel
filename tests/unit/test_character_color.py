from PIL import Image

from comicreel.character_color import color_histogram, histogram_similarity


def _solid_image(rgb: tuple[int, int, int], size: int = 40) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


def test_color_histogram_sums_to_one_for_a_saturated_image():
    hist = color_histogram(_solid_image((230, 30, 30)))  # saturated red
    assert abs(sum(hist) - 1.0) < 1e-6


def test_color_histogram_is_all_zero_for_a_grayscale_image():
    hist = color_histogram(_solid_image((128, 128, 128)))
    assert hist == [0.0] * 16


def test_color_histogram_is_all_zero_for_near_white_background():
    hist = color_histogram(_solid_image((255, 255, 255)))
    assert hist == [0.0] * 16


def test_histogram_similarity_identical_is_one():
    hist = color_histogram(_solid_image((230, 30, 30)))
    assert histogram_similarity(hist, hist) == 1.0


def test_histogram_similarity_disjoint_colors_is_zero():
    red_hist = color_histogram(_solid_image((230, 20, 20)))
    blue_hist = color_histogram(_solid_image((20, 20, 230)))
    assert histogram_similarity(red_hist, blue_hist) == 0.0


def test_histogram_similarity_is_symmetric():
    red_hist = color_histogram(_solid_image((230, 20, 20)))
    orange_hist = color_histogram(_solid_image((230, 130, 20)))
    assert histogram_similarity(red_hist, orange_hist) == histogram_similarity(
        orange_hist, red_hist
    )
