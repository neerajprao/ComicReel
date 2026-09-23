from comicreel.speaker_attribution import (
    attribute_speaker,
    nearest_character_attribution,
    tail_pointing_attribution,
)

# A bubble at top-left, with two characters below it: one directly under the
# tail (the "correct" speaker) and one far off to the side (a distractor).
BUBBLE = (0.0, 0.0, 100.0, 60.0)  # center (50, 30)
TAIL_POINT = (50.0, 60.0)  # bottom-center of the bubble, pointing straight down
SPEAKER = (20.0, 100.0, 60.0, 100.0)  # directly below the tail -- center (50, 150)
DISTRACTOR = (400.0, 90.0, 60.0, 100.0)  # far to the right, same rough height


def test_tail_pointing_finds_the_aligned_character():
    idx, confidence = tail_pointing_attribution(BUBBLE, TAIL_POINT, [DISTRACTOR, SPEAKER])
    assert idx == 1  # SPEAKER, not DISTRACTOR
    assert confidence > 0.9  # nearly dead-on alignment


def test_tail_pointing_rejects_a_character_outside_max_angle():
    # Only the distractor exists, and it's off to the side, not downward.
    idx, confidence = tail_pointing_attribution(BUBBLE, TAIL_POINT, [DISTRACTOR], max_angle_deg=30)
    assert idx is None
    assert confidence == 0.0


def test_tail_pointing_with_no_characters():
    assert tail_pointing_attribution(BUBBLE, TAIL_POINT, []) == (None, 0.0)


def test_nearest_character_picks_the_closer_one():
    idx, confidence = nearest_character_attribution(BUBBLE, [DISTRACTOR, SPEAKER], page_diagonal=600)
    assert idx == 1  # SPEAKER is much closer to the bubble than DISTRACTOR
    assert 0.0 < confidence < 1.0


def test_nearest_character_with_no_characters():
    assert nearest_character_attribution(BUBBLE, [], page_diagonal=600) == (None, 0.0)


def test_attribute_speaker_prefers_tail_pointing_when_available():
    idx, confidence = attribute_speaker(BUBBLE, TAIL_POINT, [DISTRACTOR, SPEAKER], page_diagonal=600)
    assert idx == 1
    assert confidence > 0.9


def test_attribute_speaker_falls_back_to_nearest_when_tail_doesnt_line_up():
    # Tail points down, but the only character is off to the side -- tail
    # pointing rejects it, so nearest-character fallback should still pick it.
    idx, confidence = attribute_speaker(BUBBLE, TAIL_POINT, [DISTRACTOR], page_diagonal=600)
    assert idx == 0
    assert confidence > 0.0


def test_attribute_speaker_with_no_tail_uses_nearest_character():
    idx, confidence = attribute_speaker(BUBBLE, None, [DISTRACTOR, SPEAKER], page_diagonal=600)
    assert idx == 1
    assert confidence > 0.0
