"""Runs the CLIP and DINOv2 character-embedding backends on a real sample
page and checks the result validates against CharacterIndex. Needs the
~2GB Magi model plus each embedder's own weights (~600MB CLIP, ~330MB
DINOv2-base), so slower than the rest of the suite -- run explicitly when
touching character_embedding, not expected in a tight edit loop.
"""

from pathlib import Path

from comicreel.artifacts.schemas import CharacterIndex
from comicreel.stages.character_embedding import ClipCharacterEmbedder, DinoV2CharacterEmbedder

SAMPLE_PAGE = Path(__file__).resolve().parent.parent / "fixtures/sample_pages/page_04.jpg"


def _run_and_validate(stage_cls, tmp_path, backend_name):
    stage = stage_cls(options={"cache_root": str(tmp_path)})
    result = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )

    validated = CharacterIndex(**result)
    assert len(validated.characters) > 0  # this page has visible characters
    assert len(validated.panel_assignments) > 0

    for character in validated.characters:
        assert character.representative_crop is not None
        assert Path(character.representative_crop).exists()

    for assignment in validated.panel_assignments:
        assert assignment.panel_id.startswith("page_04_p")
        assert 0.0 <= assignment.similarity <= 1.0
        assert assignment.character_id in {c.character_id for c in validated.characters}

    return validated


def test_clip_character_output_matches_schema(tmp_path):
    _run_and_validate(ClipCharacterEmbedder, tmp_path, "clip")


def test_dinov2_character_output_matches_schema(tmp_path):
    _run_and_validate(DinoV2CharacterEmbedder, tmp_path, "dinov2")


def test_repeated_pass_reuses_clusters_instead_of_duplicating(tmp_path):
    """Running the same page twice against the same store should mostly
    re-match existing clusters (near-identical crops), not double the
    character count -- a basic sanity check on the incremental clustering,
    not just the schema."""
    stage = ClipCharacterEmbedder(options={"cache_root": str(tmp_path)})
    first = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )
    second = stage.run(
        "integration_test", "page_04", {"_input": {"image_path": str(SAMPLE_PAGE)}}
    )
    assert len(second["characters"]) == len(first["characters"])
