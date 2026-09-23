from comicreel.character_store import (
    assign_character,
    load_character_names,
    load_store,
    save_store,
)

VEC_A = [1.0, 0.0, 0.0]
VEC_A_CLOSE = [0.9, 0.1, 0.0]
VEC_B = [0.0, 1.0, 0.0]


def test_assign_character_creates_a_new_cluster_when_store_is_empty():
    store = {"clusters": []}
    character_id, similarity, is_new = assign_character(store, VEC_A, threshold=0.75)

    assert is_new
    assert similarity == 0.0
    assert character_id == "char_000"
    assert len(store["clusters"]) == 1
    assert store["clusters"][0]["count"] == 1
    assert store["clusters"][0]["representative_crop"] is None


def test_assign_character_joins_an_existing_cluster_above_threshold():
    store = {"clusters": []}
    assign_character(store, VEC_A, threshold=0.75)

    character_id, similarity, is_new = assign_character(store, VEC_A_CLOSE, threshold=0.75)

    assert not is_new
    assert character_id == "char_000"
    assert similarity > 0.75
    assert len(store["clusters"]) == 1  # no new cluster created
    assert store["clusters"][0]["count"] == 2  # centroid updated, not replaced


def test_assign_character_starts_a_second_cluster_for_a_dissimilar_embedding():
    store = {"clusters": []}
    assign_character(store, VEC_A, threshold=0.75)

    character_id, _similarity, is_new = assign_character(store, VEC_B, threshold=0.75)

    assert is_new
    assert character_id == "char_001"
    assert len(store["clusters"]) == 2


def test_store_round_trips_through_disk(tmp_path):
    store = {"clusters": []}
    assign_character(store, VEC_A, threshold=0.75)
    store["clusters"][0]["representative_crop"] = "some/crop.png"

    save_store("comic_1", tmp_path, "clip", store)
    reloaded = load_store("comic_1", tmp_path, "clip")

    assert reloaded == store


def test_load_store_with_no_prior_data_returns_empty():
    assert load_store("nonexistent_comic", "/tmp/does-not-exist-xyz", "clip") == {"clusters": []}


def test_different_backends_get_independent_stores(tmp_path):
    clip_store = {"clusters": []}
    assign_character(clip_store, VEC_A, threshold=0.75)
    save_store("comic_1", tmp_path, "clip", clip_store)

    assert load_store("comic_1", tmp_path, "dinov2") == {"clusters": []}


def test_load_character_names_with_no_yaml_file(tmp_path):
    assert load_character_names("comic_1", tmp_path) == {}


def test_load_character_names_reads_overrides(tmp_path):
    comic_dir = tmp_path / "comic_1"
    comic_dir.mkdir()
    (comic_dir / "characters.yaml").write_text("char_000: Atomic Mouse\nchar_001: Grandpa\n")

    names = load_character_names("comic_1", tmp_path)
    assert names == {"char_000": "Atomic Mouse", "char_001": "Grandpa"}


def test_complete_linkage_mode_rejects_what_centroid_mode_would_accept():
    # A cluster with both VEC_A and VEC_B has a centroid VEC_A_DRIFTED is
    # close enough to under centroid mode, but complete-linkage requires
    # matching every member, including the dissimilar VEC_B.
    vec_a, vec_b, vec_a_drifted = [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]

    centroid_store = {"clusters": []}
    assign_character(centroid_store, vec_a, threshold=0.5, mode="centroid")
    assign_character(centroid_store, vec_b, threshold=0.5, mode="centroid")
    _, _, is_new_centroid = assign_character(
        centroid_store, vec_a_drifted, threshold=0.5, mode="centroid"
    )

    linkage_store = {"clusters": []}
    assign_character(linkage_store, vec_a, threshold=0.9, mode="complete_linkage")
    _, _, is_new_linkage = assign_character(
        linkage_store, vec_b, threshold=0.9, mode="complete_linkage"
    )

    assert not is_new_centroid  # joined the one cluster both belong to
    assert is_new_linkage  # vec_b doesn't clear 0.9 against vec_a -> new cluster


def test_complete_linkage_mode_keeps_full_member_history(tmp_path):
    store = {"clusters": []}
    assign_character(store, [1.0, 0.0], threshold=0.5, mode="complete_linkage")
    assign_character(store, [0.9, 0.1], threshold=0.5, mode="complete_linkage")

    assert len(store["clusters"]) == 1
    assert len(store["clusters"][0]["members"]) == 2
