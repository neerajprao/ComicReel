from comicreel.character_clustering import (
    assign_embedding,
    assign_embedding_complete_linkage,
    best_complete_linkage_match,
    best_match,
    complete_linkage_similarity,
    cosine_similarity,
    update_centroid,
)

VEC_A = [1.0, 0.0, 0.0]
VEC_A_CLOSE = [0.9, 0.1, 0.0]  # nearly identical direction to VEC_A
VEC_A_DRIFTED = [0.5, 0.5, 0.0]  # still closer to A's centroid than B's, but not to A itself
VEC_B = [0.0, 1.0, 0.0]  # orthogonal to VEC_A


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity(VEC_A, VEC_A) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity(VEC_A, VEC_B) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert cosine_similarity(VEC_A, [0.0, 0.0, 0.0]) == 0.0


def test_best_match_picks_the_closer_centroid():
    idx, similarity = best_match(VEC_A, [VEC_B, VEC_A_CLOSE])
    assert idx == 1
    assert similarity > 0.9


def test_best_match_with_no_centroids():
    assert best_match(VEC_A, []) == (None, 0.0)


def test_update_centroid_is_a_running_mean():
    # One prior member at [0, 0], adding [2, 2] should move the centroid to
    # the midpoint [1, 1], not just overwrite it.
    updated = update_centroid([0.0, 0.0], member_count=1, new_embedding=[2.0, 2.0])
    assert updated == [1.0, 1.0]


def test_assign_embedding_joins_cluster_above_threshold():
    idx, similarity = assign_embedding(VEC_A, [VEC_A_CLOSE], threshold=0.9)
    assert idx == 0
    assert similarity > 0.9


def test_assign_embedding_starts_new_cluster_below_threshold():
    idx, similarity = assign_embedding(VEC_A, [VEC_B], threshold=0.9)
    assert idx is None
    assert similarity == 0.0


def test_assign_embedding_with_no_existing_clusters():
    assert assign_embedding(VEC_A, [], threshold=0.9) == (None, 0.0)


def test_complete_linkage_similarity_is_the_minimum_over_members():
    # VEC_A_CLOSE is close to VEC_A but not to VEC_B -- the minimum should
    # reflect the worse (VEC_B) pairing, not an average.
    score = complete_linkage_similarity(VEC_A_CLOSE, [VEC_A, VEC_B])
    assert score == cosine_similarity(VEC_A_CLOSE, VEC_B)


def test_complete_linkage_similarity_with_no_members():
    assert complete_linkage_similarity(VEC_A, []) == 0.0


def test_best_complete_linkage_match_picks_the_cluster_matching_all_members():
    idx, score = best_complete_linkage_match(VEC_A, [[VEC_A, VEC_B], [VEC_A, VEC_A_CLOSE]])
    assert idx == 1  # cluster 0 is dragged down by VEC_B, cluster 1 isn't
    assert score > 0.9


def test_best_complete_linkage_match_with_no_clusters():
    assert best_complete_linkage_match(VEC_A, []) == (None, 0.0)


def test_complete_linkage_is_stricter_than_centroid_matching():
    # A cluster containing both VEC_A and VEC_B has a centroid roughly
    # between them, which VEC_A_DRIFTED is close enough to under centroid
    # matching -- but complete-linkage should reject it, since it isn't
    # close to VEC_B, an actual member.
    centroid = [(a + b) / 2 for a, b in zip(VEC_A, VEC_B, strict=True)]
    centroid_idx, _ = assign_embedding(VEC_A_DRIFTED, [centroid], threshold=0.8)
    assert centroid_idx == 0  # centroid mode joins it

    linkage_idx, _ = assign_embedding_complete_linkage(
        VEC_A_DRIFTED, [[VEC_A, VEC_B]], threshold=0.8
    )
    assert linkage_idx is None  # complete-linkage rejects it
