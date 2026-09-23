from comicreel.character_verification import (
    refine_clusters,
    strongest_inter_cluster_pair,
    weakest_intra_cluster_pair,
)

VEC_A = [1.0, 0.0, 0.0]
VEC_A_CLOSE = [0.95, 0.05, 0.0]
VEC_B = [0.0, 1.0, 0.0]


def test_weakest_intra_cluster_pair_finds_the_least_similar_two():
    members = [VEC_A, VEC_A_CLOSE, VEC_B]
    i, j, sim = weakest_intra_cluster_pair(members)
    assert {i, j} == {0, 2} or {i, j} == {1, 2}  # whichever pairs with VEC_B
    assert sim < 0.5


def test_weakest_intra_cluster_pair_with_uniform_cluster():
    members = [VEC_A, VEC_A_CLOSE]
    i, j, sim = weakest_intra_cluster_pair(members)
    assert {i, j} == {0, 1}
    assert sim > 0.9


def test_weakest_intra_cluster_pair_with_fewer_than_two_members():
    assert weakest_intra_cluster_pair([VEC_A]) is None
    assert weakest_intra_cluster_pair([]) is None


def test_strongest_inter_cluster_pair_finds_the_most_similar_across():
    members_a = [VEC_B, VEC_A_CLOSE]
    members_b = [VEC_A]
    i, j, sim = strongest_inter_cluster_pair(members_a, members_b)
    assert i == 1  # VEC_A_CLOSE in members_a
    assert j == 0  # VEC_A in members_b
    assert sim > 0.9


def test_strongest_inter_cluster_pair_with_empty_cluster():
    assert strongest_inter_cluster_pair([], [VEC_A]) is None
    assert strongest_inter_cluster_pair([VEC_A], []) is None


def _cluster(character_id, members):
    return {
        "character_id": character_id,
        "members": members,
        "crop_paths": [f"{character_id}_{i}.png" for i in range(len(members))],
        "count": len(members),
        "representative_crop": f"{character_id}_0.png",
    }


def test_refine_clusters_splits_a_false_merge():
    # A "cluster" that actually contains two different characters (VEC_A-ish
    # and VEC_B), similar enough to have been merged by centroid drift.
    clusters = [_cluster("char_000", [VEC_A, VEC_A_CLOSE, VEC_B])]

    def verify_fn(a, b):
        return False  # VLM says every checked pair here is a different character

    refined, log = refine_clusters(clusters, verify_fn, split_candidate_max_sim=0.99)

    assert len(refined) == 2
    sizes = sorted(len(c["members"]) for c in refined)
    assert sizes == [1, 2]  # VEC_B split off from {VEC_A, VEC_A_CLOSE}
    assert any(entry["type"] == "split_applied" for entry in log)


def test_refine_clusters_leaves_a_confidently_coherent_cluster_alone():
    clusters = [_cluster("char_000", [VEC_A, VEC_A_CLOSE])]

    def verify_fn(a, b):
        raise AssertionError("should not need to verify a confidently coherent cluster")

    refined, log = refine_clusters(clusters, verify_fn, split_candidate_max_sim=0.5)
    assert len(refined) == 1
    assert len(refined[0]["members"]) == 2
    assert log == []


def test_refine_clusters_merges_two_clusters_of_the_same_character():
    clusters = [_cluster("char_000", [VEC_A]), _cluster("char_001", [VEC_A_CLOSE])]

    def verify_fn(a, b):
        return True  # VLM says these are the same character

    refined, log = refine_clusters(clusters, verify_fn, merge_candidate_min_sim=0.1)

    assert len(refined) == 1
    assert len(refined[0]["members"]) == 2
    assert any(entry["type"] == "merge_applied" for entry in log)


def test_refine_clusters_does_not_merge_implausible_pairs():
    clusters = [_cluster("char_000", [VEC_A]), _cluster("char_001", [VEC_B])]

    def verify_fn(a, b):
        raise AssertionError("should not verify a pair below the merge candidate threshold")

    refined, log = refine_clusters(clusters, verify_fn, merge_candidate_min_sim=0.5)
    assert len(refined) == 2
    assert log == []


def test_refine_clusters_does_not_chain_merges_through_a_growing_cluster():
    # A regression test for a real bug: A and B are genuinely the same
    # character (both near VEC_A). C is a DIFFERENT character (near VEC_B)
    # that a bad VLM call incorrectly says matches B. The bug merged A+B
    # first, then compared the *enlarged* A+B cluster against C using
    # A+B's best-matching member -- which could spuriously include A's
    # strength, pulling C in too. Checking against ORIGINAL members fixes
    # this: C should only be evaluated against B's original members, and if
    # the verify_fn is honest about A-vs-C and B-vs-C individually, C must
    # end up on its own.
    clusters = [
        _cluster("char_000", [VEC_A]),
        _cluster("char_001", [VEC_A_CLOSE]),
        _cluster("char_002", [VEC_B]),
    ]

    def verify_fn(a, b):
        # Only the genuine same-character pair (A, A_CLOSE) says "same".
        # Any pair touching char_002's crop must honestly say "different".
        return "char_002" not in a and "char_002" not in b

    refined, _log = refine_clusters(clusters, verify_fn, merge_candidate_min_sim=0.1)

    assert len(refined) == 2
    sizes = sorted(len(c["members"]) for c in refined)
    assert sizes == [1, 2]  # char_002 stays alone; char_000+char_001 merge


def test_refine_clusters_does_not_transitively_merge_through_an_absorbed_cluster():
    # A regression test for a second, worse version of the chaining bug: a
    # union-find-based fix still transitively merged A+B+C whenever A-B and
    # B-C were each independently verified "same", even though A and C
    # were never checked against each other. Two full benchmark comics
    # collapsed into one giant cluster this way. The correct behavior:
    # once B is absorbed into A, B is "used up" for this pass -- nothing
    # else may merge with it, so C (which only ever matched B, never A)
    # must NOT end up in the final A+B+C group.
    seen_pairs = []

    def verify_fn(a, b):
        seen_pairs.append((a, b))
        # A-B: genuinely the same character. Anything touching C: a
        # hypothetical spurious "same" that must never actually run, since
        # B should already be locked by the time C would be compared to it.
        if "char_002" in a or "char_002" in b:
            raise AssertionError(f"should never verify a pair touching the already-absorbed cluster: {a}, {b}")
        return True

    clusters = [
        _cluster("char_000", [VEC_A]),
        _cluster("char_001", [VEC_A_CLOSE]),
        _cluster("char_002", [VEC_B]),
    ]

    refined, _log = refine_clusters(clusters, verify_fn, merge_candidate_min_sim=-1.0)

    # char_000+char_001 merged; char_002 was never reachable for a check
    # against the now-locked char_001, so it must remain separate.
    assert len(refined) == 2
    sizes = sorted(len(c["members"]) for c in refined)
    assert sizes == [1, 2]
