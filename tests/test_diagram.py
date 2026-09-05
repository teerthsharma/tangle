"""The data model: crossings from geometry, signs from orientation, and the ordered checks.

Every diagram here is hand-written or a braid closure, so the certified layer is tested
with no camera, no tracer and no dataset.
"""

import math

import pytest

from tangle.certify import INTERLEAVED_ENDS, REFUSED, certify, lk_interval
from tangle.diagram import SIN_MIN, Diagram

FRAME = (0.0, 0.0, 10.0, 10.0)


def signed_sum(d: Diagram, i: int, j: int) -> int:
    return sum(s for s in (d.sign(c) for c in d.between(i, j)) if s is not None)


# --------------------------------------------------------------------------------------
# crossings from geometry
# --------------------------------------------------------------------------------------


def test_hopf_has_two_inter_component_crossings_of_equal_sign():
    d = Diagram.from_braid([1, 1])
    assert len(d.cables) == 2
    assert len(d.crossings) == 2
    assert all(not c.is_self for c in d.crossings)
    signs = [d.sign(c) for c in d.between(0, 1)]
    assert signs[0] is not None and signs[0] == signs[1]
    assert abs(sum(signs)) == 2
    assert abs(lk_interval(d).exact) == 1


def test_two_disjoint_arcs_have_no_crossings():
    d = Diagram.from_polylines([[(0, 3), (10, 3)], [(0, 7), (10, 7)]], frame=FRAME)
    assert d.crossings == ()
    assert lk_interval(d).exact == 0


def test_self_crossings_are_not_inter_component():
    d = Diagram.from_braid([1, 1, 2], strands=3)  # Hopf link plus one R1 kink
    assert len(d.crossings) == 3
    assert len(d.between(0, 1)) == 2
    assert len(d.self_crossings(0)) + len(d.self_crossings(1)) == 1
    with pytest.raises(ValueError):
        d.between(0, 0)


def test_crossing_order_is_geometric_not_detection_order():
    """Crossing ids come from (lowest cable, arclength, x, y), so resampling the same
    curves at a different vertex density must not renumber anything."""
    coarse = [[(0, 3), (10, 3)], [(0, 7), (4, 1), (6, 1), (10, 7)]]

    def subdivide(poly, n=4):
        out = [poly[0]]
        for p, q in zip(poly, poly[1:]):
            for t in range(1, n + 1):
                out.append((p[0] + (q[0] - p[0]) * t / n, p[1] + (q[1] - p[1]) * t / n))
        return out

    a = Diagram.from_polylines(coarse, over_table=lambda c: "b", frame=FRAME)
    b = Diagram.from_polylines([subdivide(p) for p in coarse], over_table=lambda c: "b", frame=FRAME)
    assert len(a.crossings) == len(b.crossings) == 2
    for ca, cb in zip(a.crossings, b.crossings):
        assert ca.id == cb.id
        assert ca.cables == cb.cables
        assert ca.over == cb.over
        assert ca.base == cb.base
        assert math.dist(ca.xy, cb.xy) < 1e-9


def test_angle_deg_is_the_crossing_angle():
    d = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=FRAME)
    assert len(d.crossings) == 1
    assert d.crossings[0].angle_deg == pytest.approx(90.0)


# --------------------------------------------------------------------------------------
# signs from orientation
# --------------------------------------------------------------------------------------


def test_mirror_negates_every_sign():
    d = Diagram.from_braid([1, 1, 1, -2, 2, -2], strands=3)
    m = d.mirror()
    for c, mc in zip(d.crossings, m.crossings):
        assert d.sign(c) == -m.sign(mc)
        assert c.base == mc.base  # the planar factor is untouched; only x_c flips


def test_unknown_kills_the_sign_and_only_that_sign():
    d = Diagram.from_braid([1] * 4)
    blur = d.resolve({0: None, 2: None})
    assert len(blur.unknown_between(0, 1)) == 2
    for c in blur.crossings:
        if c.id in (0, 2):
            assert blur.sign(c) is None and c.kind == "unknown"
        else:
            assert blur.sign(c) == d.sign(c)


def test_base_is_unknown_below_sin_min():
    """A near-tangential crossing has an unreliable tangent, so base becomes None.  That is
    the same unknown +/-1 as an unreadable over/under and folds into the same k."""
    d = Diagram.from_polylines(
        [[(0, 5), (10, 5)], [(0, 4.9), (10, 5.1)]], over_table=lambda c: "a", frame=FRAME
    )
    assert len(d.crossings) == 1
    c = d.crossings[0]
    assert math.sin(math.radians(c.angle_deg)) < SIN_MIN
    assert c.base is None
    assert c.over == "a" and d.sign(c) is None
    assert len(d.unknown_between(0, 1)) == 1


def test_resolve_is_a_round_trip():
    d = Diagram.from_braid([1] * 5, strands=2)
    ids = [c.id for c in d.crossings]
    blurred = d.resolve({i: None for i in ids})
    assert all(c.over is None for c in blurred.crossings)
    back = blurred.resolve({c.id: d.crossings[c.id].over for c in blurred.crossings})
    assert [c.over for c in back.crossings] == [c.over for c in d.crossings]
    assert back.digest() == d.digest()


# --------------------------------------------------------------------------------------
# endpoints, the frame, and the ordered checks
# --------------------------------------------------------------------------------------


def test_free_end_in_frame_is_a_defect():
    ok = Diagram.from_polylines([[(0, 2), (10, 2)], [(0, 8), (10, 8)]], frame=FRAME)
    assert ok.validate() is None
    loose = Diagram.from_polylines([[(0, 2), (5, 2)], [(0, 8), (10, 8)]], frame=FRAME)
    assert loose.validate() == "FREE_END_IN_FRAME"


def test_exit_order_and_interleaving():
    crossed = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=FRAME)
    assert crossed.ends_interleave(0, 1) is True
    nested = Diagram.from_polylines([[(0, 3), (10, 3)], [(0, 7), (10, 7)]], frame=FRAME)
    assert nested.ends_interleave(0, 1) is False
    assert len(crossed.exit_order) == len(nested.exit_order) == 4


def test_closed_cables_never_interleave():
    d = Diagram.from_braid([1, 1])
    assert d.exit_order == ()
    assert d.ends_interleave(0, 1) is False


def test_triple_point_is_a_defect():
    d = Diagram.from_polylines(
        [[(0, 5), (10, 5)], [(5, 0), (5, 10)], [(0, 0), (10, 10)]], frame=FRAME
    )
    assert len(d.crossings) == 3
    assert d.validate() == "TRIPLE_POINT"


def test_digest_is_stable_and_sensitive():
    d = Diagram.from_braid([1, 1])
    assert d.digest() == Diagram.from_braid([1, 1]).digest()
    assert d.digest() != d.mirror().digest()
    assert d.digest() != d.resolve({0: None}).digest()


# --------------------------------------------------------------------------------------
# diagram-move invariance -- weaker than camera invariance, and all this layer can test
# --------------------------------------------------------------------------------------


def test_r1_kink_leaves_lk_alone():
    plain = Diagram.from_braid([1, 1])
    kinked = Diagram.from_braid([1, 1, 2], strands=3)
    assert len(kinked.crossings) == len(plain.crossings) + 1
    assert lk_interval(plain).exact == lk_interval(kinked).exact


def test_r2_finger_adds_two_cancelling_crossings():
    flat = Diagram.from_polylines(
        [[(0, 3), (10, 3)], [(0, 7), (10, 7)]], over_table=lambda c: "b", frame=FRAME
    )
    finger = Diagram.from_polylines(
        [[(0, 3), (10, 3)], [(0, 7), (4, 1), (6, 1), (10, 7)]],
        over_table=lambda c: "b",
        frame=FRAME,
    )
    assert len(flat.between(0, 1)) == 0
    assert len(finger.between(0, 1)) == 2
    signs = [finger.sign(c) for c in finger.between(0, 1)]
    assert sorted(signs) == [-1, 1]  # opposite signs, so they cancel
    assert lk_interval(flat).exact == lk_interval(finger).exact == 0


def _r3(x: float) -> Diagram:
    """Three strands; the vertical one slides across the crossing of the other two."""
    priority = {2: 3, 0: 2, 1: 1}  # cable 2 over cable 0 over cable 1, everywhere

    def over(c):
        return "a" if priority[c.a.cable] > priority[c.b.cable] else "b"

    return Diagram.from_polylines(
        [[(0, 4), (10, 6)], [(0, 6), (10, 4)], [(x, 0), (x, 10)]],
        over_table=over,
        frame=FRAME,
    )


def test_r3_slide_preserves_every_pairwise_signed_sum():
    before, after = _r3(3.0), _r3(7.0)
    assert len(before.crossings) == len(after.crossings) == 3
    moved = {c.xy for c in before.crossings} ^ {c.xy for c in after.crossings}
    assert len(moved) == 4  # two crossings genuinely moved; the third did not
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert signed_sum(before, i, j) == signed_sum(after, i, j)


def test_r3_diagrams_refuse_because_each_pair_crosses_once():
    """T2(a): an odd number of inter-component crossings means the ends interleave and lk
    is a half-integer.  The refusal is the framing, not the tangle."""
    d = _r3(3.0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert len(d.between(i, j)) == 1
        assert d.ends_interleave(i, j) is True
        v = certify(d, i, j)
        assert v.status == REFUSED and v.reason == INTERLEAVED_ENDS


# --------------------------------------------------------------------------------------
# the braid closure constructor
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,strands,cables,crossings",
    [
        ([], 2, 2, 0),
        ([1], 2, 1, 1),
        ([1, 1], 2, 2, 2),
        ([1, 1, 1], 2, 1, 3),
        ([1] * 6, 2, 2, 6),
        ([1, -2, 1, -2], 3, 1, 4),
        ([1, -2, 1, -2, -2], 3, 2, 5),
    ],
)
def test_from_braid_component_and_crossing_counts(word, strands, cables, crossings):
    d = Diagram.from_braid(word, strands=strands)
    assert len(d.cables) == cables
    assert len(d.crossings) == crossings
    assert all(cab.closed for cab in d.cables)
    assert d.validate() is None


def test_from_braid_reads_over_under_from_the_word():
    pos = Diagram.from_braid([1, 1])
    neg = Diagram.from_braid([-1, -1])
    assert [pos.sign(c) for c in pos.between(0, 1)] == [-neg.sign(c) for c in neg.between(0, 1)]


def test_from_braid_rejects_a_zero_letter():
    with pytest.raises(ValueError):
        Diagram.from_braid([1, 0, 1])
