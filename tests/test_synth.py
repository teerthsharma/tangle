"""The synthetic corpus: is its ground truth actually ground truth?

Every number `test_vision.py` reports is measured against `synth.truth`, so these tests are
about the corpus itself rather than about the reader.  Three things have to hold: the truth
is computed from the scene and not attached to it, the renderer draws what the truth says,
and the picture is reproducible from a seed.
"""

import math

import numpy as np
import pytest

from tangle import synth
from tangle.certify import CERTIFIED, LINKED, SEPARABLE, certify, lk_interval, over_everywhere

SIZE = synth.SIZE
FRAME = (0.0, 0.0, float(SIZE[0] - 1), float(SIZE[1] - 1))
SEEDS = range(24)  # the corpus the reported numbers are measured on


# --------------------------------------------------------------------------------------
# ground truth is analytic
# --------------------------------------------------------------------------------------


def test_truth_is_analytic_on_two_hand_written_segments():
    """One horizontal cable at z = +1, one vertical at z = -1, crossing where they must."""
    a = synth.Cable3(((0.0, 256.0), (511.0, 256.0)), (1.0, 1.0))
    b = synth.Cable3(((200.0, 0.0), (200.0, 511.0)), (-1.0, -1.0))
    d = synth.truth([a, b], SIZE)
    assert len(d.crossings) == 1
    c = d.crossings[0]
    assert c.xy == pytest.approx((200.0, 256.0))
    assert c.angle_deg == pytest.approx(90.0)
    assert c.over == "a" and c.branch("a").cable == 0  # the higher cable is over
    assert d.sign(c) is not None


def test_lowering_one_cable_flips_only_that_crossing():
    a = synth.Cable3(((0.0, 256.0), (511.0, 256.0)), (1.0, 1.0))
    b = synth.Cable3(((200.0, 0.0), (200.0, 511.0)), (-1.0, -1.0))
    up = synth.truth([a, b], SIZE).crossings[0]
    down = synth.truth([a, synth.Cable3(b.xy, (5.0, 5.0))], SIZE).crossings[0]
    assert up.over == "a" and down.over == "b"
    assert up.base == down.base  # the planar factor does not care about height


# --------------------------------------------------------------------------------------
# the two named scenes carry their closed-form answers
# --------------------------------------------------------------------------------------


def test_stack_is_over_everywhere_and_certifies_separable():
    d = synth.truth(synth.stack(), SIZE)
    assert len(d.between(0, 1)) == 2
    assert lk_interval(d).exact == 0
    assert over_everywhere(d, 0, 1) == 0  # cable 0 is the over-strand at both
    v = certify(d)
    assert v.status == CERTIFIED and v.claim == SEPARABLE and v.witness == "over-everywhere"


def test_clasp_certifies_linked_and_mirrors_by_sign():
    plus = synth.truth(synth.clasp(1), SIZE)
    minus = synth.truth(synth.clasp(-1), SIZE)
    assert len(plus.between(0, 1)) == 2
    assert lk_interval(plus).exact == -lk_interval(minus).exact
    for d in (plus, minus):
        v = certify(d)
        assert v.status == CERTIFIED and v.claim == LINKED and abs(v.value) == 1
        assert v.interval.unknown == 0


def test_clasp_and_stack_are_the_same_curves():
    """The two scenes differ only in the height function, so they are the same drawing."""
    for u, v in zip(synth.clasp(), synth.stack()):
        assert u.xy == v.xy


# --------------------------------------------------------------------------------------
# the renderer draws what the truth says
# --------------------------------------------------------------------------------------


def test_the_crossing_core_carries_the_over_cables_colour():
    for scene in (synth.clasp(), synth.clasp(-1), synth.stack(), synth.pile(0)):
        img, d = synth.render(scene)
        for c in d.crossings:
            want = np.array(synth.COLOURS[c.branch(c.over).cable])
            got = img[int(round(c.xy[1])), int(round(c.xy[0]))].astype(int)
            assert np.abs(got - want).max() < 40, (c.id, got.tolist(), want.tolist())


def test_silhouette_carries_no_depth():
    """Swapping which cable is on top leaves the mask and its distance transform alone.

    This is why no purely geometric cue can read over/under, and why `vision.py` reads it
    from colour-separated continuity instead of from the union silhouette.
    """
    from scipy import ndimage

    up, _ = synth.render(synth.clasp(1))
    down, _ = synth.render(synth.clasp(-1))
    m_up = np.any(up != np.array(synth.BG, dtype=np.uint8), axis=-1)
    m_down = np.any(down != np.array(synth.BG, dtype=np.uint8), axis=-1)
    assert np.array_equal(m_up, m_down)
    assert np.array_equal(
        ndimage.distance_transform_edt(m_up), ndimage.distance_transform_edt(m_down)
    )
    assert not np.array_equal(up, down)  # the colours differ, which is the whole cue


def test_render_is_deterministic_in_its_seed():
    a, _ = synth.render(synth.pile(3), noise=3.0, seed=11)
    b, _ = synth.render(synth.pile(3), noise=3.0, seed=11)
    c, _ = synth.render(synth.pile(3), noise=3.0, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_same_colour_changes_the_picture_but_not_the_truth():
    plain, d0 = synth.render(synth.clasp())
    same, d1 = synth.render(synth.clasp(), same_colour=True)
    assert d0.digest() == d1.digest()
    assert not np.array_equal(plain, same)


# --------------------------------------------------------------------------------------
# the random corpus stays inside its stated envelope
# --------------------------------------------------------------------------------------


def test_piles_are_deterministic():
    assert synth.pile(7) == synth.pile(7)
    assert synth.pile(7) != synth.pile(8)


def test_every_pile_is_inside_the_stated_envelope():
    """The envelope is a scope condition on the corpus, so it is asserted, not assumed."""
    counts = []
    for seed in SEEDS:
        d = synth.truth(synth.pile(seed), SIZE)
        inter = d.between(0, 1)
        counts.append(len(inter))
        assert not any(c.is_self for c in d.crossings), seed
        assert synth.MIN_CROSSINGS <= len(inter) <= synth.MAX_CROSSINGS, seed
        assert all(c.angle_deg >= synth.MIN_ANGLE_DEG for c in inter), seed
        assert all(c.base is not None for c in inter), seed
        sep = synth.MIN_SEP_W * synth.WIDTH
        for m in range(len(inter)):
            for n in range(m + 1, len(inter)):
                assert math.dist(inter[m].xy, inter[n].xy) >= sep, (seed, m, n)
        assert d.validate() is None, (seed, d.validate())
        assert not d.ends_interleave(0, 1), seed
        assert lk_interval(d).unknown == 0, seed
    print(f"\n  {len(SEEDS)} piles, {sum(counts)} inter-component crossings, "
          f"{min(counts)}-{max(counts)} per scene")


def test_the_corpus_contains_both_verdicts():
    """A corpus of only linked scenes, or only separable ones, measures half a tool."""
    claims = [certify(synth.truth(synth.pile(s), SIZE)).claim for s in SEEDS]
    assert claims.count(LINKED) >= 4 and claims.count(SEPARABLE) >= 4, claims


def test_the_corpus_tops_out_at_lk_one():
    """A limitation, asserted so that it cannot quietly stop being true.

    An arch weaving across another arch enters and leaves, so consecutive crossings carry
    opposite planar signs and no scene in this family can wrap.  |lk| >= 2 is measured in
    closed form on the (2, n) torus family in `test_certify.py`, never through a camera.
    """
    lks = [lk_interval(synth.truth(synth.pile(s), SIZE)).exact for s in SEEDS]
    assert max(abs(v) for v in lks) == 1, sorted(set(lks))
    print(f"\n  lk over the corpus: {sorted(set(lks))}, "
          f"{sum(1 for v in lks if v)} linked of {len(lks)}")
