"""The Alexander determinant at t = -1, against the closed-form table.

The anchor is the (2, n) torus family, whose determinant is n in closed form for every n.
It is a family rather than a handful of points, it covers knots (n odd) and links (n even)
together, and it is not a fit to anything this package computes.  Everything else --
figure-eight, Whitehead, the Reidemeister moves, the mirror, the parity theorem -- is a
prediction checked against that anchor.
"""

import time

import pytest

from tangle.alexander import (
    CHECKERBOARD_FAILED,
    K_EXCEEDS_BOUND,
    K_MAX,
    OPEN_COMPONENT,
    DeterminantRefused,
    build_shadow,
    det_parity_ok,
    det_values,
    determinant,
    is_connected,
)
from tangle.diagram import Diagram

FRAME = (0.0, 0.0, 10.0, 10.0)

# name -> (braid word, strands, components, crossings, det)
LADDER = {
    "unknot, 0 crossings": ([], 1, 1, 0, 1),
    "unknot, 1 crossing (R1 kink)": ([1], 2, 1, 1, 1),
    "Hopf link": ([1, 1], 2, 2, 2, 2),
    "trefoil 3_1": ([1, 1, 1], 2, 1, 3, 3),
    "(2,4) torus link": ([1] * 4, 2, 2, 4, 4),
    "cinquefoil 5_1": ([1] * 5, 2, 1, 5, 5),
    "(2,6) torus link": ([1] * 6, 2, 2, 6, 6),
    "(2,7) torus knot": ([1] * 7, 2, 1, 7, 7),
    "(2,8) torus link": ([1] * 8, 2, 2, 8, 8),
    "figure-eight 4_1": ([1, -2, 1, -2], 3, 1, 4, 5),
    "Whitehead link L5a1": ([1, -2, 1, -2, -2], 3, 2, 5, 8),
    "2-component unlink": ([], 2, 2, 0, 0),
}


@pytest.mark.parametrize("name", list(LADDER))
def test_closed_form_table(name):
    word, strands, comps, crossings, det = LADDER[name]
    d = Diagram.from_braid(word, strands=strands)
    assert len(d.cables) == comps, name
    assert len(d.crossings) == crossings, name
    assert determinant(d) == det, name


def test_the_unlink_is_split_and_goeritz_is_never_called():
    """A disconnected closed diagram is split, so Delta = 0 by definition."""
    d = Diagram.from_braid([], strands=2)
    assert not is_connected(d)
    assert determinant(d) == 0
    with pytest.raises(ValueError):
        build_shadow(d)  # no crossings: there is nothing to trace


def test_the_torus_family_is_a_line_not_a_handful_of_points():
    got = [determinant(Diagram.from_braid([1] * n)) for n in range(1, 11)]
    assert got == list(range(1, 11))


# --------------------------------------------------------------------------------------
# the three free checks that pin the eta convention with no example at all
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in LADDER if LADDER[n][3] > 0])
def test_euler_characteristic_of_the_traced_faces(name):
    word, strands, _, crossings, _ = LADDER[name]
    d = Diagram.from_braid(word, strands=strands)
    if not is_connected(d):
        pytest.skip("split diagram; the shadow is never built")
    sh = build_shadow(d)
    V, E, F = crossings, 2 * crossings, sh.n_faces
    assert V - E + F == 2
    assert F == crossings + 2


@pytest.mark.parametrize("name", list(LADDER))
def test_parity_theorem(name):
    """Delta(-1) = Delta(1) mod 2, and Delta(1) = +/-1 for a knot, 0 for a link (Torres).
    So det is odd on one component and even on two or more.  This catches a wrong
    row/column deletion and a mis-coloured checkerboard, which the closed-form values
    cannot distinguish because they change |det| but not the component count."""
    word, strands, comps, _, det = LADDER[name]
    d = Diagram.from_braid(word, strands=strands)
    assert det_parity_ok(det, comps), (name, det, comps)
    assert determinant(d) % 2 == (1 if comps == 1 else 0)


@pytest.mark.parametrize("name", [n for n in LADDER if LADDER[n][3] > 0])
def test_both_checkerboard_colourings_agree(name):
    """determinant(verify=True) computes the Goeritz matrix of the white faces and of the
    black faces and refuses if they disagree.  Every ladder entry exercises it."""
    word, strands, _, _, det = LADDER[name]
    d = Diagram.from_braid(word, strands=strands)
    assert determinant(d, verify=True) == det


# --------------------------------------------------------------------------------------
# invariance
# --------------------------------------------------------------------------------------


def test_mirror_leaves_the_determinant_alone():
    """The determinant cannot see chirality.  Saying so with a test is cheaper than saying
    so in prose."""
    for word, strands in ([1, 1, 1], 2), ([1, -2, 1, -2], 3), ([1, -2, 1, -2, -2], 3):
        d = Diagram.from_braid(word, strands=strands)
        assert determinant(d.mirror()) == determinant(d)


def test_r1_kink_leaves_the_determinant_alone():
    plain = Diagram.from_braid([1, 1, 1], strands=2)
    kinked = Diagram.from_braid([1, 1, 1, 2], strands=3)
    assert len(kinked.crossings) == len(plain.crossings) + 1
    assert determinant(kinked) == determinant(plain) == 3


def test_r2_pair_leaves_the_determinant_alone():
    plain = Diagram.from_braid([1, 1, 1], strands=2)
    fingered = Diagram.from_braid([1, 1, 1, 1, -1], strands=2)
    assert len(fingered.crossings) == len(plain.crossings) + 2
    assert determinant(fingered) == determinant(plain) == 3


# --------------------------------------------------------------------------------------
# the 2^k enumeration: det is multi-affine, not affine, so there is no interval
# --------------------------------------------------------------------------------------


def test_one_unreadable_crossing_on_the_trefoil_gives_two_values():
    """Flipping one crossing of the trefoil unknots it, so the achievable set is {1, 3}.
    Two values is not a certificate, and the set is not an interval of integers either --
    which is exactly why the linking number's O(k) shortcut has no analogue here."""
    d = Diagram.from_braid([1, 1, 1])
    assert det_values(d) == (3,)
    assert det_values(d.resolve({0: None})) == (1, 3)


def test_unreadable_self_crossings_still_count_toward_k_det():
    """k_det counts *all* unreadable crossings, self-crossings included, because eta is
    defined at every crossing.  k_lk does not.  The two are different numbers."""
    d = Diagram.from_braid([1, 1, 2], strands=3)  # two inter-component, one self
    self_id = (d.self_crossings(0) or d.self_crossings(1))[0].id
    blurred = d.resolve({self_id: None})
    assert len(blurred.unknown_between(0, 1)) == 0  # k_lk is 0
    assert det_values(blurred) == (2,)  # k_det is 1, and both lifts agree here
    with pytest.raises(DeterminantRefused) as e:
        det_values(blurred, k_max=0)
    assert e.value.reason == K_EXCEEDS_BOUND


def test_values_are_the_set_over_every_lift():
    d = Diagram.from_braid([1] * 4)
    blur = d.resolve({0: None, 1: None})
    vals = det_values(blur)
    brute = sorted({determinant(d.resolve({0: x, 1: y})) for x in "ab" for y in "ab"})
    assert list(vals) == brute


def test_k_max_is_enforced_not_discovered():
    d = Diagram.from_braid([1] * 4)
    blur = d.resolve({c.id: None for c in d.crossings})
    assert det_values(blur, k_max=4)
    with pytest.raises(DeterminantRefused) as e:
        det_values(blur, k_max=3)
    assert e.value.reason == K_EXCEEDS_BOUND
    assert "K_MAX" in str(e.value)
    assert K_MAX >= 4


def test_an_unresolved_crossing_is_never_silently_resolved():
    d = Diagram.from_braid([1, 1, 1]).resolve({0: None})
    with pytest.raises(ValueError):
        determinant(d)


# --------------------------------------------------------------------------------------
# scope: closed diagrams only
# --------------------------------------------------------------------------------------


def test_open_cables_are_refused_not_silently_closed():
    """The closure the lk path deliberately does not have is exactly the thing that would
    be needed here, so an open cable is refused rather than closed by convention."""
    d = Diagram.from_polylines(
        [[(0, 3), (10, 3)], [(0, 7), (4, 1), (6, 1), (10, 7)]],
        over_table=lambda c: "b",
        frame=FRAME,
    )
    with pytest.raises(DeterminantRefused) as e:
        determinant(d)
    assert e.value.reason == OPEN_COMPONENT


# --------------------------------------------------------------------------------------
# the measurement behind K_MAX
# --------------------------------------------------------------------------------------


def test_per_lift_cost(capsys):
    """K_MAX is a wall-clock budget, not a mathematical limit, so it ships with the number
    it was set from.  Run with -s to see it."""
    rows = []
    for n in (6, 10):
        d = Diagram.from_braid([1] * n)
        sh = build_shadow(d)
        reps = 500
        t0 = time.perf_counter()
        for _ in range(reps):
            determinant(d, shadow=sh, verify=False)
        us = (time.perf_counter() - t0) / reps * 1e6
        rows.append((n, us))
    with capsys.disabled():
        print("\n  per-lift determinant cost, shadow hoisted out of the loop:")
        for n, us in rows:
            print(f"    (2,{n}) torus, {n} crossings: {us:6.1f} us   ->  2^{K_MAX} lifts = {us * 2 ** K_MAX / 1e6:5.1f} s")
    assert all(us < 1000 for _, us in rows)
