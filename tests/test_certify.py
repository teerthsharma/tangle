"""The certified layer: the interval, both witnesses, the refusals, the next crossing.

The closed-form table these run against is the textbook one -- clasp, drape, stack, the
(2, n) torus family, the Whitehead link -- so nothing here is checked against anything this
package computed for itself.
"""

import pathlib
import random
from dataclasses import replace

import pytest

from tangle import alexander
from tangle.certify import (
    BANNED,
    CERTIFIED,
    INTERLEAVED_ENDS,
    LINKED,
    LK_STRADDLES_ZERO,
    LK_ZERO,
    NOT_CERTIFIED,
    ODD_CROSSING_PARITY,
    OVER_MIXED,
    REFUSED,
    SEPARABLE,
    Interval,
    Verdict,
    achievable,
    brute_force_interval,
    certify,
    intersect,
    lk_interval,
    next_crossing,
    over_everywhere,
    parity_ok,
    r_min,
)
from tangle.diagram import Diagram

FRAME = (0.0, 0.0, 10.0, 10.0)
SEED = 20260905  # printed by test_interval_matches_brute_force

# The Whitehead link, L5a1: the closure of this 3-braid.  Five crossings, two components,
# lk = 0.  It is the reason lk = 0 is never allowed to mean "these come apart".
WHITEHEAD = (1, -2, 1, -2, -2)


def whitehead() -> Diagram:
    return Diagram.from_braid(WHITEHEAD, strands=3)


def stack() -> Diagram:
    """One cable laid flat, the other draped over it and folded back.  R2, so lk = 0, and
    the same cable is over at both crossings, so it lifts off."""
    return Diagram.from_polylines(
        [[(0, 3), (10, 3)], [(0, 7), (4, 1), (6, 1), (10, 7)]],
        over_table=lambda c: "b",
        frame=FRAME,
    )


def three_fingers() -> Diagram:
    """One cable draped over another three times, the middle drape the other way up.

    Six inter-component crossings whose signs cancel, with cable 0 on top at four of them
    and cable 1 on top at two.  lk = 0 with a strict over-minority: the case that has a
    shorter list to name than "everything".
    """
    return Diagram.from_polylines(
        [
            [(0, 3), (10, 3)],
            [(0, 7), (1, 1), (2, 1), (3, 7), (4, 7), (5, 1), (6, 1), (7, 7), (8, 7), (9, 1), (9.5, 1), (10, 7)],
        ],
        over_table=["a", "a", "b", "b", "a", "a"],
        frame=FRAME,
    )


# --------------------------------------------------------------------------------------
# the LINKED certificate (T3)
# --------------------------------------------------------------------------------------


def test_clasp_certifies_linked():
    v = certify(Diagram.from_braid([1, 1]))
    assert v.status == CERTIFIED
    assert v.claim == LINKED
    assert v.witness == "lk"
    assert abs(v.value) == 1
    assert v.interval.unknown == 0
    assert v.exit_code == 0
    assert "unplug" in v.advice.lower()


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_torus_family_certifies_the_right_integer(n):
    """closure(sigma_1^n) for even n is the (2, n) torus link, with |lk| = n/2."""
    v = certify(Diagram.from_braid([1] * n))
    assert v.claim == LINKED
    assert abs(v.value) == n // 2


def test_mirror_negates_lk_and_leaves_the_verdict_alone():
    d = Diagram.from_braid([1] * 4)
    a, b = certify(d), certify(d.mirror())
    assert a.claim == b.claim == LINKED
    assert a.value == -b.value
    assert a.interval.lo == -b.interval.hi and a.interval.hi == -b.interval.lo


# --------------------------------------------------------------------------------------
# the SEPARABLE certificate (T5), pointing the opposite way
# --------------------------------------------------------------------------------------


def test_over_everywhere_certifies_separable():
    v = certify(stack())
    assert v.status == CERTIFIED
    assert v.claim == SEPARABLE
    assert v.witness == "over-everywhere"
    assert over_everywhere(stack()) == 1


def test_flipping_one_crossing_of_the_drape_makes_a_real_clasp():
    """Reading one of the two crossings the other way is not a smaller certificate -- it is
    a different tangle.  The drape becomes a clasp and lk goes from 0 to +/-1."""
    d = stack()
    flip = d.between(0, 1)[0].id
    v = certify(d.resolve({flip: "a"}))
    assert v.claim == LINKED
    assert abs(v.value) == 1


def test_mixed_over_with_lk_zero_names_the_minority_crossings():
    """Three fingers, the middle one draped the other way: lk = 0, both cables are over
    somewhere, and the two crossings where the minority cable is on top are the
    obstructions to separability.  A tie names nothing, and is reported as LK_ZERO."""
    v = certify(three_fingers())
    assert v.status == NOT_CERTIFIED
    assert v.reason == OVER_MIXED
    assert v.look_at == (2, 3)
    assert v.claim == ""
    assert v.value == 0  # the number is still carried


def test_over_everywhere_needs_every_crossing_read():
    d = stack()
    assert over_everywhere(d.resolve({d.between(0, 1)[0].id: None})) is None


def test_over_everywhere_implies_a_zero_signed_sum():
    """T5 + T3 give a free validator: over-everywhere => split => lk = 0 => S = 0."""
    for d in (stack(), Diagram.from_polylines([[(0, 3), (10, 3)], [(0, 7), (10, 7)]], frame=FRAME)):
        if over_everywhere(d) is not None:
            assert lk_interval(d).exact == 0


# --------------------------------------------------------------------------------------
# lk = 0 certifies nothing.  The Whitehead link is why.
# --------------------------------------------------------------------------------------


def test_whitehead_is_lk_zero_and_not_certified():
    d = whitehead()
    assert len(d.cables) == 2 and len(d.crossings) == 5
    v = certify(d)
    assert v.status == NOT_CERTIFIED
    assert v.reason == LK_ZERO
    assert v.value == 0  # the number is carried, the claim is not
    assert v.claim == ""
    assert v.exit_code == 1


def test_whitehead_is_not_split_by_an_independent_invariant():
    """The determinant is 8, and det = 0 for every split link, so the two components of a
    lk = 0 diagram provably do not come apart.  This is the whole reason LK_ZERO exists."""
    d = whitehead()
    assert lk_interval(d).exact == 0
    assert alexander.is_connected(d)
    assert alexander.determinant(d) == 8


# --------------------------------------------------------------------------------------
# the interval theorem (T4) and its enumeration control
# --------------------------------------------------------------------------------------


def test_interval_width_equals_k():
    rng = random.Random(SEED)
    d = Diagram.from_braid([1] * 12)
    ids = [c.id for c in d.between(0, 1)]
    for _ in range(5000):
        k = rng.randint(0, len(ids))
        blur = d.resolve({i: None for i in rng.sample(ids, k)})
        iv = lk_interval(blur)
        assert iv.hi - iv.lo == iv.unknown == k
        assert iv.lo + iv.hi == iv.known_sum


def test_interval_matches_brute_force(capsys):
    """The O(k) interval equals the min, max and achievable set of an explicit 2^k
    enumeration.  This is the test that licenses "no enumeration was performed"."""
    rng = random.Random(SEED)
    d = Diagram.from_braid([1] * 10)
    ids = [c.id for c in d.between(0, 1)]
    disagreements = 0
    trials = 1000
    lifts = 0
    for _ in range(trials):
        k = rng.randint(0, 10)
        blur = d.resolve({i: None for i in rng.sample(ids, k)})
        lifts += 2**k
        fast = lk_interval(blur)
        slow = brute_force_interval(blur)
        if fast != slow or set(achievable(blur)) != set(range(slow.lo, slow.hi + 1)):
            disagreements += 1
    with capsys.disabled():
        print(
            f"\n  O(k) interval vs explicit 2^k enumeration: {disagreements}/{trials} patterns "
            f"disagree ({lifts:,} lifts enumerated, k <= 10, seed {SEED})"
        )
    assert disagreements == 0


def test_interval_is_contiguous_so_the_three_outcomes_are_exhaustive():
    d = Diagram.from_braid([1] * 6)
    blur = d.resolve({c.id: None for c in d.between(0, 1)[:3]})
    iv = lk_interval(blur)
    vals = list(iv)
    assert vals == list(range(iv.lo, iv.hi + 1))
    assert not (any(v > 0 for v in vals) and any(v < 0 for v in vals) and 0 not in vals)


def test_interval_rejects_an_impossible_width():
    with pytest.raises(ValueError):
        Interval(lo=0, hi=3, known_sum=3, unknown=2)


def test_r_min_matches_an_exhaustive_search():
    """T6: the smallest number of crossings whose resolution can make a certificate
    possible.  Necessary, not sufficient -- the search asks whether *some* resolution
    certifies, which is exactly the best case the formula describes."""
    d = Diagram.from_braid([1] * 6)
    ids = [c.id for c in d.between(0, 1)]
    rng = random.Random(SEED + 1)
    for _ in range(40):
        k = rng.randint(1, 6)
        chosen = rng.sample(ids, k)
        blur = d.resolve({i: None for i in chosen})
        iv = lk_interval(blur)
        want = r_min(iv)
        found = None
        for r in range(0, k + 1):
            import itertools

            for subset in itertools.combinations(chosen, r):
                for bits in itertools.product("ab", repeat=r):
                    fixed = blur.resolve(dict(zip(subset, bits)))
                    if lk_interval(fixed).excludes_zero:
                        found = r
                        break
                if found is not None:
                    break
            if found is not None:
                break
        assert found == want, (k, iv, want, found)


# --------------------------------------------------------------------------------------
# the parity theorem as a tracer guard (T2b)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("word,strands", [((1, 1), 2), ((1, 1, 1, 1), 2), (WHITEHEAD, 3)])
def test_parity_refuses_a_single_injected_error(word, strands):
    """Delete one inter-component crossing from a correct diagram: 100% REFUSED, 0%
    certified.  T2(b) is a theorem, not a heuristic."""
    d = Diagram.from_braid(word, strands=strands)
    assert parity_ok(d)
    for victim in d.between(0, 1):
        broken = replace(d, crossings=tuple(c for c in d.crossings if c.id != victim.id))
        assert not parity_ok(broken)
        v = certify(broken)
        assert v.status == REFUSED
        assert v.reason == ODD_CROSSING_PARITY
        assert v.claim == ""


def test_interleaving_is_checked_before_parity():
    """An interleaved diagram legitimately has an odd count; calling that a tracer error
    would be wrong."""
    d = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=FRAME)
    assert d.ends_interleave(0, 1)
    assert parity_ok(d)  # odd count is *correct* here
    v = certify(d)
    assert v.status == REFUSED and v.reason == INTERLEAVED_ENDS


def test_lk_interval_raises_rather_than_returning_a_half_integer():
    d = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=FRAME)
    with pytest.raises(ValueError):
        lk_interval(d)


# --------------------------------------------------------------------------------------
# refusal, and the crossing it names
# --------------------------------------------------------------------------------------


def test_straddling_interval_refuses_and_names_a_crossing():
    d = Diagram.from_braid([1, 1])
    blur = d.resolve({d.between(0, 1)[0].id: None})
    v = certify(blur)
    assert v.status == REFUSED
    assert v.reason == LK_STRADDLES_ZERO
    assert len(v.look_at) == 1
    assert v.exit_code == 2


def test_next_crossing_only_ranks_crossings_that_can_move_this_verdict():
    d = Diagram.from_braid([1, 1, 2], strands=3)  # two inter-component, one self
    self_id = d.self_crossings(0)[0].id if d.self_crossings(0) else d.self_crossings(1)[0].id
    blur = d.resolve({c.id: None for c in d.crossings})
    picked = next_crossing(blur)
    assert picked is not None
    assert picked[0] != self_id
    assert picked[0] in {c.id for c in d.between(0, 1)}
    assert 0.0 <= picked[2] < 180.0


def test_next_crossing_is_none_when_everything_is_read():
    assert next_crossing(Diagram.from_braid([1, 1])) is None


def test_free_end_refuses_and_prints_no_linking_number():
    d = Diagram.from_polylines([[(0, 2), (5, 2)], [(0, 8), (10, 8)]], frame=FRAME)
    v = certify(d)
    assert v.status == REFUSED
    assert v.reason == "FREE_END_IN_FRAME"
    assert v.value is None and v.interval is None


def test_refusal_is_terminal():
    """A diagram that fails parity never reaches the over-everywhere witness, even when
    that witness would have fired."""
    d = stack()
    victim = d.between(0, 1)[0]
    broken = replace(d, crossings=tuple(c for c in d.crossings if c.id != victim.id))
    assert over_everywhere(broken) is not None  # the second witness would fire
    v = certify(broken)
    assert v.status == REFUSED and v.claim == ""


def test_a_bad_pair_index_refuses():
    d = Diagram.from_braid([1, 1])
    assert certify(d, 0, 0).status == REFUSED
    assert certify(d, 0, 5).status == REFUSED


def test_a_non_int_pair_index_raises_a_named_type_error_not_a_bare_comparison_failure():
    """`certify(d, 'a', 'b')` used to reach `0 <= i < len(...)` and raise the numeric
    comparison's own `TypeError: '<=' not supported between instances of 'int' and 'str'`
    three frames in.  A caller's mistake in the *type* of the pair is not the diagram's
    problem, so it is raised at the door with the value named, not returned as a Verdict."""
    d = Diagram.from_braid([1, 1])
    for bad_pair in (("a", "b"), (0, "b"), (None, 1), (0.5, 1), (True, 0)):
        with pytest.raises(TypeError):
            certify(d, *bad_pair)


# --------------------------------------------------------------------------------------
# tau lives inside the certified layer
# --------------------------------------------------------------------------------------


def test_certify_downgrades_low_confidence_reads_itself():
    d = Diagram.from_braid([1, 1])
    shaky = replace(
        d, crossings=tuple(replace(c, over_conf=0.4) for c in d.crossings)
    )
    assert certify(shaky, tau=0.0).claim == LINKED  # trusted: certifies
    v = certify(shaky, tau=0.8)  # not trusted: both crossings become UNKNOWN
    assert v.status == REFUSED and v.reason == LK_STRADDLES_ZERO
    assert v.look_at


# --------------------------------------------------------------------------------------
# multi-view (T7)
# --------------------------------------------------------------------------------------


def test_intersecting_views_narrows_the_interval():
    got = intersect([Interval(0, 2, 2, 2), Interval(1, 3, 4, 2)])
    assert got == Interval(1, 2, 3, 1)


def test_disjoint_views_prove_one_trace_wrong():
    assert intersect([Interval(-1, 0, -1, 1), Interval(1, 2, 3, 1)]) is None


def test_intersecting_one_view_is_the_identity():
    iv = Interval(-2, 1, -1, 3)
    assert intersect([iv]) == iv


# --------------------------------------------------------------------------------------
# the never-claimed list, enforced by tests rather than by discipline
# --------------------------------------------------------------------------------------


def _battery():
    yield Diagram.from_braid([1, 1])
    yield Diagram.from_braid([1, -1])
    yield Diagram.from_braid([1] * 5)
    yield whitehead()
    yield stack()
    yield three_fingers()
    yield stack().resolve({stack().between(0, 1)[0].id: "a"})
    yield Diagram.from_braid([1, 1]).resolve({0: None})
    yield Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=FRAME)
    yield Diagram.from_polylines([[(0, 2), (5, 2)], [(0, 8), (10, 8)]], frame=FRAME)


def test_separable_is_unreachable_from_the_lk_path():
    for d in _battery():
        v = certify(d)
        if v.witness == "lk":
            assert v.claim != SEPARABLE
        if v.claim == SEPARABLE:
            assert v.witness == "over-everywhere"


def test_no_verdict_carries_a_banned_phrase():
    for d in _battery():
        v = certify(d)
        blob = " ".join((v.claim, v.witness, v.reason, v.advice, v.status)).lower()
        for word in BANNED:
            assert word not in blob


def test_the_package_source_carries_no_banned_phrase():
    """Grep the source too.  The line that defines BANNED is the only permitted mention."""
    root = pathlib.Path(__file__).resolve().parent.parent / "tangle"
    hits = []
    for path in sorted(root.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "BANNED" in line or line.lstrip().startswith("#"):
                continue
            low = line.lower()
            hits += [(path.name, n, w) for w in BANNED if w in low]
    assert hits == [], hits


def test_a_claim_without_a_certificate_is_a_hard_error():
    with pytest.raises(ValueError):
        Verdict(status=NOT_CERTIFIED, claim=LINKED)
    with pytest.raises(ValueError):
        Verdict(status=CERTIFIED, claim=LINKED, witness="")
    with pytest.raises(ValueError):
        Verdict(status=REFUSED, advice="safe to pull")


def test_every_verdict_carries_the_convention_and_an_exit_code():
    for d in _battery():
        v = certify(d)
        assert "ends" in v.convention and "fixed" in v.convention
        assert v.exit_code in (0, 1, 2)
        assert v.line() and v.block() and v.json()["convention"] == v.convention
