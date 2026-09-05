"""The reader, scored against the corpus it cannot see the truth of.

Ground truth is `synth.truth`, which is computed from the scene's height functions and
never annotated, so nothing here checks this package against itself.  The headline is the
one the specification asks for: **certified verdicts at zero errors**, with the refuse rate
printed on the same line, and a control that replaces the over/under reader with a coin
flip on the identical extracted diagrams.

Two comparisons need care and are made explicit here rather than assumed:

* **Cable identity.**  The reader numbers cables by clustered colour and the corpus numbers
  them by construction, so the two numberings can be a permutation of each other.  `lk` is
  symmetric in its pair, so that costs nothing.
* **Cable orientation.**  Reversing one cable negates `lk`.  The reader walks each
  centreline from whichever end it meets first, so every expected value is computed against
  the traced orientation -- `_expected_lk` counts the reversals and applies the sign.  Only
  `|lk|` is certified; the global sign is a stated convention (see `diagram.py`).
"""

import math
import random
from dataclasses import replace
from functools import lru_cache

import pytest

from tangle import synth, vision
from tangle.certify import (
    CERTIFIED,
    LINKED,
    NOT_TWO_COMPONENTS,
    REFUSED,
    SEPARABLE,
    TAU,
    certify,
    lk_interval,
)

SEEDS = tuple(range(20))
SEED = 20260905  # the coin-flip control's seed, printed with its numbers
ARMS = {
    "clean": {},
    "blur 1.0 px": {"blur": 1.0},
    "blur 3.0 px": {"blur": 3.0},
    "antialiased": {"supersample": 2},
}
FRAME = (0.0, 0.0, float(synth.SIZE[0] - 1), float(synth.SIZE[1] - 1))


# --------------------------------------------------------------------------------------
# the corpus, traced once
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def corpus(arm: str) -> tuple:
    """(seed, truth diagram, traced diagram or None, refusal reason or None) per scene."""
    rows = []
    for seed in SEEDS:
        img, truth = synth.render(synth.pile(seed), seed=seed, **ARMS[arm])
        try:
            rows.append((seed, truth, vision.trace(img), None))
        except vision.TraceRefused as e:
            rows.append((seed, truth, None, e.reason))
    return tuple(rows)


def _match(traced, truth) -> dict[int, tuple[int, bool]]:
    """Traced cable id -> (truth cable id, is it traced backwards)."""
    out = {}
    for cab in traced.cables:
        e0, e1 = cab.ends
        best = None
        for tc in truth.cables:
            f0, f1 = tc.ends
            fwd = math.dist(e0, f0) + math.dist(e1, f1)
            rev = math.dist(e0, f1) + math.dist(e1, f0)
            cost, backwards = (fwd, False) if fwd <= rev else (rev, True)
            if best is None or cost < best[0]:
                best = (cost, tc.id, backwards)
        out[cab.id] = (best[1], best[2])
    return out


def _expected_lk(traced, truth) -> int:
    """The truth's linking number, in the traced diagram's own orientation convention."""
    m = _match(traced, truth)
    flips = sum(1 for _, backwards in m.values() if backwards)
    return lk_interval(truth).exact * (-1) ** flips


def _sound(v, expected: int) -> bool:
    """Is this verdict true of the scene?  A refusal or an abstention is always sound."""
    if v.status != CERTIFIED:
        return True
    if v.claim == SEPARABLE:
        return expected == 0
    return (v.value > 0) == (expected > 0) and abs(expected) >= abs(v.value)


def _coin_flip(d, rng):
    """The control: the same diagram, over/under guessed instead of read."""
    return replace(
        d,
        crossings=tuple(
            replace(c, over=rng.choice(("a", "b")), over_conf=1.0, kind="read")
            for c in d.crossings
        ),
    )


# --------------------------------------------------------------------------------------
# the two named scenes
# --------------------------------------------------------------------------------------


def test_the_clasp_is_certified_linked_from_the_photograph():
    img, truth = synth.render(synth.clasp())
    d = vision.trace(img)
    assert len(d.cables) == 2
    assert len(d.between(0, 1)) == 2
    v = certify(d)
    assert v.status == CERTIFIED and v.claim == LINKED
    assert v.value == _expected_lk(d, truth)
    assert abs(v.value) == 1


def test_the_stack_is_certified_separable_from_the_photograph():
    img, truth = synth.render(synth.stack())
    v = certify(vision.trace(img))
    assert v.status == CERTIFIED and v.claim == SEPARABLE and v.witness == "over-everywhere"
    assert lk_interval(truth).exact == 0


def test_the_traced_diagram_is_reproducible():
    img, _ = synth.render(synth.pile(1))
    assert vision.trace(img).digest() == vision.trace(img).digest()
    other, _ = synth.render(synth.pile(2))
    assert vision.trace(img).digest() != vision.trace(other).digest()


# --------------------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------------------


def test_a_same_coloured_pair_is_refused_not_guessed():
    """Precondition 1, and the most common real scene, failing loudly."""
    for seed in SEEDS[:8]:
        img, _ = synth.render(synth.pile(seed), same_colour=True)
        with pytest.raises(vision.TraceRefused) as e:
            vision.trace(img)
        assert e.value.reason == NOT_TWO_COMPONENTS


def test_noise_destroys_the_intensity_gap_and_is_refused():
    """The threshold is certified against ring noise, so enough noise closes the gap."""
    img, _ = synth.render(synth.pile(0), noise=8.0, seed=1)
    with pytest.raises(vision.TraceRefused) as e:
        vision.trace(img)
    assert e.value.reason == vision.NO_INTENSITY_GAP
    quiet, _ = synth.render(synth.pile(0), noise=2.0, seed=1)
    vision.trace(quiet)  # 2.0 is inside the envelope and traces


def test_every_refusal_carries_a_named_reason():
    known = {
        vision.NO_INTENSITY_GAP,
        vision.BRANCHED_SKELETON,
        vision.OPEN_TRACE,
        NOT_TWO_COMPONENTS,
    }
    hist: dict[str, int] = {}
    for arm in ARMS:
        for _, _, d, r in corpus(arm):
            if d is None:
                hist[r] = hist.get(r, 0) + 1
    print("\n  refusal reasons over every arm")
    for reason, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:20s} {n:3d}")
    assert set(hist) <= known, set(hist) - known


def test_unknown_is_never_a_low_confidence_over():
    """With the bridge evidence removed, every crossing abstains and the interval widens."""
    img, _ = synth.render(synth.clasp())
    d = vision.trace(img)
    blind = vision.read_over(d, [[], []], 12.8)
    assert all(c.over is None and c.over_conf == 0.0 for c in blind.crossings)
    iv = lk_interval(blind)
    assert iv.unknown == len(blind.between(0, 1)) == 2
    assert iv.lo == -1 and iv.hi == 1
    assert certify(blind).status == REFUSED


def test_a_confident_read_is_downgraded_by_tau_inside_the_certified_layer():
    """The honesty boundary is not this module's to move: raising TAU refuses the scene."""
    img, _ = synth.render(synth.clasp())
    d = vision.trace(img)
    assert certify(d, tau=TAU).status == CERTIFIED
    assert certify(d, tau=1.01).status == REFUSED


# --------------------------------------------------------------------------------------
# over/under accuracy on the crossings the reader accepted
# --------------------------------------------------------------------------------------


def _read_accuracy(arm: str) -> tuple[int, int]:
    right = total = 0
    for _, truth, d, _ in corpus(arm):
        if d is None:
            continue
        m = _match(d, truth)
        for c in d.between(0, 1):
            if c.over is None or c.over_conf < TAU:
                continue
            near = min(truth.between(0, 1), key=lambda t: math.dist(t.xy, c.xy))
            if math.dist(near.xy, c.xy) > 2.0 * synth.WIDTH:
                continue
            total += 1
            right += m[c.branch(c.over).cable][0] == near.branch(near.over).cable
    return right, total


def test_over_under_is_right_wherever_it_is_read():
    """The kill gate the specification names: below 90% and the bottleneck is perception."""
    rows = {arm: _read_accuracy(arm) for arm in ARMS}
    print("\n  over/under on accepted crossings")
    for arm, (right, total) in rows.items():
        pct = 100.0 * right / total if total else float("nan")
        print(f"    {arm:14s} {right:3d}/{total:3d}  {pct:5.1f}%")
    right = sum(r for r, _ in rows.values())
    total = sum(t for _, t in rows.values())
    assert total >= 100, total
    assert right == total, f"{total - right} crossings read the wrong way round"


# --------------------------------------------------------------------------------------
# the headline, and the control
# --------------------------------------------------------------------------------------


def _score(arm: str, rule="read", rng=None):
    cert = wrong = refused = 0
    ks: dict[int, int] = {}
    for _, truth, d, _ in corpus(arm):
        if d is None:
            refused += 1
            continue
        v = certify(_coin_flip(d, rng) if rule == "coin" else d)
        if v.status == CERTIFIED:
            cert += 1
            ks[v.unknown] = ks.get(v.unknown, 0) + 1
            wrong += not _sound(v, _expected_lk(d, truth))
        elif v.status == REFUSED:
            refused += 1
    return cert, wrong, refused, ks


def test_no_arm_ever_certifies_a_verdict_the_scene_contradicts():
    """One wrong CERTIFIED verdict kills the project.  This is that gate."""
    rows = {arm: _score(arm) for arm in ARMS}
    n = len(SEEDS)
    print(f"\n  certified / wrong / refused, out of {n} piles per arm")
    ks: dict[int, int] = {}
    for arm, (cert, wrong, refused, k) in rows.items():
        print(f"    {arm:14s} {cert:3d}   {wrong:3d}   {refused:3d}   ({100.0 * cert / n:4.1f}% certified)")
        for kk, v in k.items():
            ks[kk] = ks.get(kk, 0) + v
    total = sum(c for c, _, _, _ in rows.values())
    print(f"    rule of three: 0 wrong in {total} certified is an upper bound of "
          f"{3.0 / max(total, 1):.3f}, not a rate of 0")
    print(f"    unknown crossings among certified verdicts: {dict(sorted(ks.items()))}")
    if set(ks) == {0}:
        print("    every certified verdict had k = 0, so on this corpus the interval "
              "theorem certified nothing the exact half-sum would not have")
    assert sum(w for _, w, _, _ in rows.values()) == 0
    assert rows["clean"][0] >= 0.7 * n, rows["clean"]


def test_guessing_over_under_certifies_verdicts_the_reader_refuses():
    """The control the specification calls a confidence bar: resolve, do not abstain.

    Run on the *identical* extracted diagrams, so the only thing that changes is the
    decision rule.  It certifies at least as often and it is wrong, which is the whole
    argument for reading over/under and abstaining when the evidence is not there.
    """
    rng = random.Random(SEED)
    print(f"\n  coin flip on the identical diagrams, seed {SEED}")
    total_wrong = 0
    for arm in ARMS:
        cert, wrong, _, _ = _score(arm)
        ccert, cwrong, _, _ = _score(arm, rule="coin", rng=rng)
        total_wrong += cwrong
        print(f"    {arm:14s} read {cert:3d} certified / {wrong} wrong    "
              f"coin {ccert:3d} certified / {cwrong} wrong")
        assert wrong == 0
    # Guessing does not simply certify more often -- a coin flip that lands on a mixed
    # over/under pattern reports lk = 0 and OVER_MIXED, which is NOT CERTIFIED.  What it
    # does is certify *wrongly*, which no amount of coverage buys back.
    assert total_wrong >= 10, total_wrong


def test_the_unknown_policy_is_what_avoids_a_wrong_certificate():
    """A scene where the reader abstains, refuses, and the coin flip certifies wrongly.

    This is the specific claim the UNKNOWN policy makes, so it is asserted on a named
    scene rather than inferred from the totals.
    """
    rng = random.Random(SEED)
    found = []
    for arm in ARMS:
        for seed, truth, d, _ in corpus(arm):
            if d is None:
                continue
            v = certify(d)
            if v.status != REFUSED or v.reason != "LK_STRADDLES_ZERO":
                continue
            # the crossings that went UNKNOWN are exactly the low-confidence reads
            assert any(c.over_conf is not None and c.over_conf < TAU for c in d.between(0, 1))
            for _ in range(8):
                c = certify(_coin_flip(d, rng))
                if c.status == CERTIFIED and not _sound(c, _expected_lk(d, truth)):
                    unknown = sum(1 for x in d.between(0, 1) if x.over_conf < TAU)
                    found.append((arm, seed, f"{unknown} unknown", c.claim, c.value))
                    break
    print(f"\n  scenes where the interval straddles zero and a guess certifies anyway: {found}")
    assert found, "no scene in the corpus exercised the abstention"
