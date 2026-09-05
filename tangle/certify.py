"""The certified layer: the interval, the two witnesses, the refusals, the next crossing.

Pure Python integers.  No numpy, no pixels, no image ever reaches this module -- which is
the point: the claim this layer makes is "given a correct diagram, the verdict is exact",
and nothing here is evidence that the diagram is correct.

There is no closure.  The object is a 2-string tangle with pinned ends and the invariant is
the tangle linking number, which needs no closure at all; a closure built at the image
frame moves with the camera and was the single largest error in the previous design.
`CONVENTION` states the pinning and travels on every Verdict.

The theorems this file implements, in the order it uses them:

  T2  S + k is even iff the endpoint pairs do not interleave.  Interleaving is checked
      first (an interleaved diagram legitimately has an odd count); parity is then a free
      tracer guard that refuses every odd-cardinality inter-component crossing error.
  T4  sign(c) = base(c) * x_c with both factors +/-1, so lk is affine in every unreadable
      factor and the achievable set over all 2^k resolutions is exactly k+1 consecutive
      integers, computable in O(k) with no enumeration.
  T6  r_min = (k - |S|)/2 + 1 crossings must be resolved before any certificate is
      possible.  Necessary, not sufficient.
  T3  separable => lk = 0.  Contrapositive: lk != 0 => the cables cannot be pulled apart
      with their ends held.  That is the LINKED certificate.
  T5  one cable over at every inter-component crossing, ends not interleaved => separable.
      That is the SEPARABLE certificate, pointing the opposite way.

Self-check:  python -m tangle.certify
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .diagram import Crossing, Diagram, Point

CONVENTION = (
    "The scene is a ball. Each cable is an arc properly embedded in it, with its two ends "
    "fixed where the cable leaves the picture. The verdict is invariant under any motion of "
    "the cables inside the scene that keeps them disjoint and keeps all four ends fixed. "
    "Nothing is claimed about what the cables do outside the picture."
)

# ponytail: TAU is a per-rig calibration knob, not a constant.  It lives inside the
# certified layer on purpose -- certify() downgrades any crossing below it to UNKNOWN
# itself rather than trusting the caller, so the honesty boundary sits where a test can
# hold it.
TAU = 0.80

# -- vocabulary ------------------------------------------------------------------------

CERTIFIED = "CERTIFIED"
NOT_CERTIFIED = "NOT CERTIFIED"
REFUSED = "REFUSED"

LINKED = "LINKED"
SEPARABLE = "SEPARABLE"

# not-certified reasons: the computation succeeded and proves nothing
LK_ZERO = "LK_ZERO"
OVER_MIXED = "OVER_MIXED"

# refusal reasons: the computation declined, with a cause and a next action
NOT_TWO_COMPONENTS = "NOT_TWO_COMPONENTS"
DIAGRAM_NOT_CERTIFIED = "DIAGRAM_NOT_CERTIFIED"
FREE_END_IN_FRAME = "FREE_END_IN_FRAME"
INTERLEAVED_ENDS = "INTERLEAVED_ENDS"
ODD_CROSSING_PARITY = "ODD_CROSSING_PARITY"
LK_STRADDLES_ZERO = "LK_STRADDLES_ZERO"
TRIPLE_POINT = "TRIPLE_POINT"
VIEWS_DISAGREE = "VIEWS_DISAGREE"

EXIT = {CERTIFIED: 0, NOT_CERTIFIED: 1, REFUSED: 2}

# Never printed, in any field, by any code path.  Enforced in Verdict.__post_init__, and
# by test_no_verdict_carries_a_banned_phrase and test_the_package_source_carries_no_banned_phrase.
BANNED = ("unlinked", "not linked", "unknotted", "safe to pull", "just pull them")


# --------------------------------------------------------------------------------------
# the interval
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """The achievable lk values: lo..hi inclusive, k+1 consecutive integers (T4)."""

    lo: int
    hi: int
    known_sum: int  # S, the signed sum over the readable inter-component crossings
    unknown: int  # k, the number of unreadable ones

    def __post_init__(self) -> None:
        # ValueError, never assert: assert is stripped under python -O and this identity
        # is the one that catches every hand-typed example block.
        if self.hi - self.lo != self.unknown:
            raise ValueError(f"hi - lo == k is violated: {self.hi} - {self.lo} != {self.unknown}")
        if self.lo + self.hi != self.known_sum:
            raise ValueError(f"lo + hi == S is violated: {self.lo} + {self.hi} != {self.known_sum}")

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    @property
    def exact(self) -> int | None:
        return self.lo if self.lo == self.hi else None

    def __contains__(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def __iter__(self):
        return iter(range(self.lo, self.hi + 1))

    def __str__(self) -> str:
        return f"[{self.lo}, {self.hi}]" if self.lo != self.hi else f"{self.lo}"


def lk_interval(d: Diagram, i: int = 0, j: int = 1) -> Interval:
    """T4, in O(k), with no enumeration.

    S is the signed sum over the readable inter-component crossings; k counts the ones
    whose sign is unknown for any reason -- unreadable over/under, or a tangent too nearly
    parallel for base.  Both are the same unknown +/-1 in the same product.
    """
    S = 0
    k = 0
    for c in d.between(i, j):
        s = d.sign(c)
        if s is None:
            k += 1
        else:
            S += s
    if (S + k) % 2 != 0:
        raise ValueError(f"S + k is odd (S={S}, k={k}); parity_ok must be checked first (T2)")
    return Interval(lo=(S - k) // 2, hi=(S + k) // 2, known_sum=S, unknown=k)


def achievable(d: Diagram, i: int = 0, j: int = 1) -> range:
    iv = lk_interval(d, i, j)
    return range(iv.lo, iv.hi + 1)


def parity_ok(d: Diagram, i: int = 0, j: int = 1) -> bool:
    """T2(b).  Free tracer guard: refuses every odd-cardinality inter-component error.

    S + k must be even when the endpoint pairs do not interleave, and odd when they do.

    T2 is stated for two arcs with ends on the boundary, or for two closed components.  A
    pair with one of each is outside it: a closed curve may separate an arc's two ends, in
    which case an odd count is correct and this returns False.  The failure is a refusal,
    never a certificate, so the direction of the error is the safe one.
    """
    S = sum(s for s in (d.sign(c) for c in d.between(i, j)) if s is not None)
    k = len(d.unknown_between(i, j))
    want = 1 if d.ends_interleave(i, j) else 0
    return (S + k) % 2 == want


def r_min(iv: Interval) -> int:
    """T6.  Crossings that must be resolved before any certificate is possible.

    Necessary, not sufficient: an adversarial resolution of r_min crossings certifies
    nothing.  k - |S| is even by T2, so there is no rounding.
    """
    if iv.excludes_zero:
        return 0
    return (iv.unknown - abs(iv.known_sum)) // 2 + 1


def brute_force_interval(d: Diagram, i: int = 0, j: int = 1) -> Interval:
    """Explicit 2^k enumeration.  Tests only, exponential, and the control for T4.

    Crossings whose `over` is unknown but whose `base` is known are resolved through
    Diagram.resolve and Diagram.sign, so this exercises the real sign path rather than
    re-deriving the interval formula.  A crossing whose `base` is unknown cannot be fixed
    by any over/under assignment, so its +/-1 is enumerated directly.
    """
    unk = d.unknown_between(i, j)
    by_over = [c for c in unk if c.base is not None]
    forced = [c for c in unk if c.base is None]
    known = sum(s for s in (d.sign(c) for c in d.between(i, j)) if s is not None)
    vals: set[int] = set()
    for over_bits in itertools.product("ab", repeat=len(by_over)):
        resolved = d.resolve({c.id: o for c, o in zip(by_over, over_bits)})
        S = sum(s for s in (resolved.sign(c) for c in resolved.between(i, j)) if s is not None)
        for extra in itertools.product((1, -1), repeat=len(forced)):
            total = S + sum(extra)
            if total % 2 != 0:
                raise ValueError("odd crossing sum; parity must be checked first")
            vals.add(total // 2)
    if not vals:
        vals = {known // 2}
    return Interval(lo=min(vals), hi=max(vals), known_sum=known, unknown=len(unk))


def intersect(intervals: Sequence[Interval]) -> Interval | None:
    """T7.  lk is one physical number, so two views' intervals must intersect.

    No crossing correspondence across views is computed, needed, or possible.  Disjoint
    intervals prove at least one trace is wrong; that is VIEWS_DISAGREE, not a merge.
    """
    if not intervals:
        raise ValueError("intersect() needs at least one interval")
    lo = max(iv.lo for iv in intervals)
    hi = min(iv.hi for iv in intervals)
    if lo > hi:
        return None
    return Interval(lo=lo, hi=hi, known_sum=lo + hi, unknown=hi - lo)


# --------------------------------------------------------------------------------------
# the second witness, and active perception
# --------------------------------------------------------------------------------------


def over_everywhere(d: Diagram, i: int = 0, j: int = 1) -> int | None:
    """T5.  The cable that is over at every inter-component crossing, or None.

    Returns a cable id (the spec's 'a'/'b' labels are per-crossing and ambiguous once the
    pair is not (0, 1)).  With no inter-component crossings at all the condition holds
    vacuously and the lower id is returned.  Requires every inter-component crossing to be
    read: a single unreadable over/under is enough to make the witness unavailable.

    This is over/under only.  A crossing whose tangent is too tangential for `base` still
    widens the lk interval, but it cannot affect separability -- T5's proof uses the height
    function and the endpoint order, never a tangent -- so the witness can fire where the
    lk path cannot.  That is strictly stronger than a flat "k = 0" precondition and still
    sound.
    """
    cs = d.between(i, j)
    if any(c.over is None for c in cs):
        return None
    tops = {c.branch(c.over).cable for c in cs}
    if not tops:
        return min(i, j)
    if len(tops) == 1:
        return tops.pop()
    return None


def _minority_over(d: Diagram, i: int, j: int) -> tuple[int, ...]:
    """Crossings where the less-often-over cable is on top.

    These are the obstructions to separability and the useful thing to re-shoot.  A tie is
    not a minority: there is nothing shorter to name, so the honest reason is LK_ZERO.
    """
    cs = [c for c in d.between(i, j) if c.over is not None]
    counts = {i: 0, j: 0}
    for c in cs:
        counts[c.branch(c.over).cable] += 1
    if counts[i] == counts[j]:
        return ()
    loser = i if counts[i] < counts[j] else j
    return tuple(c.id for c in cs if c.branch(c.over).cable == loser)


def next_crossing(d: Diagram, i: int = 0, j: int = 1) -> tuple[int, Point, float] | None:
    """The crossing to look at, its pixel, and the camera bearing.  None if there is none.

    Ranks only crossings that can move this verdict: unknown *inter-component* crossings
    for this pair.  A self-crossing cannot change lk(i, j) by any resolution, and sending
    the photographer 30 degrees left to look at one is a wasted photograph.

    The ranking is |sin theta| ascending -- most tangential first.  That is a perception
    heuristic about which crossing a new view most improves, not a decision-theoretic one:
    every unknown shrinks the interval width by exactly 1, so no crossing is more decisive
    than another and the "which crossing splits the lifts most evenly" schedule is vacuous
    for the linking number.

    The bearing is the in-plane bisector of the two tangents.  Rotating about the bisector
    foreshortens along it and opens the apparent crossing angle toward 90 degrees.
    """
    unk = d.unknown_between(i, j)
    if not unk:
        return None
    c = min(unk, key=lambda x: (x.angle_deg, x.id))
    ta, tb = c.a.t, c.b.t
    if ta[0] * tb[0] + ta[1] * tb[1] < 0:
        tb = (-tb[0], -tb[1])
    bx, by = ta[0] + tb[0], ta[1] + tb[1]
    if bx == 0.0 and by == 0.0:
        bx, by = ta
    bearing = math.degrees(math.atan2(by, bx)) % 180.0
    return (c.id, c.xy, bearing)


# --------------------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    status: str
    claim: str = ""
    witness: str = ""
    value: int | None = None
    interval: Interval | None = None
    unknown: int = 0
    reason: str = ""
    look_at: tuple[int, ...] = ()
    advice: str = ""
    convention: str = CONVENTION
    exit_code: int = 0

    def __post_init__(self) -> None:
        # ValueError, never assert: assert is stripped under python -O and this is the
        # invariant that keeps an unearned claim from ever being printed.
        if self.claim and self.status != CERTIFIED:
            raise ValueError(f"claim {self.claim!r} without a certificate ({self.status})")
        if self.status == CERTIFIED and not self.witness:
            raise ValueError("a certificate needs a named witness")
        blob = " ".join(
            str(v) for v in (self.claim, self.witness, self.reason, self.advice, self.status)
        ).lower()
        for word in BANNED:
            if word in blob:
                raise ValueError(f"banned phrase {word!r} in a verdict")

    def line(self) -> str:
        head = self.claim or self.reason or self.status
        num = "" if self.value is None else f"  lk = {self.value}"
        if self.interval is not None and self.interval.unknown:
            num = f"  lk in {self.interval}  (k = {self.interval.unknown})"
        return f"{self.status}  {head}{num}"

    def block(self) -> str:
        rows = [self.line()]
        if self.advice:
            rows.append(f"  {self.advice}")
        if self.look_at:
            rows.append(f"  look at crossing(s): {', '.join(str(x) for x in self.look_at)}")
        rows.append(f"  convention: {self.convention}")
        return "\n".join(rows)

    def json(self) -> dict:
        return {
            "status": self.status,
            "claim": self.claim,
            "witness": self.witness,
            "value": self.value,
            "interval": None if self.interval is None else [self.interval.lo, self.interval.hi],
            "known_sum": None if self.interval is None else self.interval.known_sum,
            "unknown": self.unknown,
            "reason": self.reason,
            "look_at": list(self.look_at),
            "advice": self.advice,
            "convention": self.convention,
            "exit_code": self.exit_code,
        }


def _refuse(reason: str, advice: str, look_at: Iterable[int] = ()) -> Verdict:
    return Verdict(
        status=REFUSED,
        reason=reason,
        advice=advice,
        look_at=tuple(look_at),
        exit_code=EXIT[REFUSED],
    )


def _apply_tau(d: Diagram, tau: float) -> Diagram:
    """Downgrade any crossing read below tau to UNKNOWN, inside the certified layer."""
    drop = {
        c.id: None
        for c in d.crossings
        if c.over is not None and c.over_conf is not None and c.over_conf < tau
    }
    return d.resolve(drop) if drop else d


def certify(d: Diagram, i: int = 0, j: int = 1, tau: float = TAU) -> Verdict:
    """The verdict for the pair (i, j).  Four outcomes, in the order the spec fixes.

    Defects, then interleaving, then parity -- all before any invariant.  Then the lk
    interval; then, only if lk did not certify, the over-everywhere witness.  Every
    REFUSED is terminal: no second witness runs on a refused diagram.
    """
    if i == j or not (0 <= i < len(d.cables)) or not (0 <= j < len(d.cables)):
        return _refuse(
            NOT_TWO_COMPONENTS,
            "two visually distinct cables are required; this diagram does not have the pair asked for",
        )

    d = _apply_tau(d, tau)

    defect = d.validate()
    if defect is not None:
        return _refuse(defect, _DEFECT_ADVICE[defect])

    if d.ends_interleave(i, j):
        return _refuse(
            INTERLEAVED_ENDS,
            "the four exit points interleave on the frame boundary, so lk is a half-integer. "
            "The obstruction is the framing, not the tangle: step back or re-frame.",
        )

    if not parity_ok(d, i, j):
        return _refuse(
            ODD_CROSSING_PARITY,
            "an inter-component crossing was missed or invented, or an exit side was misread. "
            "The trace is wrong.",
        )

    iv = lk_interval(d, i, j)
    k = iv.unknown

    if iv.excludes_zero:
        bound = iv.lo if iv.lo > 0 else iv.hi
        claim_num = f"lk = {bound}" if k == 0 else f"|lk| >= {abs(bound)}"
        return Verdict(
            status=CERTIFIED,
            claim=LINKED,
            witness="lk",
            value=bound,
            interval=iv,
            unknown=k,
            advice=f"{claim_num} over all 2^{k} resolutions. The cables cannot be separated "
            f"while their ends stay put. Unplug one.",
            exit_code=EXIT[CERTIFIED],
        )

    top = over_everywhere(d, i, j)
    if top is not None:
        return Verdict(
            status=CERTIFIED,
            claim=SEPARABLE,
            witness="over-everywhere",
            value=iv.exact,
            interval=iv,
            unknown=k,
            advice=f"cable {top} is the over-strand at every inter-component crossing, so the "
            f"two are separated by a disk. Just pull.",
            exit_code=EXIT[CERTIFIED],
        )

    if k > 0:
        nxt = next_crossing(d, i, j)
        need = r_min(iv)
        look = (nxt[0],) if nxt else ()
        bearing = f", camera bearing {nxt[2]:.0f} deg" if nxt else ""
        return _refuse(
            LK_STRADDLES_ZERO,
            f"the achievable interval {iv} contains 0. At least {need} crossing(s) must be "
            f"resolved, and only if all {need} go the same way; if they do not, the honest "
            f"outcome is a number that still proves nothing"
            + (f". Start with crossing {nxt[0]} at {nxt[1][0]:.0f},{nxt[1][1]:.0f}{bearing}" if nxt else ""),
            look,
        )

    minority = _minority_over(d, i, j)
    if minority:
        return Verdict(
            status=NOT_CERTIFIED,
            value=iv.exact,
            interval=iv,
            unknown=k,
            reason=OVER_MIXED,
            look_at=minority,
            advice="both cables are the over-strand somewhere, so the separability witness "
            "cannot fire and lk is 0. These crossings are the obstructions; re-shoot them.",
            exit_code=EXIT[NOT_CERTIFIED],
        )
    return Verdict(
        status=NOT_CERTIFIED,
        value=iv.exact,
        interval=iv,
        unknown=k,
        reason=LK_ZERO,
        advice="lk = 0 is not evidence of anything. The Whitehead link has lk = 0 and its two "
        "components cannot be pulled apart.",
        exit_code=EXIT[NOT_CERTIFIED],
    )


_DEFECT_ADVICE = {
    DIAGRAM_NOT_CERTIFIED: "the traced diagram carries a defect; the invariant is not run on it",
    FREE_END_IN_FRAME: "a cable end lies inside the picture, so the ends are not held and the "
    "question has no topological content. Re-frame so both cables run out of the picture.",
    TRIPLE_POINT: "three strands are concurrent; this is not a generic diagram",
}


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    hopf = Diagram.from_braid([1, 1])  # a clasp
    v = certify(hopf)
    assert v.status == CERTIFIED and v.claim == LINKED, v
    assert abs(v.value) == 1, v
    assert v.interval.unknown == 0 and v.exit_code == 0

    # mirror negates lk exactly and leaves the verdict alone
    m = certify(hopf.mirror())
    assert m.claim == LINKED and m.value == -v.value, (m, v)

    # one unreadable crossing: the interval is still exact enough to certify?  No -- with
    # 2 crossings and one unknown the interval is [0, 1], which straddles.
    blurred = hopf.resolve({hopf.crossings[0].id: None})
    b = certify(blurred)
    assert b.status == REFUSED and b.reason == LK_STRADDLES_ZERO, b
    assert b.look_at, b

    # the interval formula agrees with the explicit enumeration
    iv = lk_interval(blurred)
    assert iv == brute_force_interval(blurred), (iv, brute_force_interval(blurred))
    assert iv.hi - iv.lo == iv.unknown == 1

    # a stack: one cable over the other at both crossings, lk = 0, separable
    stack = Diagram.from_polylines(
        [[(0, 3), (10, 3)], [(0, 7), (4, 5), (6, 5), (10, 7)]],
        frame=(0, 0, 10, 10),
    )
    assert len(stack.between(0, 1)) == 0
    s = certify(stack)
    assert s.status == CERTIFIED and s.claim == SEPARABLE, s

    # T7: intervals from two views intersect; disjoint ones refuse
    assert intersect([Interval(0, 2, 2, 2), Interval(1, 3, 4, 2)]) == Interval(1, 2, 3, 1)
    assert intersect([Interval(0, 0, 0, 0), Interval(1, 1, 2, 0)]) is None

    # a refusal is terminal and carries no claim
    plus = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=(0, 0, 10, 10))
    p = certify(plus)
    assert p.status == REFUSED and p.reason == INTERLEAVED_ENDS and p.claim == "", p

    print("certify: ok")


if __name__ == "__main__":
    _demo()
