"""The Alexander determinant at t = -1, from the Goeritz matrix.

    det(L) = |Delta_L(-1)| = |det G|

`G` needs no orientation and no crossing signs -- only the checkerboard colouring of the
shadow and each crossing's type eta(c) in {+1, -1}.  That is why it is handled completely
differently from `lk`:

    lk   sign(c) = base(c) * x_c is *affine* in every unknown, so the achievable set over
         all 2^k resolutions is an interval computable in O(k) with no enumeration (T4).

    det  eta(c) *flips* under a crossing change and the matrix entries are affine in eta,
         so det is a degree-n polynomial in the unknowns.  It is multi-affine, not affine:
         no interval bound of T4's kind exists and the 2^k lifts must be enumerated.
         K_MAX is a wall-clock budget, not a mathematical limit.

The two counts are different numbers and share no name in this package.  `k_lk` counts
unreadable *inter-component* crossings for one pair; `k_det` counts *all* unreadable
crossings, self-crossings included, because eta is defined at every crossing.

What is never claimed, in any code path or string:

  * a knot name.  det is multiplicative under connected sum and there are infinitely many
    determinant-1 knots, so every value has infinitely many preimages:
    det(3_1) = det(8_19) = 3, det(4_1) = det(5_1) = 5, det(granny) = det(reef) = 9.
    There is no name table in this package.
  * that det = 1 means anything.  The Kinoshita-Terasaka knot has det 1.
  * chirality.  The determinant cannot see it.

Scope: closed diagrams only.  An open cable has no determinant without a closure, and the
closure this package deliberately does not have is exactly the thing the `lk` path deleted
because it moves with the camera.  Open cables are refused, not silently closed.

Three checks run for free and pin the eta convention independently of any example:

  1. Euler:  V - E + F = 2 on the traced faces.
  2. Parity: Delta(-1) = Delta(1) mod 2, and Delta(1) = +/-1 for a knot, 0 for a link with
     two or more components.  So det is odd for one component and even for two or more.
     This catches a wrong row/column deletion, a mis-coloured checkerboard and an
     off-by-one in the reduction -- none of which the closed-form values can distinguish,
     because those change |det| but not the component count.
  3. Both checkerboard colourings give the same |det|.

Self-check:  python -m tangle.alexander
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .diagram import Crossing, Diagram

# ponytail: K_MAX is a wall-clock budget, not a mathematical limit.  Measured with
# `python -m pytest tests/test_alexander.py -k per_lift -s`: 14 us per lift on the
# 6-crossing (2,6) torus link and 38 us on the 10-crossing (2,10), with the shadow hoisted
# out of the loop.  2^16 lifts is therefore ~2.5 s at ten crossings, which is the budget.
# Raise it only with a measurement.
K_MAX = 16

OPEN_COMPONENT = "OPEN_COMPONENT"
CHECKERBOARD_FAILED = "CHECKERBOARD_FAILED"
K_EXCEEDS_BOUND = "K_EXCEEDS_BOUND"


class DeterminantRefused(Exception):
    """The determinant declined, with a cause.  Refusal is an output, not an error path."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------------------
# the shadow: darts, faces, checkerboard.  Independent of over/under, so it is built once
# and reused across every lift.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shadow:
    """The 4-valent plane graph underlying a closed diagram, with its checkerboard.

    A dart is (crossing id, slot), slot in 0..3 indexing the four directions at that
    crossing sorted by angle.  Corner `slot` is the angular sector from direction `slot` to
    direction `slot + 1`, so the corners of a crossing are indexed by the same 0..3.
    """

    slot_of: dict[tuple[int, str, str], int]  # (crossing, 'a'|'b', 'in'|'out') -> slot
    alpha: dict[tuple[int, int], tuple[int, int]]  # dart -> the dart at the other end
    corner_face: dict[tuple[int, int], int]  # corner -> face index
    colour: tuple[int, ...]  # per face, 0 or 1
    n_faces: int
    n_crossings: int
    n_components: int


def build_shadow(d: Diagram) -> Shadow:
    for cab in d.cables:
        if not cab.closed:
            raise DeterminantRefused(
                OPEN_COMPONENT,
                f"cable {cab.id} is an open arc; the determinant needs a closed diagram and "
                "this package states no closure",
            )
    if not d.crossings:
        raise ValueError("build_shadow needs at least one crossing")

    # slots: the four directions at each crossing, sorted by angle
    slot_of: dict[tuple[int, str, str], int] = {}
    for c in d.crossings:
        dirs = [
            (math.atan2(c.a.t[1], c.a.t[0]), ("a", "out")),
            (math.atan2(-c.a.t[1], -c.a.t[0]), ("a", "in")),
            (math.atan2(c.b.t[1], c.b.t[0]), ("b", "out")),
            (math.atan2(-c.b.t[1], -c.b.t[0]), ("b", "in")),
        ]
        dirs.sort()
        for slot, (_, (branch, io)) in enumerate(dirs):
            slot_of[(c.id, branch, io)] = slot

    # edges: consecutive visits along each cable, cyclically
    alpha: dict[tuple[int, int], tuple[int, int]] = {}
    for cab in d.cables:
        visits = sorted(
            [(c.a.s, c.id, "a") for c in d.crossings if c.a.cable == cab.id]
            + [(c.b.s, c.id, "b") for c in d.crossings if c.b.cable == cab.id]
        )
        if not visits:
            continue
        for idx in range(len(visits)):
            _, cid, br = visits[idx]
            _, ncid, nbr = visits[(idx + 1) % len(visits)]
            u = (cid, slot_of[(cid, br, "out")])
            v = (ncid, slot_of[(ncid, nbr, "in")])
            alpha[u] = v
            alpha[v] = u

    darts = [(c.id, s) for c in d.crossings for s in range(4)]
    if len(alpha) != len(darts):
        raise DeterminantRefused(
            CHECKERBOARD_FAILED, f"{len(alpha)} darts paired out of {len(darts)}"
        )

    # face tracing.  Walking along a dart and turning as far clockwise as possible at the
    # far end keeps one face on the same side throughout; the corner swept at each step is
    # the corner indexed by the dart we leave on.
    corner_face: dict[tuple[int, int], int] = {}
    faces: list[list[tuple[int, int]]] = []
    for start in darts:
        if start in corner_face:
            continue
        idx = len(faces)
        orbit: list[tuple[int, int]] = []
        cur = start
        while cur not in corner_face:
            corner_face[cur] = idx
            orbit.append(cur)
            w, j = alpha[cur]
            cur = (w, (j - 1) % 4)
        faces.append(orbit)

    V, E, F = len(d.crossings), 2 * len(d.crossings), len(faces)
    if V - E + F != 2:
        raise DeterminantRefused(
            CHECKERBOARD_FAILED, f"Euler: V - E + F = {V} - {E} + {F} = {V - E + F}, not 2"
        )

    # checkerboard: faces across an edge differ, and consecutive corners at a crossing differ
    differ: dict[int, set[int]] = {i: set() for i in range(F)}
    for u, v in alpha.items():
        differ[corner_face[u]].add(corner_face[v])
        differ[corner_face[v]].add(corner_face[u])
    for c in d.crossings:
        for s in range(4):
            x, y = corner_face[(c.id, s)], corner_face[(c.id, (s + 1) % 4)]
            differ[x].add(y)
            differ[y].add(x)

    colour = [-1] * F
    for root in range(F):
        if colour[root] != -1:
            continue
        colour[root] = 0
        stack = [root]
        while stack:
            u = stack.pop()
            for v in differ[u]:
                if colour[v] == -1:
                    colour[v] = 1 - colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    raise DeterminantRefused(
                        CHECKERBOARD_FAILED, f"faces {u} and {v} must differ but cannot"
                    )

    return Shadow(
        slot_of=slot_of,
        alpha=alpha,
        corner_face=corner_face,
        colour=tuple(colour),
        n_faces=F,
        n_crossings=len(d.crossings),
        n_components=len(d.cables),
    )


# --------------------------------------------------------------------------------------
# connectivity: a disconnected closed diagram is split, so det = 0 and Goeritz is never
# called (spec section 3, re-entry condition ii)
# --------------------------------------------------------------------------------------


def is_connected(d: Diagram) -> bool:
    """Is the diagram connected as a subset of the plane?

    Each component's image is a connected curve, so the diagram is connected exactly when
    the graph on components (adjacent iff they share a crossing) is connected.
    """
    n = len(d.cables)
    if n <= 1:
        return True
    adj: dict[int, set[int]] = {c.id: set() for c in d.cables}
    for c in d.crossings:
        i, j = c.cables
        adj[i].add(j)
        adj[j].add(i)
    seen = {d.cables[0].id}
    stack = [d.cables[0].id]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


# --------------------------------------------------------------------------------------
# the Goeritz matrix
# --------------------------------------------------------------------------------------


def _eta_and_faces(c: Crossing, sh: Shadow, white: int) -> tuple[int, int, int]:
    """(eta, white face, white face) for one crossing.

    Rotating the under-strand counterclockwise onto the over-strand sweeps exactly the
    corner indexed by the under-strand's outgoing dart, together with its opposite.
    eta = +1 when the swept pair is the white pair.  Flipping the crossing moves that
    corner by one slot, which is the other colour class -- so eta flips, which is the whole
    reason the determinant is not affine in the unknowns.
    """
    under = c.under
    if under is None:
        raise ValueError(f"crossing {c.id} is unresolved; eta is undefined")
    m = sh.slot_of[(c.id, under, "out")]
    f0 = sh.corner_face[(c.id, m)]
    if sh.colour[f0] == white:
        return 1, f0, sh.corner_face[(c.id, (m + 2) % 4)]
    return (
        -1,
        sh.corner_face[(c.id, (m + 1) % 4)],
        sh.corner_face[(c.id, (m + 3) % 4)],
    )


def goeritz(d: Diagram, sh: Shadow, white: int = 0) -> list[list[int]]:
    """The Goeritz matrix of the white faces, with one row and column deleted."""
    faces = [f for f in range(sh.n_faces) if sh.colour[f] == white]
    index = {f: i for i, f in enumerate(faces)}
    p = len(faces)
    g = [[0] * p for _ in range(p)]
    for c in d.crossings:
        eta, fx, fy = _eta_and_faces(c, sh, white)
        x, y = index[fx], index[fy]
        if x != y:
            g[x][y] -= eta
            g[y][x] -= eta
    for i in range(p):
        g[i][i] = -sum(g[i][j] for j in range(p) if j != i)
    return [row[1:] for row in g[1:]]


def _int_det(m: list[list[int]]) -> int:
    """Exact integer determinant, Bareiss.  Python ints, so no overflow and no float."""
    n = len(m)
    if n == 0:
        return 1  # the empty determinant; a one-white-face diagram has det 1
    a = [row[:] for row in m]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            for r in range(k + 1, n):
                if a[r][k] != 0:
                    a[k], a[r] = a[r], a[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
        prev = a[k][k]
    return sign * a[n - 1][n - 1]


def det_parity_ok(value: int, n_components: int) -> bool:
    """Delta(-1) = Delta(1) mod 2; Delta(1) = +/-1 for a knot and 0 for a link (Torres)."""
    return value % 2 == (1 if n_components == 1 else 0)


def determinant(d: Diagram, shadow: Shadow | None = None, verify: bool = True) -> int:
    """|Delta_L(-1)| for a fully-resolved closed diagram.  Exact integer.

    `shadow` is the hoisted face structure; pass it when enumerating lifts, since only eta
    changes across them.
    """
    unresolved = [c.id for c in d.crossings if c.over is None]
    if unresolved:
        raise ValueError(f"crossings {unresolved} are unresolved; use det_values()")
    if not is_connected(d):
        return 0  # a disconnected closed diagram is split, so Delta = 0.  Goeritz not called.
    if not d.crossings:
        return 1  # a single closed component with no crossings
    sh = shadow if shadow is not None else build_shadow(d)
    value = abs(_int_det(goeritz(d, sh, white=0)))
    if verify:
        other = abs(_int_det(goeritz(d, sh, white=1)))
        if other != value:
            raise DeterminantRefused(
                CHECKERBOARD_FAILED,
                f"the two checkerboard colourings disagree: {value} vs {other}",
            )
    if not det_parity_ok(value, sh.n_components):
        raise ValueError(
            f"determinant parity violated: det = {value} on {sh.n_components} component(s); "
            "det is odd for a knot and even for a link (Torres)"
        )
    return value


def det_values(d: Diagram, k_max: int = K_MAX) -> tuple[int, ...]:
    """Every |Delta(-1)| achievable over the 2^k lifts of the unreadable crossings.

    k counts *all* unreadable crossings, self-crossings included, because eta is defined
    at every crossing.  A singleton return is a certificate that the determinant is the
    same on every diagram consistent with the trace; anything longer is not.
    """
    unknown = [c.id for c in d.crossings if c.over is None]
    k = len(unknown)
    if k > k_max:
        raise DeterminantRefused(
            K_EXCEEDS_BOUND,
            f"{k} unreadable crossings exceeds the wall-clock budget K_MAX = {k_max}; "
            "the determinant is not affine in the unknowns, so there is no interval to fall "
            "back on",
        )
    if not is_connected(d):
        return (0,)
    if not d.crossings:
        return (1,)
    sh = build_shadow(d)
    out: set[int] = set()
    for bits in itertools.product("ab", repeat=k):
        lift = d.resolve(dict(zip(unknown, bits)))
        out.add(determinant(lift, shadow=sh, verify=False))
    return tuple(sorted(out))


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    # The (2, n) torus family has det = n in closed form.  It is not a fit to anything this
    # package computes, and it exercises knots (n odd) and links (n even) together with the
    # parity theorem.
    for n in range(2, 9):
        d = Diagram.from_braid([1] * n)
        got = determinant(d)
        assert got == n, (n, got)

    # figure-eight: closure of (sigma_1 sigma_2^-1)^2, 4 crossings, one component, det 5
    fig8 = Diagram.from_braid([1, -2, 1, -2], strands=3)
    assert len(fig8.cables) == 1 and len(fig8.crossings) == 4
    assert determinant(fig8) == 5, determinant(fig8)

    # unknot: no crossings, and with an R1 kink
    circle = Diagram.from_braid([], strands=2)
    assert len(circle.cables) == 2 and determinant(circle) == 0  # split: the 2-unlink
    kinked = Diagram.from_braid([1, 1, 1, 2], strands=3)  # trefoil plus a kink
    assert determinant(kinked) == 3, determinant(kinked)

    # mirror leaves the determinant alone (the determinant cannot see chirality)
    tref = Diagram.from_braid([1, 1, 1])
    assert determinant(tref.mirror()) == determinant(tref) == 3

    # one unreadable crossing on the trefoil: flipping it gives the unknot, so the
    # achievable set is {1, 3} and nothing is certified
    blur = tref.resolve({tref.crossings[0].id: None})
    assert det_values(blur) == (1, 3), det_values(blur)
    assert det_values(tref) == (3,)

    # the bound is enforced, not discovered
    try:
        det_values(tref, k_max=-1)
    except DeterminantRefused as e:
        assert e.reason == K_EXCEEDS_BOUND
    else:
        raise AssertionError("K_MAX was not enforced")

    print("alexander: ok")


if __name__ == "__main__":
    _demo()
