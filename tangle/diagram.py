"""The data model, its conventions, and its validation.

One modelling decision the whole certificate rests on: a crossing's sign factors into
a planar part read from the two in-plane tangents and an over/under part that the
photograph is sometimes unable to read.

    sign(c) = base(c) * x_c        base(c) = sgn(det[t_a, t_b])        x_c = +1 iff a is over

Both factors are +/-1.  That is the entire reason `lk` is affine in every unreadable
factor, which is what makes the achievable set an interval computable in O(k) with no
enumeration (T4, see certify.py).

Coordinates are image coordinates with y down.  Image handedness negates every crossing
sign relative to the usual y-up mathematical convention.  Only |lk| is ever certified and
the global sign is a stated convention, so this costs nothing; it is written here so that
nobody re-derives it from a failing test.

Branch `a` of a crossing is the branch on the lower-numbered cable; for a self-crossing it
is the branch with the smaller arclength.  `over == 'a'` therefore means "the lower cable
passes over", and the labelling is a function of the diagram, not of detection order.

Self-check:  python -m tangle.diagram
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

Point = tuple[float, float]

# ponytail: SIN_MIN is a per-rig calibration knob, not a constant.  Below it the two
# in-plane tangents are too nearly parallel for sgn(det[t_a, t_b]) to be trusted, so
# `base` becomes None.  An unknown in `base` is the same unknown +/-1 as an unknown in
# `x_c` -- it widens the interval (T4) and never refuses.
SIN_MIN = 0.15

# ponytail: a resource cap, not a mathematical one.  `from_braid` traces one polyline of
# roughly `len(word)` points per strand and `from_polylines` then finds crossings in
# O(total_points^2) segment comparisons; measured at 5,000 total points (500 letters, 10
# strands) that is already ~4s, and at 64,000 (1000 letters, 64 strands) it does not return
# in two minutes.  Every known caller -- the (2, n) torus family, the demo corpus -- stays
# under a few dozen points, so this costs no legitimate use and turns a hang into a refusal.
MAX_BRAID_POINTS = 4000


# --------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Branch:
    """One of the two strands meeting at a crossing."""

    cable: int
    s: float  # arclength along that cable at the crossing
    t: Point  # unit tangent, oriented along the cable

    def reversed_t(self) -> Point:
        return (-self.t[0], -self.t[1])


@dataclass(frozen=True)
class Crossing:
    id: int
    xy: Point
    a: Branch
    b: Branch
    over: str | None  # 'a', 'b', or None == UNKNOWN
    over_conf: float | None
    base: int | None  # +1 / -1 / None == UNKNOWN
    angle_deg: float  # in (0, 90]; the crossing angle used by the contraction rule
    kind: str  # 'read' | 'unknown'

    @property
    def cables(self) -> tuple[int, int]:
        return (self.a.cable, self.b.cable)

    @property
    def is_self(self) -> bool:
        return self.a.cable == self.b.cable

    def branch(self, which: str) -> Branch:
        return self.a if which == "a" else self.b

    @property
    def under(self) -> str | None:
        if self.over is None:
            return None
        return "b" if self.over == "a" else "a"


@dataclass(frozen=True)
class Cable:
    id: int
    points: tuple[Point, ...]
    closed: bool
    ends: tuple[Point, Point] | None  # None iff closed

    @property
    def length(self) -> float:
        pts = self.points + (self.points[0],) if self.closed else self.points
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


@dataclass(frozen=True)
class Diagram:
    cables: tuple[Cable, ...]
    crossings: tuple[Crossing, ...]
    frame: tuple[float, float, float, float]  # x0, y0, x1, y1
    defects: tuple[str, ...] = ()
    exit_order: tuple[int, ...] = ()  # cable ids in cyclic order round the frame boundary
    provenance: str = ""

    # -- crossing sets ------------------------------------------------------------------

    def between(self, i: int, j: int) -> tuple[Crossing, ...]:
        """Every crossing where cables i and j meet (i != j)."""
        if i == j:
            raise ValueError("between() is for two distinct cables; use self_crossings()")
        want = {i, j}
        return tuple(c for c in self.crossings if set(c.cables) == want)

    def unknown_between(self, i: int, j: int) -> tuple[Crossing, ...]:
        """Inter-component crossings whose sign is not determined: over or base unknown."""
        return tuple(c for c in self.between(i, j) if self.sign(c) is None)

    def self_crossings(self, i: int) -> tuple[Crossing, ...]:
        return tuple(c for c in self.crossings if c.cables == (i, i))

    # -- signs --------------------------------------------------------------------------

    def sign(self, c: Crossing) -> int | None:
        """base * (+1 if a is over else -1); None if either factor is unknown."""
        if c.base is None or c.over is None:
            return None
        return c.base if c.over == "a" else -c.base

    # -- endpoints ----------------------------------------------------------------------

    def open_cables(self) -> tuple[int, ...]:
        return tuple(c.id for c in self.cables if not c.closed)

    def ends_interleave(self, i: int, j: int) -> bool:
        """T2: do the two endpoint pairs interleave in the cyclic order on the boundary?

        O(1) from `exit_order`.  Closed cables have no ends, so they never interleave.
        """
        ci = self.cables[i]
        cj = self.cables[j]
        if ci.closed or cj.closed:
            return False
        seq = [k for k in self.exit_order if k in (i, j)]
        if len(seq) != 4:
            # not exactly two exits each; the caller sees this as FREE_END_IN_FRAME
            return False
        return seq[0] != seq[1] and seq[1] != seq[2] and seq[2] != seq[3]

    # -- transforms ---------------------------------------------------------------------

    def mirror(self) -> "Diagram":
        """Swap over and under at every crossing.  Negates every sign, hence every lk."""
        cs = tuple(
            replace(c, over=(None if c.over is None else ("b" if c.over == "a" else "a")))
            for c in self.crossings
        )
        return replace(self, crossings=cs, provenance=self.provenance + "|mirror")

    def resolve(self, assignment: Mapping[int, str | None]) -> "Diagram":
        """Return the diagram with `over` set (or cleared, with None) at the given ids."""
        cs = tuple(
            replace(c, over=assignment[c.id], kind=("unknown" if assignment[c.id] is None else "read"))
            if c.id in assignment
            else c
            for c in self.crossings
        )
        return replace(self, crossings=cs)

    # -- validation ---------------------------------------------------------------------

    def validate(self, boundary_tol: float | None = None) -> str | None:
        """First structural defect, as a reason code, or None.

        Runs before any invariant.  Does *not* check the component count, interleaving or
        parity: the first belongs to the pair being asked about and the other two are
        ordered checks that belong to certify(), where their order matters (T2b).
        """
        if self.defects:
            return "DIAGRAM_NOT_CERTIFIED"
        tol = boundary_tol if boundary_tol is not None else _default_tol(self.frame)
        for cab in self.cables:
            if cab.closed or cab.ends is None:
                continue
            for p in cab.ends:
                if _boundary_distance(p, self.frame) > tol:
                    return "FREE_END_IN_FRAME"
        # a node of degree > 4 after contraction shows up here as coincident crossings
        for m in range(len(self.crossings)):
            for n in range(m + 1, len(self.crossings)):
                if math.dist(self.crossings[m].xy, self.crossings[n].xy) <= tol:
                    return "TRIPLE_POINT"
        return None

    # -- identity -----------------------------------------------------------------------

    def digest(self) -> str:
        """sha1 over integer pixel coordinates and crossing states.

        Determinism is claimed for this string, not for rendered bytes.
        """
        h = hashlib.sha1()
        h.update(b"tangle/diagram/v1")
        for cab in self.cables:
            h.update(f"|C{cab.id}:{int(cab.closed)}:{len(cab.points)}".encode())
            for x, y in cab.points:
                h.update(f",{round(x)},{round(y)}".encode())
        for c in self.crossings:
            h.update(
                f"|X{c.id}:{round(c.xy[0])},{round(c.xy[1])}"
                f":{c.a.cable},{c.b.cable}:{c.over}:{c.base}".encode()
            )
        return h.hexdigest()

    # -- constructors -------------------------------------------------------------------

    @staticmethod
    def from_polylines(
        polylines: Sequence[Sequence[Point]],
        over_table: Mapping[int, str | None] | Sequence[str | None] | Callable[[Crossing], str | None] | None = None,
        frame: tuple[float, float, float, float] | None = None,
        closed: bool | Sequence[bool] | None = None,
        sin_min: float = SIN_MIN,
        provenance: str = "from_polylines",
    ) -> "Diagram":
        """Build a diagram from oriented planar polylines.  No pixels involved.

        `closed` defaults to "the polyline repeats its first point at the end".
        `over_table` is indexed by crossing id, which is assigned by the reproducible
        ordering of stage 17 -- (lowest cable touching the crossing, arclength along that
        cable, x, y) -- so it is a function of the geometry and not of detection order.
        """
        polys = [list(p) for p in polylines]
        if not polys:
            raise ValueError("from_polylines needs at least one polyline, got zero")
        for idx, p in enumerate(polys):
            if not p:
                raise ValueError(f"polyline {idx} is empty; every cable needs at least one point")
        if closed is None:
            flags = [len(p) > 2 and math.dist(p[0], p[-1]) < 1e-12 for p in polys]
        elif isinstance(closed, bool):
            flags = [closed] * len(polys)
        else:
            flags = list(closed)
        polys = [p[:-1] if f and math.dist(p[0], p[-1]) < 1e-12 else p for p, f in zip(polys, flags)]

        cables = tuple(
            Cable(
                id=i,
                points=tuple((float(x), float(y)) for x, y in p),
                closed=f,
                ends=None if f else ((float(p[0][0]), float(p[0][1])), (float(p[-1][0]), float(p[-1][1]))),
            )
            for i, (p, f) in enumerate(zip(polys, flags))
        )
        if frame is None:
            frame = _bbox([pt for cab in cables for pt in cab.points])

        raw = _find_crossings(cables, sin_min)
        raw.sort(key=lambda r: (min(r["a"].cable, r["b"].cable), r["key_s"], r["xy"][0], r["xy"][1]))
        crossings = tuple(
            Crossing(
                id=i,
                xy=r["xy"],
                a=r["a"],
                b=r["b"],
                over=None,
                over_conf=None,
                base=r["base"],
                angle_deg=r["angle_deg"],
                kind="unknown",
            )
            for i, r in enumerate(raw)
        )
        d = Diagram(
            cables=cables,
            crossings=crossings,
            frame=frame,
            exit_order=_exit_order(cables, frame),
            provenance=provenance,
        )
        if over_table is None:
            return d
        if callable(over_table):
            assignment = {c.id: over_table(c) for c in crossings}
        elif isinstance(over_table, Mapping):
            assignment = {c.id: over_table.get(c.id) for c in crossings}
        else:
            seq = list(over_table)
            if len(seq) != len(crossings):
                raise ValueError(f"over_table has {len(seq)} entries for {len(crossings)} crossings")
            assignment = {c.id: seq[c.id] for c in crossings}
        return d.resolve(assignment)

    @staticmethod
    def from_braid(
        word: Sequence[int],
        strands: int | None = None,
        provenance: str | None = None,
    ) -> "Diagram":
        """Closure of a braid word, as closed planar polylines with known over/under.

        `+i` is sigma_i (the strand travelling from position i to i+1 passes over),
        `-i` is its inverse.  This exists so the certified layer has known-answer inputs
        with no camera, no tracer and no dataset: lk of a closure is the signed exponent
        sum over the inter-component letters, and the (2, n) torus links give a closed-form
        determinant ladder (det = n) that is not a fit to anything this package computes.
        """
        word = [int(w) for w in word]
        if any(w == 0 for w in word):
            raise ValueError("braid letters are nonzero")
        n = strands if strands is not None else (max(abs(w) for w in word) + 1 if word else 2)
        if n < 1 or (word and n <= max(abs(w) for w in word)):
            raise ValueError(f"{n} strands cannot carry the word {word}")
        if n * len(word) > MAX_BRAID_POINTS:
            raise ValueError(
                f"{n} strands x {len(word)} letters traces roughly {n * len(word)} points, "
                f"over the {MAX_BRAID_POINTS}-point budget that keeps crossing-finding "
                "(O(points^2) in from_polylines) from hanging; use fewer strands or a "
                "shorter word"
            )
        m = len(word)
        depth = max(m, 1)

        # strand paths through the braid, top position p -> bottom position sigma(p)
        paths: dict[int, list[Point]] = {}
        sigma: dict[int, int] = {}
        for p in range(1, n + 1):
            pos = p
            pts = [(float(pos), 0.0)]
            for r, w in enumerate(word):
                i = abs(w)
                if pos == i:
                    pos = i + 1
                elif pos == i + 1:
                    pos = i
                pts.append((float(pos), float(r + 1)))
            if not word:
                pts.append((float(pos), float(depth)))
            paths[p] = pts
            sigma[p] = pos

        def arc(q: int) -> list[Point]:
            d = 0.4 * (n - q + 1)  # nested: position 1 is the outermost closure arc
            x = n + 0.5 + d
            return [
                (float(q), float(depth) + d),
                (x, float(depth) + d),
                (x, -d),
                (float(q), -d),
            ]

        polylines: list[list[Point]] = []
        seen: set[int] = set()
        for p in range(1, n + 1):
            if p in seen:
                continue
            pts: list[Point] = []
            q = p
            while q not in seen:
                seen.add(q)
                pts.extend(paths[q])
                pts.extend(arc(sigma[q]))
                q = sigma[q]
            polylines.append(pts)

        # over/under is dictated by the word, matched back to the crossing by position
        letters = {(abs(w) + 0.5, r + 0.5): w for r, w in enumerate(word)}

        def over_of(c: Crossing) -> str:
            key = (round(c.xy[0] * 2) / 2, round(c.xy[1] * 2) / 2)
            w = letters.get(key)
            if w is None:
                raise ValueError(f"braid closure produced an unexpected crossing at {c.xy}")
            a_goes_right = c.a.t[0] > 0
            return "a" if a_goes_right == (w > 0) else "b"

        return Diagram.from_polylines(
            polylines,
            over_table=over_of,
            closed=True,
            provenance=provenance or f"from_braid({list(word)}, strands={n})",
        )


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def _bbox(pts: Iterable[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _default_tol(frame: tuple[float, float, float, float]) -> float:
    diag = math.dist((frame[0], frame[1]), (frame[2], frame[3]))
    return 1e-9 + 1e-6 * diag


def _boundary_distance(p: Point, frame: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = frame
    return min(abs(p[0] - x0), abs(x1 - p[0]), abs(p[1] - y0), abs(y1 - p[1]))


def _boundary_param(p: Point, frame: tuple[float, float, float, float]) -> float:
    """Perimeter coordinate of a boundary point, clockwise from the top-left corner."""
    x0, y0, x1, y1 = frame
    w, h = x1 - x0, y1 - y0
    d = [abs(p[1] - y0), abs(x1 - p[0]), abs(p[1] - y1), abs(p[0] - x0)]
    side = d.index(min(d))
    if side == 0:
        return p[0] - x0
    if side == 1:
        return w + (p[1] - y0)
    if side == 2:
        return w + h + (x1 - p[0])
    return 2 * w + h + (y1 - p[1])


def _exit_order(cables: Sequence[Cable], frame: tuple[float, float, float, float]) -> tuple[int, ...]:
    exits = []
    for cab in cables:
        if cab.closed or cab.ends is None:
            continue
        for p in cab.ends:
            exits.append((_boundary_param(p, frame), cab.id))
    exits.sort()
    return tuple(cid for _, cid in exits)


def _seg_dir(p: Point, q: Point) -> Point:
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return (0.0, 0.0)
    return (dx / n, dy / n)


def _cross2(u: Point, v: Point) -> float:
    return u[0] * v[1] - u[1] * v[0]


def _segments(cab: Cable) -> list[tuple[Point, Point]]:
    pts = list(cab.points)
    if cab.closed:
        pts.append(pts[0])
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _cumlen(cab: Cable) -> list[float]:
    segs = _segments(cab)
    out = [0.0]
    for p, q in segs:
        out.append(out[-1] + math.dist(p, q))
    return out


def _find_crossings(cables: Sequence[Cable], sin_min: float) -> list[dict]:
    """Every proper transversal intersection between polyline segments.

    Half-open segment parameters keep a crossing that lands on a shared polyline vertex
    from being counted twice, and adjacent segments of one polyline are skipped outright.
    """
    segs = {cab.id: _segments(cab) for cab in cables}
    cums = {cab.id: _cumlen(cab) for cab in cables}
    out: list[dict] = []
    ids = [cab.id for cab in cables]
    for ii, ci in enumerate(ids):
        for cj in ids[ii:]:
            same = ci == cj
            ns, ms = len(segs[ci]), len(segs[cj])
            closed_i = cables[ci].closed
            for si in range(ns):
                start = si + 2 if same else 0
                for sj in range(start, ms):
                    if same and closed_i and si == 0 and sj == ms - 1:
                        continue  # wrap-adjacent
                    hit = _seg_cross(*segs[ci][si], *segs[cj][sj])
                    if hit is None:
                        continue
                    t, u, xy = hit
                    ti = _seg_dir(*segs[ci][si])
                    tj = _seg_dir(*segs[cj][sj])
                    si_len = math.dist(*segs[ci][si])
                    sj_len = math.dist(*segs[cj][sj])
                    bi = Branch(ci, cums[ci][si] + t * si_len, ti)
                    bj = Branch(cj, cums[cj][sj] + u * sj_len, tj)
                    a, b = (bi, bj) if (ci, bi.s) <= (cj, bj.s) else (bj, bi)
                    cr = _cross2(a.t, b.t)
                    base = None if abs(cr) < sin_min else (1 if cr > 0 else -1)
                    ang = math.degrees(math.asin(min(1.0, abs(cr))))
                    out.append(
                        {
                            "xy": xy,
                            "a": a,
                            "b": b,
                            "base": base,
                            "angle_deg": ang,
                            "key_s": a.s if a.cable <= b.cable else b.s,
                        }
                    )
    return out


def _seg_cross(p0: Point, p1: Point, q0: Point, q1: Point):
    rx, ry = p1[0] - p0[0], p1[1] - p0[1]
    sx, sy = q1[0] - q0[0], q1[1] - q0[1]
    den = rx * sy - ry * sx
    if den == 0.0:
        return None
    dx, dy = q0[0] - p0[0], q0[1] - p0[1]
    t = (dx * sy - dy * sx) / den
    u = (dx * ry - dy * rx) / den
    if not (0.0 <= t < 1.0 and 0.0 <= u < 1.0):
        return None
    return t, u, (p0[0] + t * rx, p0[1] + t * ry)


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    # a clasp: two arcs hooked once, two inter-component crossings of equal sign
    d = Diagram.from_braid([1, 1])  # Hopf link, the closure of sigma_1^2
    assert len(d.cables) == 2, d.cables
    assert len(d.crossings) == 2, d.crossings
    assert all(not c.is_self for c in d.crossings)
    s = [d.sign(c) for c in d.between(0, 1)]
    assert s[0] == s[1] and s[0] is not None, s
    assert abs(sum(s)) == 2, s

    # mirror negates every sign
    m = d.mirror()
    assert [m.sign(c) for c in m.between(0, 1)] == [-x for x in s]

    # an unknown crossing kills exactly one sign and nothing else
    b = d.resolve({0: None})
    assert d.sign(d.crossings[0]) is not None
    assert b.sign(b.crossings[0]) is None
    assert len(b.unknown_between(0, 1)) == 1

    # the (2, n) torus family: n crossings, all inter-component when n is even
    for n in range(2, 7):
        t = Diagram.from_braid([1] * n)
        assert len(t.crossings) == n, (n, len(t.crossings))
        assert len(t.cables) == (2 if n % 2 == 0 else 1)

    # trefoil with a kink: R1 adds a self-crossing and nothing else
    tref = Diagram.from_braid([1, 1, 1], strands=3)
    kink = Diagram.from_braid([1, 1, 1, 2], strands=3)
    assert len(tref.crossings) == 3 and len(kink.crossings) == 4
    assert len(kink.cables) == 1

    # digest is stable and sensitive
    assert d.digest() == Diagram.from_braid([1, 1]).digest()
    assert d.digest() != m.digest()

    # validation: a free end inside the frame refuses
    ok = Diagram.from_polylines([[(0, 2), (10, 2)], [(0, 8), (10, 8)]], frame=(0, 0, 10, 10))
    assert ok.validate() is None, ok.validate()
    bad = Diagram.from_polylines([[(0, 2), (5, 2)], [(0, 8), (10, 8)]], frame=(0, 0, 10, 10))
    assert bad.validate() == "FREE_END_IN_FRAME", bad.validate()

    # interleaving is read off the cyclic order of the four exits
    plus = Diagram.from_polylines([[(0, 5), (10, 5)], [(5, 0), (5, 10)]], frame=(0, 0, 10, 10))
    assert plus.ends_interleave(0, 1) is True, plus.exit_order
    assert ok.ends_interleave(0, 1) is False, ok.exit_order

    print("diagram: ok")


if __name__ == "__main__":
    _demo()
