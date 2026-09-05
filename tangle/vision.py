"""Image -> Diagram.  Segmentation, skeleton, occlusion bridging, over/under, UNKNOWN.

This is the layer that carries all of the risk.  The certified layer's claim is "given a
correct diagram, the verdict is exact"; nothing in here is evidence for the diagram being
correct, and the refusals below exist so that a diagram this module is not sure about
never reaches `certify()` at all.

**Precondition 1: two visually distinct cables.**  Components are separated by clustering
Lab colour, so a pile of identically-coloured cables yields one cluster and
`NOT_TWO_COMPONENTS`.  That is the most common real scene and it is refused loudly.

**Over/under comes from continuity, and only from continuity.**  For opaque cables the
silhouette of the *union* carries exactly zero depth information -- set union does not
depend on stacking order -- so every geometric cue is dead on arrival.  What does carry it
is that the under strand's own mask is *interrupted* where the over strand crosses it.
This is Huffman's junction labelling with a photometric component separation in front of
it.  Concretely: each cable is traced as a chain of arcs joined across occlusion gaps, and
at a crossing the cable that had to be bridged is the one underneath:

    over  = the cable that was *not* bridged
    conf  = |g_a - g_b| / (g_a + g_b)  *  exp(-(log(g / (K * w / sin theta)) / S)^2)

The first factor is discrimination -- 1 when exactly one cable was interrupted, 0 when both
were interrupted by the same amount.  The second is consistency: a bridge that is meant to
explain a crossing at angle theta has to be about `w / sin theta` long, and one that is
twice or half that is spanning something else.  A crossing where neither cable was
interrupted has no evidence at all and is UNKNOWN with confidence 0; UNKNOWN is never a
low-confidence over.  Blur closes an occlusion gap before it flips it, so the failure
direction is abstention rather than a wrong read.  `certify()` then downgrades anything
below its own `TAU` to UNKNOWN inside the certified layer, so the honesty boundary is not
this module's to move.

Measured on the synthetic corpus (`tests/test_vision.py`, 20 piles x 4 nuisance arms):
133 of 133 accepted crossings read the right way round, 45 certified verdicts and none
that the scene contradicts, against 32 wrong certificates from the same diagrams with the
reader replaced by a coin flip.

Refusals raised here, before any Diagram exists, and each one terminal:

    NO_INTENSITY_GAP    the histogram of cable-ness has no gap wide against ring noise
    NOT_TWO_COMPONENTS  one colour cluster, or two closer than a Lab JND
    BRANCHED_SKELETON   a cable's own skeleton still branches after spur pruning
    OPEN_TRACE          the arcs of one cable do not chain into a single boundary-to-
                        boundary curve

Self-check:  python -m tangle.vision
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import ClusterError, kmeans2
from skimage.color import rgb2lab
from skimage.draw import line as raster_line
from skimage.measure import approximate_polygon, label
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects, skeletonize

from .certify import NOT_TWO_COMPONENTS
from .diagram import Diagram, Point, _boundary_distance

# --------------------------------------------------------------------------------------
# refusals owned by this layer
# --------------------------------------------------------------------------------------

NO_INTENSITY_GAP = "NO_INTENSITY_GAP"
BRANCHED_SKELETON = "BRANCHED_SKELETON"
OPEN_TRACE = "OPEN_TRACE"


class TraceRefused(Exception):
    """The trace declined.  `.reason` is one of the codes above."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------------------
# calibration knobs.  Every one is a multiple of w_est except the two colour thresholds.
# --------------------------------------------------------------------------------------

RING = 12  # px of border ring used for the background estimate
GAP_SIGMAS = 5.0  # the intensity gap must be this many ring sigmas wide
JND_SEP = 12.0  # Lab distance between the two cluster centres, below which they are one
CLUST_MARGIN = 0.25  # a pixel this far inside the mixed band is evidence for neither cable
MIN_SHARE = 0.15  # each cable must own this share of the cable pixels, or there is one cable
SPUR_W = 2.5  # spurs shorter than this many widths are pruned
G_OCC_W = 8.0  # the widest occlusion gap that may be bridged, in widths
FACE_DEG = 35.0  # how far from antiparallel two bridged tangents may be
BOUND_W = 2.0  # an endpoint this close to the frame is an exit, not an occlusion
NOISE_W = 0.5  # a "gap" shorter than this many widths is noise, not an occlusion
GAP_R_W = 2.5  # a bridge counts as evidence about a crossing within this many widths
# A bridge that explains a crossing at angle theta has to be about w / sin(theta) long.
# Measured over 72 bridges on 36 seeded piles: the observed length is 1.165 times that,
# with a spread of 0.132 in log units and a worst case of 0.356.  BRIDGE_S is six times
# that spread on purpose -- the factor exists to catch a bridge that is twice or half the
# length its own crossing angle predicts, not to police the scatter raster thinning adds.
BRIDGE_K = 1.165
BRIDGE_S = 0.85
RDP_TOL = 0.75  # polyline simplification tolerance, px


# --------------------------------------------------------------------------------------
# stages 1-6: photograph -> two colour-separated masks
# --------------------------------------------------------------------------------------


def background(rgb: np.ndarray, ring: int = RING) -> tuple[np.ndarray, np.ndarray, float]:
    """The image in Lab, the median Lab of the border ring, and that ring's spread.

    Both are medians, so the four cable feet crossing the ring (a few hundred pixels out
    of tens of thousands) do not move either number.
    """
    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    m = np.zeros(rgb.shape[:2], dtype=bool)
    m[:ring] = m[-ring:] = True
    m[:, :ring] = m[:, -ring:] = True
    px = lab[m]
    bg = np.median(px, axis=0)
    sigma = float(np.median(np.linalg.norm(px - bg, axis=1)))
    return lab, bg, max(sigma, 1e-3)


def threshold(de: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """Widest empty run in a 512-bin histogram of cable-ness; midpoint is the threshold.

    Certified against ring noise, not against sparsity: a clean image with two occupied
    bins has a gap equal to its whole range and passes, which the old "gap vs median
    inter-bin gap" rule refused.
    """
    hist, edges = np.histogram(de, bins=512)
    occ = np.flatnonzero(hist)
    best_w, best_t = 0.0, None
    for a, b in zip(occ[:-1], occ[1:]):
        if b - a > 1:
            w = float(edges[b] - edges[a + 1])
            if w > best_w:
                best_w, best_t = w, 0.5 * float(edges[a + 1] + edges[b])
    if best_t is None or best_w < GAP_SIGMAS * sigma:
        raise TraceRefused(
            NO_INTENSITY_GAP,
            f"widest gap {best_w:.2f} < {GAP_SIGMAS} * sigma_ring {sigma:.2f}; "
            "the photograph does not separate cable from background",
        )
    return de > best_t, best_w


def _clean(mask: np.ndarray) -> np.ndarray:
    n = max(int(0.005 * mask.sum()), 8)
    mask = closing(mask, disk(2))
    mask = remove_small_holes(mask, max_size=n)
    return remove_small_objects(mask, max_size=n)


def components(lab: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Split the mask into two masks by clustering Lab colour.

    Deterministic: kmeans2 with a fixed seed and '++' init, and the two clusters are
    ordered by their centroid so the cable ids are a function of the picture.
    """
    # Fit the two colours on the *core* of the mask.  Antialiased boundary pixels are a
    # continuum between cable and background, and letting them into the fit lets k-means
    # answer "core versus fringe" on a pile of identically coloured cables -- which is the
    # one scene the JND test exists to refuse.
    core = ndimage.binary_erosion(mask, np.ones((5, 5), bool))
    if core.sum() < 64:
        raise TraceRefused(NOT_TWO_COMPONENTS, "almost nothing was segmented as cable")
    try:
        centres, lb_core = kmeans2(
            lab[core].astype(np.float64), 2, minit="++", seed=0, missing="raise"
        )
    except ClusterError as e:  # one cluster came back empty: there is one cable colour
        raise TraceRefused(NOT_TWO_COMPONENTS, str(e)) from e
    sep = float(np.linalg.norm(centres[0] - centres[1]))
    share = float(min(np.bincount(lb_core, minlength=2)) / lb_core.size)
    if share < MIN_SHARE:
        # k-means always returns two centres.  On one cable colour it returns the colour
        # and its outliers, and the separation between those two is meaningless.
        raise TraceRefused(
            NOT_TWO_COMPONENTS,
            f"one colour cluster holds {100 * (1 - share):.1f}% of the cable pixels; "
            "two visually distinct cables are required",
        )
    if sep < JND_SEP:
        raise TraceRefused(
            NOT_TWO_COMPONENTS,
            f"the two colour clusters are {sep:.1f} apart in Lab, inside one JND of each "
            "other; two visually distinct cables are required",
        )
    # A pixel whose colour sits in the band between the two cables is evidence for
    # neither: it is the antialiased boundary where the two jackets meet, and keeping it
    # welds a spur onto whichever cluster wins by a hair.
    px = lab[mask]
    dist = np.linalg.norm(px[:, None, :] - centres[None, :, :], axis=2)
    lb = np.argmin(dist, axis=1)
    margin = np.abs(dist[:, 0] - dist[:, 1]) / sep
    order = sorted(range(2), key=lambda i: tuple(centres[i]))
    out = []
    for want in order:
        m = np.zeros_like(mask)
        m[mask] = (lb == want) & (margin >= CLUST_MARGIN)
        m = _clean(m)
        if not m.any():
            raise TraceRefused(NOT_TWO_COMPONENTS, "one colour cluster survived cleanup")
        out.append(m)
    return out


def width_estimate(mask: np.ndarray) -> float:
    """2 * median distance-to-background on the skeleton of the union mask."""
    dt = ndimage.distance_transform_edt(mask)
    sk = skeletonize(mask)
    return float(2.0 * np.median(dt[sk]))


# --------------------------------------------------------------------------------------
# stages 8-11: mask -> ordered arcs
# --------------------------------------------------------------------------------------


def _degree(skel: np.ndarray) -> np.ndarray:
    """3x3 sum *including the centre*: endpoint = 2, interior = 3, branch >= 4."""
    return ndimage.convolve(skel.astype(np.int16), np.ones((3, 3), np.int16), mode="constant")


def prune(skel: np.ndarray, min_len: float) -> np.ndarray:
    """Delete spurs: short runs that carry a free endpoint *and* touch a branch pixel.

    Both conditions matter.  Without the branch-adjacency test this deletes a legitimately
    short arc between two nearby crossings; without the free-endpoint test it deletes the
    bridge of an H.

    An obliquely truncated strip is where the spurs come from and it is not an artifact of
    noise: where one cable is cut off by another crossing it at angle theta, the medial
    axis of the remaining wedge forks towards the two corners of the cut.  The exact
    medial-axis prong is w/(2 tan theta) long; Lee thinning puts it about 1.65x further
    out, the same divergence the bridge-contraction table records, which is why the
    threshold is a measured 2.5 w and not the predicted 1.6 w.
    """
    while True:
        deg = _degree(skel) * skel
        branch = deg >= 4
        if not branch.any():
            return skel
        near = ndimage.binary_dilation(branch, np.ones((3, 3), bool))
        runs = label(skel & (deg < 4), connectivity=2)
        killed = False
        for rid in range(1, int(runs.max()) + 1):
            m = runs == rid
            if m.sum() >= min_len or not (m & near).any() or not ((deg == 2) & m).any():
                continue
            skel = skel & ~m
            killed = True
        if not killed:
            return skel
        # Re-thin: deleting a prong leaves the fork's own 2-3 pixel cluster behind, and a
        # triangle of mutually adjacent pixels is a degree-3 knot that no deletion of runs
        # can reach.  Thinning removes it, being simple points.
        skel = skeletonize(skel)


def arcs(mask: np.ndarray, w_est: float) -> list[np.ndarray]:
    """Ordered centrelines, one per connected piece of the cable's own mask."""
    skel = prune(skeletonize(mask), SPUR_W * w_est)
    deg = _degree(skel) * skel
    if (deg >= 4).any():
        raise TraceRefused(
            BRANCHED_SKELETON,
            "a cable's skeleton still branches after pruning: a self-crossing, or two "
            "strands merged into one blob",
        )
    out = []
    lab_img = label(skel, connectivity=2)
    for rid in range(1, int(lab_img.max()) + 1):
        m = lab_img == rid
        ends = np.argwhere(m & (deg == 2))
        if len(ends) != 2:
            raise TraceRefused(
                OPEN_TRACE, f"a traced piece has {len(ends)} endpoints, not 2 (a closed loop?)"
            )
        pix = _walk({(int(r), int(c)) for r, c in np.argwhere(m)}, (int(ends[0][0]), int(ends[0][1])))
        xy = np.array([[c, r] for r, c in pix], dtype=float)
        out.append(approximate_polygon(xy, tolerance=RDP_TOL))
    return out


def _walk(pix: set, start: tuple[int, int]) -> list[tuple[int, int]]:
    order, seen, cur = [start], {start}, start
    while True:
        nb = [
            (cur[0] + dr, cur[1] + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (cur[0] + dr, cur[1] + dc) in pix and (cur[0] + dr, cur[1] + dc) not in seen
        ]
        if not nb:
            return order
        nb.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))  # 4-neighbours first
        cur = nb[0]
        order.append(cur)
        seen.add(cur)


# --------------------------------------------------------------------------------------
# stage 12: bridge the occlusion gaps into one boundary-to-boundary polyline
# --------------------------------------------------------------------------------------


def _tangent(poly: np.ndarray, end: int, span: float) -> np.ndarray:
    """Outward unit tangent at one end of an arc, over `span` pixels of arclength."""
    p = poly if end == 0 else poly[::-1]
    d = np.cumsum(np.concatenate([[0.0], np.hypot(*(p[1:] - p[:-1]).T)]))
    i = int(np.searchsorted(d, span))
    i = min(max(i, 1), len(p) - 1)
    v = p[0] - p[i]
    n = float(np.hypot(*v))
    return v / n if n else np.array([0.0, 0.0])


def _runs_hit(a: np.ndarray, b: np.ndarray, other: np.ndarray) -> int:
    """How many separate runs of the other cable's mask the straight bridge passes over."""
    rr, cc = raster_line(int(round(a[1])), int(round(a[0])), int(round(b[1])), int(round(b[0])))
    rr = np.clip(rr, 0, other.shape[0] - 1)
    cc = np.clip(cc, 0, other.shape[1] - 1)
    hit = other[rr, cc].astype(np.int8)
    return int(np.sum(np.diff(np.concatenate([[0], hit])) == 1))


def chain(
    pieces: list[np.ndarray],
    other_mask: np.ndarray,
    w_est: float,
    frame: tuple[float, float, float, float],
) -> tuple[np.ndarray, list[tuple[Point, float]]]:
    """Join a cable's arcs across occlusion gaps.  Returns (polyline, bridges).

    A bridge is admissible only when the gap is short enough to be one cable width seen at
    an angle, the two outward tangents are antiparallel to within FACE_DEG, and the
    straight bridge passes over *exactly one* run of the other cable -- otherwise it
    invents a crossing behind an obstruction, or invents one where two exist.  Then the
    admissible bridges are searched, not taken greedily: what has to come out is one chain
    from one edge of the picture to the other, and anything else is OPEN_TRACE.
    """
    ends = []  # (piece, which end, xy, outward tangent, is_boundary)
    for i, p in enumerate(pieces):
        for e in (0, 1):
            xy = p[0] if e == 0 else p[-1]
            ends.append(
                (i, e, xy, _tangent(p, e, 1.5 * w_est), _boundary_distance(tuple(xy), frame) <= BOUND_W * w_est)
            )

    cos_lim = math.cos(math.radians(FACE_DEG))
    cand = []
    for m in range(len(ends)):
        for n in range(m + 1, len(ends)):
            i, _, pi, ti, bi = ends[m]
            j, _, pj, tj, bj = ends[n]
            if i == j or bi or bj:
                continue
            gap = float(np.hypot(*(pj - pi)))
            if gap > G_OCC_W * w_est or gap == 0.0:
                continue
            # Only the tangents are asked to agree.  A "do the two ends face each other"
            # test was tried and removed: an obliquely cut strand retreats to the two
            # opposite corners of the cut, the two ends end up a width or more off each
            # other's tangent line, and at a 36 degree crossing the chord between them
            # runs nearly perpendicular to both tangents while being an entirely correct
            # bridge.  What is left is strong enough: antiparallel tangents, a bounded
            # gap, exactly one run of the other cable underneath, and a chain that has to
            # close (below).  A bridge that survives all four and is still wrong shows up
            # as a length its crossing angle does not predict, which `read_over` turns
            # into an abstention rather than a certificate.
            if float(ti @ -tj) < cos_lim:
                continue
            if _runs_hit(pi, pj, other_mask) != 1:
                continue
            cand.append((gap, m, n))

    cand.sort()

    # Choose the *chain*, not the bridges.  Greedily taking the shortest admissible bridge
    # first can spend an endpoint on a bridge that leaves the rest unjoinable, and the
    # failure then looks like a cable that does not reach the edge of the picture.  A
    # cable's arcs have to form one path from one exit to the other, so that is what is
    # searched for: depth-first over admissible bridges, shortest first.  Scenes here have
    # at most a handful of arcs, so the search is exact and its cost is nothing.
    out: dict[int, list[tuple[float, int]]] = {}
    for gap, m, n in cand:
        out.setdefault(m, []).append((gap, n))
        out.setdefault(n, []).append((gap, m))

    def walk(entry: int, used: frozenset) -> list[int] | None:
        piece = entry // 2
        exit_ = 2 * piece + (1 - entry % 2)
        used = used | {piece}
        if len(used) == len(pieces):
            return [entry] if ends[exit_][4] else None
        for _, nxt in out.get(exit_, ()):
            if nxt // 2 in used:
                continue
            rest = walk(nxt, used)
            if rest is not None:
                return [entry] + rest
        return None

    chain_ends = None
    for m, e in enumerate(ends):
        if e[4]:  # a boundary end: a place the cable leaves the picture
            chain_ends = walk(m, frozenset())
            if chain_ends is not None:
                break
    if chain_ends is None:
        raise TraceRefused(
            OPEN_TRACE,
            f"{len(pieces)} traced piece(s) do not chain into one curve from one edge of "
            "the picture to the other",
        )

    pts: list[np.ndarray] = []
    bridges: list[tuple[Point, float]] = []
    for idx, entry in enumerate(chain_ends):
        i, e = ends[entry][0], ends[entry][1]
        pts.append(pieces[i] if e == 0 else pieces[i][::-1])
        if idx + 1 < len(chain_ends):
            a = ends[2 * i + (1 - e)][2]
            b = ends[chain_ends[idx + 1]][2]
            bridges.append((((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), float(np.hypot(*(b - a)))))
    return np.vstack(pts), bridges


def to_frame(poly: np.ndarray, frame: tuple[float, float, float, float]) -> np.ndarray:
    """Project the two free ends onto the nearest frame edge, so both feet are pinned.

    Skeletonisation stops about half a cable width short of the picture edge.  The cable
    does leave the picture, so the diagram must say so or `validate()` reports every trace
    as FREE_END_IN_FRAME.
    """
    x0, y0, x1, y1 = frame
    out = poly.copy()
    for idx in (0, -1):
        x, y = out[idx]
        d = [abs(y - y0), abs(x1 - x), abs(y1 - y), abs(x - x0)]
        side = int(np.argmin(d))
        out[idx] = [(x, y0), (x1, y), (x, y1), (x0, y)][side]
    return out


# --------------------------------------------------------------------------------------
# stage 14: over/under from continuity, with a confidence
# --------------------------------------------------------------------------------------


def _gap_near(bridges: list[tuple[Point, float]], xy: Point, radius: float) -> float:
    best = 0.0
    for mid, gap in bridges:
        if math.dist(mid, xy) <= radius and gap > best:
            best = gap
    return best


def read_over(
    d: Diagram, bridges: list[list[tuple[Point, float]]], w_est: float
) -> Diagram:
    """Label every crossing from the bridge evidence.  UNKNOWN where neither strand broke.

    Two independent factors, multiplied:

      margin  |g_a - g_b| / (g_a + g_b), 1 when exactly one strand was bridged and 0 when
              both were bridged by the same amount -- the discriminative part;
      fit     how close the bridge is to the length w / sin(theta) that this crossing's own
              angle predicts -- the consistency part, which is what refuses a bridge that
              spans something other than the crossing it is being read as.

    A crossing where neither strand was bridged has no evidence at all and is UNKNOWN with
    confidence 0; it is never a low-confidence over.
    """
    out = []
    for c in d.crossings:
        ga = _gap_near(bridges[c.a.cable], c.xy, GAP_R_W * w_est)
        gb = _gap_near(bridges[c.b.cable], c.xy, GAP_R_W * w_est)
        if max(ga, gb) < NOISE_W * w_est:
            out.append(replace(c, over=None, over_conf=0.0, kind="unknown"))
            continue
        over = "b" if ga > gb else "a"
        margin = abs(ga - gb) / (ga + gb)
        pred = BRIDGE_K * w_est / math.sin(math.radians(c.angle_deg))
        fit = math.exp(-((math.log(max(ga, gb) / pred) / BRIDGE_S) ** 2))
        out.append(replace(c, over=over, over_conf=float(margin * fit), kind="read"))
    return replace(d, crossings=tuple(out))


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


def trace(rgb: np.ndarray, frame: tuple[float, float, float, float] | None = None) -> Diagram:
    """Photograph -> Diagram, or `TraceRefused`.

    The crossings are not detected here.  Two centrelines are, and the crossings between
    them are found by `Diagram.from_polylines` -- the same geometric code the closed-form
    tests run against -- so crossing numbering, `base` and the crossing angle come from one
    implementation and this module only has to answer over/under.
    """
    h, w = rgb.shape[:2]
    if frame is None:
        frame = (0.0, 0.0, float(w - 1), float(h - 1))

    lab, bg, sigma = background(rgb)
    de = np.linalg.norm(lab - bg, axis=-1).astype(np.float32)
    mask, _ = threshold(de, sigma)
    mask = _clean(mask)
    masks = components(lab, mask)
    w_est = width_estimate(mask)

    pieces = [arcs(m, w_est) for m in masks]
    polys, bridges = [], []
    for i, ps in enumerate(pieces):
        poly, br = chain(ps, masks[1 - i], w_est, frame)
        polys.append(to_frame(poly, frame))
        bridges.append(br)

    d = Diagram.from_polylines(
        [[(float(x), float(y)) for x, y in p] for p in polys],
        frame=frame,
        closed=False,
        provenance=f"vision w_est={w_est:.1f}",
    )
    return read_over(d, bridges, w_est)


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    from . import synth
    from .certify import CERTIFIED, LINKED, SEPARABLE, TAU, certify, lk_interval

    # a clasp: two crossings, both read, and the certified integer matches the scene
    img, t = synth.render(synth.clasp())
    d = trace(img)
    assert len(d.cables) == 2 and all(len(c.points) > 20 for c in d.cables)
    assert len(d.between(0, 1)) == len(t.between(0, 1)) == 2, d.between(0, 1)
    assert all(c.over_conf > TAU for c in d.between(0, 1)), [c.over_conf for c in d.crossings]
    v = certify(d)
    assert v.status == CERTIFIED and v.claim == LINKED, v.line()
    assert abs(v.value) == abs(lk_interval(t).exact) == 1, (v.value, lk_interval(t).exact)

    # a stack: the same geometry, one height function changed, and it lifts apart
    img, _ = synth.render(synth.stack())
    v = certify(trace(img))
    assert v.status == CERTIFIED and v.claim == SEPARABLE, v.line()

    # the known worst case fails loudly rather than quietly
    img, _ = synth.render(synth.clasp(), same_colour=True)
    try:
        trace(img)
    except TraceRefused as e:
        assert e.reason == NOT_TWO_COMPONENTS, e.reason
    else:
        raise AssertionError("a same-coloured pair was traced instead of refused")

    # a crossing with no bridge evidence at all is UNKNOWN, never a low-confidence over
    d0 = trace(synth.render(synth.clasp())[0])
    blind = read_over(d0, [[], []], 12.8)
    assert all(c.over is None and c.over_conf == 0.0 for c in blind.crossings)
    assert lk_interval(blind).unknown == len(blind.between(0, 1))
    print("ok")

if __name__ == "__main__":
    _demo()
