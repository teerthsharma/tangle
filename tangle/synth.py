"""Synthetic cable piles whose ground truth is analytic, never annotated.

A scene is a list of `Cable3`: a planar polyline plus a height `z` sampled at the same
vertices.  Nothing about over/under is authored.  The crossings are found geometrically by
`Diagram.from_polylines` -- the same code the certified layer uses -- and each one's
over/under falls out of comparing the two heights at the two branch arclengths.  So the
truth is a *consequence* of the scene, not a label attached to it, and the renderer draws
what the truth says rather than the truth recording what the renderer drew.

The drawing is two Pillow polylines plus, at every crossing, a patch of the over strand
redrawn on top.  That is the whole occlusion model and it is exact: the patch runs PATCH_W
widths either side of the crossing and is cut at the cable's own vertices, so it is a
subset of the stroke already there; the generator rejects any scene whose crossings are
closer than MIN_SEP_W > 2 * PATCH_W widths, so two patches can never contend for a pixel.
`test_silhouette_carries_no_depth` asserts the consequence: swapping which cable is on top
leaves the mask bit-identical, and every drop of over/under information is in the colour.

Nuisance axes, all seeded: `blur` (Gaussian, in pixels), `noise` (additive Gaussian on
RGB), `same_colour` (both cables in one colour) and `supersample` (antialiased edges).
The last two are the known worst cases and are here to be failed loudly, not passed.

Two limitations of the corpus, stated here rather than discovered in the results:

* **|lk| <= 1 everywhere.**  An arch weaving across another arch enters and leaves, and
  consecutive crossings then carry opposite planar signs, so no scene can wrap.  Verdicts
  with |lk| >= 2 are covered in closed form by the (2, n) torus family in
  `tests/test_certify.py`, on diagrams that never went near a camera.
* **The easiest possible photometry.**  Matte constant-colour cables on a matte
  background, with no contact shadow, no specularity, no per-arclength colour variation,
  no JPEG and no camera model.  This is the best case for the reader in `vision.py`, and
  every number measured on this corpus carries that sentence.

Self-check:  python -m tangle.synth
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .diagram import Diagram, Point

# --------------------------------------------------------------------------------------
# the palette, and the one calibration knob
# --------------------------------------------------------------------------------------

BG: tuple[int, int, int] = (214, 212, 208)  # matte, far from both cables in Lab
COLOURS: tuple[tuple[int, int, int], ...] = ((198, 62, 48), (46, 92, 196))
WIDTH = 13.0  # drawn cable width in pixels; every tolerance downstream is a multiple of it
N_POINTS = 161  # vertices per cable: 6 px segments at 512 px, sagitta < 0.05 px
SIZE = (512, 512)

# The generator's stated envelope.  A scene outside it is resampled, not rendered, and
# that is a scope condition on the corpus rather than a result about the reader.
MIN_ANGLE_DEG = 25.0  # below this two strips merge into one blob under any threshold
MIN_SEP_W = 4.0  # crossing separation, in cable widths; keeps the redrawn patches apart
PATCH_W = 1.5  # half-length of a redrawn occlusion patch, in widths (covers theta >= 20 deg)
MIN_CROSSINGS = 2
MAX_CROSSINGS = 8
EDGE_CLEAR = 24.0  # px of clearance from the top/left/right frame edges


@dataclass(frozen=True)
class Cable3:
    """A drawn cable: a planar polyline plus the height that decides every occlusion."""

    xy: tuple[Point, ...]
    z: tuple[float, ...]
    colour: tuple[int, int, int] = COLOURS[0]
    width: float = WIDTH

    def __post_init__(self) -> None:
        if len(self.xy) != len(self.z):
            raise ValueError(f"{len(self.xy)} vertices carry {len(self.z)} heights")


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def _cum(xy) -> np.ndarray:
    """Cumulative arclength at each vertex -- the same parameter `Branch.s` carries."""
    p = np.asarray(xy, dtype=float)
    d = np.hypot(*(p[1:] - p[:-1]).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def arch(
    x0: float,
    x1: float,
    height: float,
    base_y: float,
    wobble=(),
    z=(0.0, 0.0, 0.0, 0.0),
    dip: float = 0.0,
    n: int = N_POINTS,
) -> tuple[tuple[Point, ...], tuple[float, ...]]:
    """One cable: an arch from (x0, base_y) to (x1, base_y), with a height function.

    Both feet sit exactly on `base_y`, which is the frame's bottom edge, so every cable
    runs out of the picture and no scene ever trips FREE_END_IN_FRAME.  Every wobble term
    vanishes at t = 0 and t = 1, so wobble never moves a foot.

    `dip` pulls the middle of the arch down (or pushes it up) without moving either foot,
    turning one hump into two.  It is what puts scenes with four and six inter-component
    crossings into the corpus: with a single hump the inner arch can only enter and leave
    the outer one once, so every scene would be a clasp or a stack and no verdict with
    |lk| > 1 would ever be measured.
    """
    t = np.linspace(0.0, 1.0, n)
    x = x0 + (x1 - x0) * t
    y = base_y - height * np.sin(np.pi * t) * (1.0 + dip * np.cos(2 * np.pi * t))
    for k, ax, bx, ay, by in wobble:
        s, c = np.sin(2 * np.pi * k * t), 1.0 - np.cos(2 * np.pi * k * t)
        x = x + ax * s + bx * c
        y = y + ay * s + by * c
    zz = z[0] + z[1] * t + z[2] * np.sin(np.pi * t) + z[3] * np.sin(2 * np.pi * t)
    return tuple(zip(x.tolist(), y.tolist())), tuple(zz.tolist())


def truth(cables, size: tuple[int, int] = SIZE) -> Diagram:
    """The ground-truth diagram: crossings found geometrically, over/under read from z."""
    w, h = size
    frame = (0.0, 0.0, float(w - 1), float(h - 1))
    cums = [_cum(c.xy) for c in cables]
    zs = [np.asarray(c.z, dtype=float) for c in cables]

    def over(c) -> str:
        za = float(np.interp(c.a.s, cums[c.a.cable], zs[c.a.cable]))
        zb = float(np.interp(c.b.s, cums[c.b.cable], zs[c.b.cable]))
        return "a" if za > zb else "b"

    return Diagram.from_polylines(
        [c.xy for c in cables],
        over_table=over,
        frame=frame,
        closed=False,
        provenance="synth",
    )


# --------------------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------------------


def stack(size: tuple[int, int] = SIZE) -> tuple[Cable3, ...]:
    """Cable 0 lies over cable 1 at every crossing.  Over-everywhere, so SEPARABLE."""
    w, h = size
    a, _ = arch(0.08 * w, 0.92 * w, 0.42 * h, h - 1)
    b, _ = arch(0.30 * w, 0.70 * w, 0.78 * h, h - 1)
    return (
        Cable3(a, (1.0,) * len(a), COLOURS[0]),
        Cable3(b, (-1.0,) * len(b), COLOURS[1]),
    )


def clasp(sign: int = 1, size: tuple[int, int] = SIZE) -> tuple[Cable3, ...]:
    """Cable 1 threads through cable 0: under at one crossing, over at the other.

    Two inter-component crossings of equal sign, so |lk| = 1 and the pair cannot be
    separated with the four feet held.  `sign` mirrors the scene and negates lk.
    """
    w, h = size
    a, _ = arch(0.08 * w, 0.92 * w, 0.42 * h, h - 1)
    b, zb = arch(0.30 * w, 0.70 * w, 0.78 * h, h - 1, z=(-1.0 * sign, 2.0 * sign, 0.0, 0.0))
    return (
        Cable3(a, (0.0,) * len(a), COLOURS[0]),
        Cable3(b, zb, COLOURS[1]),
    )


@lru_cache(maxsize=None)
def pile(
    seed: int,
    size: tuple[int, int] = SIZE,
    min_crossings: int | None = None,
    tries: int = 3000,
) -> tuple[Cable3, ...]:
    """A random two-cable pile inside the generator's stated envelope.

    Cable 0 is a wide arch; cable 1 is a narrower arch whose feet nest inside cable 0's, so
    the four exit points never interleave.  Cable 1 carries a two- or three-lobed wobble at
    roughly cable 0's own height, which is what makes it weave in and out instead of simply
    poking through once.

    The corpus is **stratified**, and deliberately: every third seed is required to produce
    at least four inter-component crossings.  Without that, the envelope's separation and
    angle rules reject almost every multi-crossing candidate, every scene in the corpus is
    a clasp or a stack, and no verdict with |lk| > 1 is ever measured.

    Rejection sampling, deterministic in `seed`.  Rejected: self-crossings (the reader
    refuses those by design), crossings closer than MIN_SEP_W widths, crossing angles below
    MIN_ANGLE_DEG, a cable leaving the frame, and a crossing count outside
    [min_crossings, MAX_CROSSINGS].  Everything the envelope excludes is a scope condition
    on the corpus, not a result about the reader.
    """
    if min_crossings is None:
        min_crossings = 4 if seed % 3 == 2 else MIN_CROSSINGS
    rng = np.random.default_rng(seed)
    w, h = size
    for _ in range(tries):
        a_xy, _ = arch(0.05 * w, 0.95 * w, rng.uniform(0.36, 0.50) * h, h - 1)
        amp = rng.uniform(30.0, 70.0)
        wob = ((int(rng.integers(2, 4)), 0.0, 0.0, rng.uniform(-amp, amp), rng.uniform(-amp, amp)),)
        x0 = rng.uniform(0.14, 0.22) * w
        cables = (
            Cable3(a_xy, (0.0,) * len(a_xy), COLOURS[0]),
            Cable3(
                *arch(
                    x0,
                    w - x0,
                    rng.uniform(0.45, 0.70) * h,
                    h - 1,
                    wob,
                    z=tuple(rng.uniform(-1.0, 1.0, 4).tolist()),
                    dip=rng.uniform(-0.10, 0.35),
                ),
                COLOURS[1],
            ),
        )
        if _in_envelope(cables, size, min_crossings):
            return cables
    raise RuntimeError(f"no in-envelope scene after {tries} tries at seed {seed}")


def _in_envelope(cables, size: tuple[int, int], min_crossings: int = MIN_CROSSINGS) -> bool:
    w, h = size
    for c in cables:
        xs = [p[0] for p in c.xy]
        ys = [p[1] for p in c.xy]
        if min(xs) < EDGE_CLEAR or max(xs) > w - 1 - EDGE_CLEAR:
            return False
        if min(ys) < EDGE_CLEAR or max(ys) > h - 1:
            return False
    d = truth(cables, size)
    if any(c.is_self for c in d.crossings):
        return False
    inter = d.between(0, 1)
    if not (min_crossings <= len(inter) <= MAX_CROSSINGS):
        return False
    if any(c.angle_deg < MIN_ANGLE_DEG for c in inter):
        return False
    sep = MIN_SEP_W * max(c.width for c in cables)
    for m in range(len(inter)):
        for n in range(m + 1, len(inter)):
            if math.dist(inter[m].xy, inter[n].xy) < sep:
                return False
    return True


# --------------------------------------------------------------------------------------
# the renderer
# --------------------------------------------------------------------------------------


def _patch(cable: Cable3, s: float, half: float) -> list[Point]:
    """The sub-polyline of `cable` covering `half` either side of arclength `s`.

    The patch is cut at the cable's own vertices, never between them, and reaches one
    vertex past the requested span.  That makes the redrawn patch a *subset* of the stroke
    already there -- the same segments at the same width -- so redrawing it changes which
    colour a pixel carries and never which pixels are covered.  Cutting mid-segment left a
    butt cap that rasterised two pixels wide of the original, and
    `test_silhouette_carries_no_depth` is the assertion that caught it.
    """
    cum = _cum(cable.xy)
    inside = np.flatnonzero((cum >= s - half) & (cum <= s + half))
    lo = max(int(inside[0]) - 1, 0) if inside.size else 0
    hi = min(int(inside[-1]) + 1, len(cum) - 1) if inside.size else len(cum) - 1
    return list(cable.xy[lo : hi + 1])


def render(
    cables,
    size: tuple[int, int] = SIZE,
    *,
    seed: int = 0,
    blur: float = 0.0,
    noise: float = 0.0,
    same_colour: bool = False,
    supersample: int = 1,
) -> tuple[np.ndarray, Diagram]:
    """Draw the scene and return (uint8 HxWx3 image, ground-truth Diagram).

    Occlusion is applied crossing by crossing: every cable is drawn once, then the over
    strand of each crossing is redrawn over a patch PATCH_W widths long on each side.  Two
    patches can never contend: the generator rejects crossings closer than MIN_SEP_W
    widths and MIN_SEP_W > 2 * PATCH_W.

    `supersample > 1` draws at that multiple of the output size and resamples down, so
    cable edges are antialiased.  It defaults to 1, and that default is a measurement
    rather than laziness: an antialiased edge spreads a few thousand pixels evenly across
    the whole cable-ness range, which leaves the widest-bin-gap threshold of stage 4 with
    no gap to find.  The corpus every number in the tests is measured on therefore has
    hard edges, and the antialiased arm is reported as a refusal rate, not hidden.
    """
    d = truth(cables, size)
    cols = [cables[0].colour] * len(cables) if same_colour else [c.colour for c in cables]
    k = max(int(supersample), 1)
    im = Image.new("RGB", (size[0] * k, size[1] * k), BG)
    dr = ImageDraw.Draw(im)
    for c, col in zip(cables, cols):
        dr.line([(k * x, k * y) for x, y in c.xy], fill=col, width=int(round(k * c.width)), joint="curve")
    for x in d.crossings:
        if x.over is None:
            continue
        br = x.branch(x.over)
        cab = cables[br.cable]
        dr.line(
            [(k * px, k * py) for px, py in _patch(cab, br.s, PATCH_W * cab.width)],
            fill=cols[br.cable],
            width=int(round(k * cab.width)),
            joint="curve",
        )
    if k > 1:
        im = im.resize(size, Image.LANCZOS)
    if blur > 0:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(im, dtype=np.float32)
    if noise > 0:
        arr = arr + np.random.default_rng(seed).normal(0.0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8), d


def save(path: str, cables, **kw) -> Diagram:
    """Render to a PNG.  Used by the demos; the tests never touch the filesystem."""
    img, d = render(cables, **kw)
    Image.fromarray(img).save(path)
    return d


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    from .certify import CERTIFIED, LINKED, SEPARABLE, certify, lk_interval

    # the stack: one cable over the other everywhere, lk = 0, and separable by witness
    d = truth(stack())
    assert len(d.between(0, 1)) == 2, d.between(0, 1)
    assert lk_interval(d).exact == 0
    v = certify(d)
    assert v.status == CERTIFIED and v.claim == SEPARABLE, v.line()

    # the clasp: the same geometry, one height function changed, and it is linked
    for sign in (1, -1):
        d = truth(clasp(sign))
        assert len(d.between(0, 1)) == 2, len(d.between(0, 1))
        assert lk_interval(d).exact == sign * lk_interval(truth(clasp(1))).exact
        v = certify(d)
        assert v.status == CERTIFIED and v.claim == LINKED and abs(v.value) == 1, v.line()

    # the render is deterministic, and the drawn crossing core carries the over colour
    img, d = render(clasp(), noise=2.0, seed=7)
    again, _ = render(clasp(), noise=2.0, seed=7)
    assert np.array_equal(img, again)
    for x in d.crossings:
        col = COLOURS[x.branch(x.over).cable]
        px = img[int(round(x.xy[1])), int(round(x.xy[0]))].astype(int)
        assert max(abs(px - np.array(col))) < 40, (x.id, px, col)

    # random piles land inside the envelope, and every one has a known lk
    for seed in range(6):
        d = truth(pile(seed))
        assert 2 <= len(d.between(0, 1)) <= MAX_CROSSINGS
        assert lk_interval(d).unknown == 0
    print("ok")


if __name__ == "__main__":
    _demo()
