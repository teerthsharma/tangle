"""tangle/viz.py -- the overlay, and the frames the GIF is made of.

Three crossing states, three glyphs, and never a colour-only distinction, because a
colour-only legend is unreadable to a colour-blind reader and unreadable in a greyscale
screenshot:

    READ     a filled square.  The over-strand runs through unbroken; the under-strand is
             drawn with a GAP_PX gap, which is the standard knot-diagram convention and
             the only mark on the picture that asserts a depth ordering.
    UNKNOWN  a hollow dashed amber square with a `?`, and **both strands continuous**.
             The picture does not draw a decision that was not made.
    TARGET   the crossing a verdict named, ringed, with the camera bearing as an arrow.

A crossing named by a verdict is drawn as TARGET whether or not it was read: `LK_STRADDLES_ZERO`
names an unknown, `OVER_MIXED` names crossings that *were* read and are the obstruction to
separability.  Both are "the thing to re-shoot", which is what the ring means.

**No byte-exact golden PNG test exists and none can.**  PNG bytes depend on the zlib build
and on Pillow's rasteriser, so determinism is claimed for `Diagram.digest()` -- which every
frame prints in its footer -- and the overlay is checked structurally: glyph count and
state, banner colour at fixed pixels, footer string.

The font: the spec called for a bundled OFL monospace TTF because Pillow's
`load_default()` used to return a proportional bitmap face, which cannot hold a banner
integer still between GIF frames.  Measured on the Pillow actually installed here (12.3.0),
`ImageFont.load_default(size=48)` returns a real TrueType face (Aileron Regular) whose ten
digit advances are all 28.0 px -- tabular.  The banner integer does not move, no binary
ships in the repo, and `test_viz.py::test_font_digits_are_tabular` fails if the font that
gets picked ever stops being tabular.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from .certify import CERTIFIED, LINKED, NOT_CERTIFIED, REFUSED, SEPARABLE, Verdict, next_crossing
from .diagram import Diagram, Point

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

READ = "read"
UNKNOWN = "unknown"
TARGET = "target"

#: banner ground per status.  Sampled by the structural test at three fixed pixels.
BANNER_RGB = {
    CERTIFIED: (21, 105, 72),
    NOT_CERTIFIED: (146, 106, 20),
    REFUSED: (155, 42, 42),
}

PAPER = (250, 249, 246)
INK = (28, 28, 30)
FOOTER_RGB = (238, 236, 231)
UNKNOWN_RGB = (214, 150, 20)
TARGET_RGB = (198, 46, 46)
CABLE_RGB = (
    (44, 92, 168),
    (198, 78, 40),
    (46, 138, 92),
    (128, 66, 158),
    (176, 140, 30),
)

GAP_PX = 7.0  # the under-strand break, in output pixels
BANNER_H = 88
FOOTER_H = 26
STAGES = ("cables", "crossings", "banner")

# ponytail: a candidate list beats a committed binary.  The last entry always resolves.
_FONT_CANDIDATES = (
    os.path.join(os.path.dirname(__file__), "..", "DejaVuSansMono.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
)


@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.FreeTypeFont:
    """A face at `size`.  Monospace where the system has one, Pillow's default otherwise."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------------------
# what the picture is allowed to say
# --------------------------------------------------------------------------------------


def glyph_states(d: Diagram, verdict: Verdict | None = None) -> dict[int, str]:
    """Crossing id -> READ / UNKNOWN / TARGET.  The renderer and its test share this."""
    named = set(verdict.look_at) if verdict is not None else set()
    return {
        c.id: TARGET if c.id in named else (READ if c.over is not None else UNKNOWN)
        for c in d.crossings
    }


def banner_text(verdict: Verdict | None) -> tuple[str, str, str]:
    """(headline, stamped number, action line).  Every string comes from the Verdict.

    The headline is exactly what the task asks the banner to carry: `LINKED`, `SEPARABLE`,
    `NOT CERTIFIED`, or `REFUSED: look at crossing N`.  No string here is invented; the
    action line is the Verdict's own advice, truncated, so the never-claimed list is
    enforced by `Verdict.__post_init__` and not by the renderer's good manners.
    """
    if verdict is None:
        return ("NO VERDICT", "", "")
    if verdict.status == CERTIFIED:
        head = verdict.claim
    elif verdict.status == REFUSED:
        head = REFUSED
        if verdict.look_at:
            ids = ", ".join(str(x) for x in verdict.look_at)
            head = f"{REFUSED}: look at crossing {ids}"
        elif verdict.reason:
            head = f"{REFUSED}: {verdict.reason}"
    else:
        head = NOT_CERTIFIED

    num = ""
    iv = verdict.interval
    if iv is not None and iv.unknown:
        bound = iv.lo if iv.lo > 0 else (iv.hi if iv.hi < 0 else None)
        num = f"|lk| >= {abs(bound)}" if bound is not None else f"lk in {iv}"
    elif verdict.value is not None:
        num = f"lk = {verdict.value}"

    action = verdict.advice.split(". ")[0].strip().rstrip(".")
    if verdict.status == NOT_CERTIFIED and verdict.reason == "LK_ZERO":
        action = "lk = 0 is not evidence of anything"
    return (head, num, action)


def footer_text(d: Diagram, verdict: Verdict | None = None) -> str:
    """The line every frame carries, so a GIF cannot be faked frame by frame."""
    bits = [f"digest {d.digest()[:12]}", f"{len(d.crossings)} crossings"]
    if verdict is not None and verdict.interval is not None:
        bits.append(f"k = {verdict.interval.unknown}")
    bits.append("ends pinned; no closure")
    return "  |  ".join(bits)


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def _fit(frame: tuple[float, float, float, float], box: tuple[int, int, int, int]):
    """Isotropic diagram -> pixel map into `box` = (x0, y0, w, h)."""
    fx0, fy0, fx1, fy1 = frame
    bx, by, bw, bh = box
    fw, fh = max(fx1 - fx0, 1e-9), max(fy1 - fy0, 1e-9)
    s = min(bw / fw, bh / fh)
    ox = bx + (bw - fw * s) / 2.0
    oy = by + (bh - fh * s) / 2.0

    def to_px(p: Point) -> tuple[float, float]:
        return (ox + (p[0] - fx0) * s, oy + (p[1] - fy0) * s)

    return to_px, s


def _cum(points: Sequence[Point], closed: bool) -> tuple[list[Point], list[float]]:
    pts = list(points) + ([points[0]] if closed and len(points) > 1 else [])
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    return pts, cum


def _at(pts: Sequence[Point], cum: Sequence[float], s: float) -> Point:
    """Point at arclength `s`, by linear interpolation between vertices."""
    if s <= cum[0]:
        return pts[0]
    if s >= cum[-1]:
        return pts[-1]
    for i in range(len(cum) - 1):
        if cum[i] <= s <= cum[i + 1]:
            span = cum[i + 1] - cum[i]
            t = 0.0 if span == 0 else (s - cum[i]) / span
            return (
                pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                pts[i][1] + t * (pts[i + 1][1] - pts[i][1]),
            )
    return pts[-1]


def breaks(d: Diagram, cable: int, gap: float) -> list[tuple[float, float]]:
    """Arclength intervals where `cable` runs *under* a crossing that was read.

    An unknown crossing produces no break: both strands stay continuous, which is the
    whole point of the UNKNOWN glyph.
    """
    cab = d.cables[cable]
    _, cum = _cum(cab.points, cab.closed)
    total = cum[-1]
    out: list[tuple[float, float]] = []
    for c in d.crossings:
        if c.over is None:
            continue
        under = c.branch(c.under)
        if under.cable != cable:
            continue
        lo, hi = under.s - gap, under.s + gap
        out.append((lo, hi))
        if cab.closed and total > 0:  # the break wraps round the seam
            if lo < 0:
                out.append((total + lo, total))
            if hi > total:
                out.append((0.0, hi - total))
    return out


def pieces(
    points: Sequence[Point], closed: bool, cuts: Sequence[tuple[float, float]]
) -> list[list[Point]]:
    """The polyline with `cuts` removed, as a list of sub-polylines."""
    pts, cum = _cum(points, closed)
    if len(pts) < 2:
        return []
    samples = list(zip(cum, pts))
    for lo, hi in cuts:
        for s in (lo, hi):
            if cum[0] < s < cum[-1]:
                samples.append((s, _at(pts, cum, s)))
    samples.sort(key=lambda t: t[0])

    out: list[list[Point]] = []
    cur: list[Point] = []
    for (s0, p0), (s1, p1) in zip(samples, samples[1:]):
        mid = 0.5 * (s0 + s1)
        if any(lo <= mid <= hi for lo, hi in cuts):
            if len(cur) > 1:
                out.append(cur)
            cur = []
        else:
            if not cur:
                cur = [p0]
            cur.append(p1)
    if len(cur) > 1:
        out.append(cur)
    return out


# --------------------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------------------


def _label(dr: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, f) -> None:
    """A crossing number on a paper chip, so it stays legible over a photograph."""
    x, y = xy
    x0, y0, x1, y1 = dr.textbbox((x, y), text, font=f, anchor="ls")
    dr.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=PAPER)
    dr.text((x, y), text, font=f, fill=INK, anchor="ls")


def _dashed_rect(dr: ImageDraw.ImageDraw, box, colour, dash: int = 3, width: int = 2) -> None:
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        dr.line([(x, y0), (min(x + dash, x1), y0)], fill=colour, width=width)
        dr.line([(x, y1), (min(x + dash, x1), y1)], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        dr.line([(x0, y), (x0, min(y + dash, y1))], fill=colour, width=width)
        dr.line([(x1, y), (x1, min(y + dash, y1))], fill=colour, width=width)


def render(
    d: Diagram,
    verdict: Verdict | None = None,
    image: Image.Image | None = None,
    *,
    size: tuple[int, int] = (900, 620),
    stages: Iterable[str] = STAGES,
    pair: tuple[int, int] = (0, 1),
    bearing: float | None = None,
    cable_width: int = 5,
) -> Image.Image:
    """One overlay frame.

    `image` is the photograph the diagram was traced from, in which case the diagram's
    coordinates are already its pixels; with no image the diagram is drawn on paper.
    `stages` selects how much is drawn, which is how the GIF builds up.
    """
    stages = tuple(stages)
    if image is not None:
        photo = image.convert("RGB")
        w, h = photo.size
        canvas = Image.new("RGB", (w, h + BANNER_H + FOOTER_H), PAPER)
        canvas.paste(photo, (0, BANNER_H))
        box = (0, BANNER_H, w, h)
        frame = (0.0, 0.0, float(w), float(h))
        to_px_raw, scale = _fit(frame, box)
        to_px = to_px_raw
    else:
        w, h = size
        canvas = Image.new("RGB", (w, h + BANNER_H + FOOTER_H), PAPER)
        pad = 34
        box = (pad, BANNER_H + pad, w - 2 * pad, h - 2 * pad)
        to_px, scale = _fit(d.frame, box)

    dr = ImageDraw.Draw(canvas)
    states = glyph_states(d, verdict)
    gap_diag = GAP_PX / max(scale, 1e-9)

    # -- cables, with the under-strand broken at every READ crossing --------------------
    if "cables" in stages:
        for cab in d.cables:
            colour = CABLE_RGB[cab.id % len(CABLE_RGB)]
            cuts = breaks(d, cab.id, gap_diag) if "crossings" in stages else []
            for piece in pieces(cab.points, cab.closed, cuts):
                dr.line([to_px(p) for p in piece], fill=colour, width=cable_width, joint="curve")

    # -- crossings ----------------------------------------------------------------------
    if "crossings" in stages:
        f = font(14)
        r = 7
        for c in d.crossings:
            x, y = to_px(c.xy)
            state = states[c.id]
            if state == READ:
                dr.rectangle((x - r, y - r, x + r, y + r), fill=INK)
            else:
                ru = r + 3
                _dashed_rect(dr, (x - ru, y - ru, x + ru, y + ru), UNKNOWN_RGB, dash=4)
                # the `?` sits outside the box, not on the node: dead centre is where the
                # two strands cross, and both of them stay continuous at an unknown
                # crossing, so anything drawn there is illegible by construction.
                dr.text((x - ru - 4, y), "?", font=font(16), fill=UNKNOWN_RGB, anchor="rm")
            if state == TARGET:
                dr.ellipse((x - r - 11, y - r - 11, x + r + 11, y + r + 11), outline=TARGET_RGB, width=3)
                b = bearing
                if b is None:
                    try:
                        nxt = next_crossing(d, *pair)
                    except (ValueError, IndexError):
                        nxt = None
                    b = nxt[2] if nxt is not None and nxt[0] == c.id else None
                if b is not None:
                    a = math.radians(b)
                    dx, dy = math.cos(a) * 34, math.sin(a) * 34
                    dr.line([(x + dx * 0.5, y + dy * 0.5), (x + dx, y + dy)], fill=TARGET_RGB, width=3)
                    dr.regular_polygon((x + dx, y + dy, 6), 3, rotation=90 - b, fill=TARGET_RGB)
            _label(dr, (x + r + 6, y - r - 4), str(c.id), f)

    # -- banner and footer ---------------------------------------------------------------
    if "banner" in stages:
        status = verdict.status if verdict is not None else NOT_CERTIFIED
        ground = BANNER_RGB.get(status, INK)
        dr.rectangle((0, 0, canvas.width, BANNER_H - 1), fill=ground)
        head, num, action = banner_text(verdict)
        dr.text((22, 22), head, font=font(30), fill=(255, 255, 255), anchor="ls")
        nfont = font(40)
        room = canvas.width - 44 - (dr.textlength(num, font=nfont) + 22 if num else 0)
        if action:
            afont = font(15)
            action = action[:96]
            if dr.textlength(action, font=afont) > room:
                while action and dr.textlength(action + " ...", font=afont) > room:
                    action = action[: action.rfind(" ")] if " " in action else action[:-2]
                action += " ..."
            dr.text((22, 62), action, font=afont, fill=(238, 238, 238), anchor="ls")
        if num:
            dr.text((canvas.width - 22, 46), num, font=nfont, fill=(255, 255, 255), anchor="rm")

    dr.rectangle((0, canvas.height - FOOTER_H, canvas.width, canvas.height), fill=FOOTER_RGB)
    foot, ffont = footer_text(d, verdict), font(12)
    # Narrow pages (a 512 px render) cannot hold the whole line; drop trailing fields, never
    # the digest, rather than letting the text run off the edge.
    while dr.textlength(foot, font=ffont) > canvas.width - 24 and "  |  " in foot:
        foot = foot.rsplit("  |  ", 1)[0]
    dr.text(
        (12, canvas.height - FOOTER_H // 2),
        foot,
        font=ffont,
        fill=(90, 90, 92),
        anchor="lm",
    )
    return canvas


def frames(
    d: Diagram,
    verdict: Verdict | None = None,
    image: Image.Image | None = None,
    **kw,
) -> list[Image.Image]:
    """The build-up a GIF plays: bare, then skeletons, then crossings, then the banner.

    Every frame carries the diagram digest in its footer, so a frame that is not of this
    diagram is visible in the picture rather than only in the commit history.
    """
    return [
        render(d, verdict, image, stages=s, **kw)
        for s in ((), ("cables",), ("cables", "crossings"), STAGES)
    ]


def save_gif(path: str, seq: Sequence[Image.Image], duration: int = 1100) -> str:
    if not seq:
        raise ValueError("save_gif needs at least one frame")
    head, rest = seq[0], list(seq[1:])
    head.save(path, save_all=True, append_images=rest, duration=duration, loop=0, optimize=False)
    return path


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    from .certify import certify

    hopf = Diagram.from_braid([1, 1])
    v = certify(hopf)
    assert v.status == CERTIFIED and v.claim == LINKED, v

    states = glyph_states(hopf, v)
    assert set(states.values()) == {READ}, states
    # both strands of a read crossing cannot both be continuous: exactly one is broken
    cuts = [len(breaks(hopf, i, 0.05)) for i in range(len(hopf.cables))]
    assert sum(cuts) == len(hopf.crossings), cuts

    img = render(hopf, v)
    assert img.size[1] == 620 + BANNER_H + FOOTER_H, img.size
    assert img.getpixel((10, 6)) == BANNER_RGB[CERTIFIED], img.getpixel((10, 6))

    blurred = hopf.resolve({0: None})
    vb = certify(blurred)
    assert vb.status == REFUSED, vb
    sb = glyph_states(blurred, vb)
    assert sb[0] == TARGET and sb[1] == READ, sb
    ib = render(blurred, vb)
    assert ib.getpixel((10, 6)) == BANNER_RGB[REFUSED]

    assert len(frames(hopf, v)) == 4
    head, num, action = banner_text(v)
    assert head == LINKED and num == "lk = 1", (head, num)
    assert hopf.digest()[:12] in footer_text(hopf, v)
    print("ok")


if __name__ == "__main__":  # pragma: no cover
    _demo()
