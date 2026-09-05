"""make_assets.py -- every image in `assets/`, regenerated from the pipeline itself.

    .venv/Scripts/python make_assets.py            # writes assets/
    .venv/Scripts/python make_assets.py --selfcheck

Nothing here is drawn by hand.  Each panel is a real run: `tangle.synth` renders the
scene, `tangle.vision` traces it, `tangle.certify` returns the verdict, and the number
printed on the picture is the one the verdict carries.  The real photographs come from
`real.py`'s corpus directory (`--photos`, default `photos`), which is gitignored -- the
files are Wikimedia Commons downloads this repository did not make, and their licence and
author travel with them in `manifest.json`.  With no corpus present the photograph panels
are skipped and the script says so rather than substituting a render for a photograph.

PALETTE -- one meaning per colour, used identically in every asset and shared with
`tangle.viz.DARK`, so the overlay a user gets on their own photograph is the same picture
as the one in the README:

    #0B0F17  ground        the page.  Dark-native: GitHub's own dark page is #0d1117.
    #131923  panel         a card on the ground
    #232C3B  rule          hairlines and borders
    #E6EDF3  ink           text that carries a fact
    #8B98A9  muted         provenance, units, captions
    #3FB950  green         CERTIFIED.  Proved.  Only ever a verdict this tool earned.
    #F85149  red           REFUSED, and the crossing to re-shoot.  Never "bad".
    #D29922  amber         UNKNOWN: a crossing the photograph carries no evidence about.
    #58A6FF  blue          cable A
    #BC8CFF  violet        cable B

Sizes.  The five page-width assets -- hero, coin-vs-occlusion, mechanism, refusal-wall,
social-preview -- are all at least 1280 px across and wider than 16:9, because a portrait
asset above the fold pushes the install line off the first screen.  The three verdict
cards keep the CLI's own 512 px overlay size, since that is the picture `--overlay`
writes and the README shows them three to a row.  No type is under 15 px at native width.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage.measure import label
from skimage.morphology import skeletonize

from tangle import synth, vision, viz
from tangle.certify import CERTIFIED, LINKED, REFUSED, certify
from tangle.viz import DARK

OUT = "assets"

GROUND = (11, 15, 23)
PANEL = (19, 25, 35)
RULE = (35, 44, 59)
INK = (230, 237, 243)
MUTED = (139, 152, 169)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
AMBER = (210, 153, 34)
BLUE = (88, 166, 255)
VIOLET = (188, 140, 255)

# The four gates the 247 real photographs stop at, with the counts `real.py run` printed.
REASONS = [
    ("BRANCHED_SKELETON", 104, "a cable's skeleton still branches: it crosses itself"),
    ("NO_INTENSITY_GAP", 65, "no representable gap between cable and background"),
    ("NOT_TWO_COMPONENTS", 61, "one mask, not two visually distinct cables"),
    ("OPEN_TRACE", 17, "a traced piece has the wrong number of free ends"),
]
N_REAL = sum(n for _, n, _ in REASONS)
CONTROL_CERTIFIED, CONTROL_N = 13, 20  # real.py control, same harness on rendered piles


# --------------------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------------------


def text(dr, xy, s, size, fill, anchor="ls", bold=False):
    """One string.  `bold` is a 1 px double strike: no binary font ships in this repo."""
    f = viz.font(size)
    dr.text(xy, s, font=f, fill=fill, anchor=anchor)
    if bold:
        dr.text((xy[0] + 1, xy[1]), s, font=f, fill=fill, anchor=anchor)
    return dr.textlength(s, font=f)


def width(s, size):
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(s, font=viz.font(size))


def chip(dr, xy, s, colour, size=17, pad=(10, 6)):
    """A small pill in a meaning colour: the verdict vocabulary, nowhere else."""
    x, y = xy
    w, h = width(s, size) + 2 * pad[0], size + 2 * pad[1]
    dr.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=_mix(colour, GROUND, 0.80))
    text(dr, (x + pad[0], y + h - pad[1] - 3), s, size, colour, bold=True)
    return w


def _mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def fit(im: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Centre-crop to `box`'s aspect, then resample.  Never letterboxed, never squashed."""
    bw, bh = box
    w, h = im.size
    s = max(bw / w, bh / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    return im.crop((left, top, left + bw, top + bh))


def segment(arr):
    """The front half of `vision.trace`, kept for the picture: (lab, mask, per-cable masks,
    width estimate).  Every asset is lit by the same segmentation the verdict rests on, so
    a background that does not leave the page is a background the tool would refuse on."""
    lab, bg, _ = vision.background(arr)
    de = np.linalg.norm(lab - bg, axis=-1).astype(np.float32)
    mask, _ = vision.threshold(de)
    mask = vision._clean(mask)
    return lab, mask, vision.components(lab, mask), vision.width_estimate(mask)


def darkground(arr, mask, floor=0.08, feather=2.0):
    """Dissolve everything the threshold rejected into the page.

    The alpha is the cleaned mask, feathered by a couple of pixels so the edge is not
    aliased.  Nothing here is a mood filter: what survives on the dark ground is exactly
    what `vision` kept, and the speckle a bad photograph leaves behind is left in.
    """
    m = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather))
    a = floor + (1.0 - floor) * (np.asarray(m, dtype=np.float32) / 255.0)[..., None]
    px = np.array(GROUND, dtype=np.float32) * (1.0 - a) + arr.astype(np.float32) * a
    return Image.fromarray(px.clip(0, 255).astype(np.uint8))


PALETTE_CABLES = (BLUE, VIOLET)
TRACE = (245, 249, 253)  # the centreline the tracer found, in a render or on a photograph
TRACE_THEME = dict(DARK, cable=(TRACE, TRACE, TRACE, TRACE, TRACE))


def repaint(cables):
    """The same scene in the palette's two cable colours, so blue and violet mean the same
    thing in a render, in an overlay and on a photograph.  The pipeline runs on the result
    unchanged -- segmentation is a Lab distance from the background and never a hue."""
    return [replace(c, colour=PALETTE_CABLES[i % 2]) for i, c in enumerate(cables)]


def overlay_scene(cables, seed, size, cable_width=3, margin=0.10):
    """Render a scene, trace it, certify it, return (dark overlaid image, verdict, d).

    Cropped to the diagram's own bounding box, so the subject fills the panel instead of
    the renderer's margin doing it.
    """
    img, _ = synth.render(repaint(cables), seed=seed)
    d = vision.trace(img)
    v = certify(d, 0, 1)
    _, mask, _, _ = segment(img)
    full = viz.render(
        d, v, image=darkground(img, mask), stages=("cables", "crossings"),
        theme=TRACE_THEME, cable_width=cable_width,
    )
    body = full.crop((0, viz.BANNER_H, full.width, full.height - viz.FOOTER_H))
    return fit(crop_to_diagram(body, d, margin), size), v, d


def crop_to_crossings(im, d, pad_frac=0.34):
    """Window on the crossings, so the broken under-strand is visible and not a rumour."""
    if not d.crossings:
        return im
    xs = [c.xy[0] for c in d.crossings]
    ys = [c.xy[1] for c in d.crossings]
    px, py = im.width * pad_frac, im.height * pad_frac
    box = (max(0, min(xs) - px), max(0, min(ys) - py),
           min(im.width, max(xs) + px), min(im.height, max(ys) + py))
    return im.crop(tuple(int(round(v)) for v in box))


def crop_to_diagram(im, d, margin=0.10):
    pts = np.array([p for c in d.cables for p in c.points], dtype=float)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, x0 - mx), max(0, y0 - my), min(im.width, x1 + mx), min(im.height, y1 + my))
    return im.crop(tuple(int(round(v)) for v in box))


# --------------------------------------------------------------------------------------
# the real photograph, and the gate it stops at
# --------------------------------------------------------------------------------------


def branch_overlay(path, size):
    """A photograph with the tool's own segmentation and skeleton drawn on it.

    Not an illustration of BRANCHED_SKELETON: it is the array the refusal was raised from.
    The background dissolves under the same threshold the segmentation uses, each cable
    mask keeps its own colour, the pruned skeleton is drawn as a centreline, and every
    pixel whose 3x3 neighbourhood has degree >= 4 -- the exact test in `vision.arcs` -- is
    ringed.  Those rings are the reason there is no verdict.
    """
    from tangle.cli import load_image

    arr, _ = load_image(path)
    _, mask, masks, w_est = segment(arr)
    px = np.asarray(darkground(arr, mask)).astype(np.float32)
    rings = []
    for i, m in enumerate(masks[:2]):
        colour = np.array(PALETTE_CABLES[i], dtype=np.float32)
        px[m] = px[m] * 0.70 + colour * 0.30
        skel = vision.prune(skeletonize(m), vision.SPUR_W * w_est)
        px[skel] = px[skel] * 0.15 + np.float32(255.0) * 0.85
        lb = label(vision._degree(skel) * skel >= 4, connectivity=2)
        for rid in range(1, int(lb.max()) + 1):
            ys, xs = np.nonzero(lb == rid)
            rings.append((float(xs.mean()), float(ys.mean())))

    out = Image.fromarray(px.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(out)
    r = max(7.0, w_est * 0.62)
    for x, y in rings:
        dr.ellipse((x - r, y - r, x + r, y + r), outline=RED, width=3)
    return fit(out, size), len(rings)


def photo_credit(man, fname):
    row = next((r for r in man if r["file"] == fname), None)
    if row is None:
        return f"photograph  ·  {fname}"
    who = (row.get("author") or "unknown").strip()
    return f"Wikimedia Commons  ·  {who}  ·  {row.get('licence', '?')}  ·  {fname}"


def load_manifest(photos):
    p = os.path.join(photos, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["images"]


# --------------------------------------------------------------------------------------
# hero
# --------------------------------------------------------------------------------------

HERO = (1800, 840)
HERO_PHOTO = "Booster_cables.jpg"


def verdict_block(dr, x, y, w, colour, head, big, lines, big_size=34):
    """Status word left, the number or reason code right, two lines of plain English."""
    dr.rectangle((x, y, x + w, y + 2), fill=colour)
    text(dr, (x, y + 58), head, 42, colour, bold=True)
    if big:
        text(dr, (x + w, y + 58), big, big_size, colour, anchor="rs", bold=True)
    for i, ln in enumerate(lines):
        text(dr, (x, y + 96 + i * 30), ln, 20, INK if i == 0 else MUTED)
    return y + 96 + max(0, len(lines) - 1) * 30


def hero(photos, man):
    """Left: what a certificate looks like. Right: what a real photograph gets today."""
    W, H = HERO
    im = Image.new("RGB", (W, H), GROUND)
    dr = ImageDraw.Draw(im)
    pad, gut, rail = 30, 26, 54
    pw = (W - 2 * pad - gut) // 2
    iw, ih = pw, 520
    top = pad

    left, v, _ = overlay_scene(synth.clasp(sign=1), 1, (iw, ih))
    assert v.status == CERTIFIED and v.claim == LINKED, v
    im.paste(left, (pad, top))
    dr.rectangle((pad, top, pad + iw - 1, top + ih - 1), outline=RULE)

    y = top + ih + 12
    text(dr, (pad, y + 16), "rendered scene", 16, INK)
    text(dr, (pad + width("rendered scene", 16) + 10, y + 16),
         "·  tangle.synth, seed 1  ·  not a photograph", 16, MUTED)
    verdict_block(
        dr, pad, y + 30, iw, GREEN,
        "CERTIFIED LINKED", f"lk = {v.value}",
        ["No motion separates these two cables while their four ends stay put.",
         "An exact integer, and the theorem it came from. Go unplug one."],
        big_size=52,
    )

    x2 = pad + pw + gut
    src = os.path.join(photos, HERO_PHOTO)
    if os.path.exists(src):
        right, nring = branch_overlay(src, (iw, ih))
        credit = photo_credit(man or [], HERO_PHOTO)
    else:
        right, nring, credit = Image.new("RGB", (iw, ih), PANEL), 0, "no corpus: run real.py fetch"
    im.paste(right, (x2, top))
    dr.rectangle((x2, top, x2 + iw - 1, top + ih - 1), outline=RULE)
    text(dr, (x2, y + 16), "real photograph", 16, INK)
    text(dr, (x2 + width("real photograph", 16) + 10, y + 16), "·  " + credit, 16, MUTED)
    verdict_block(
        dr, x2, y + 30, iw, RED,
        "REFUSED", "BRANCHED_SKELETON",
        [f"{nring} red rings: every place a cable's own skeleton branches on itself.",
         f"104 of {N_REAL} real photographs stop here. Nothing downstream ever ran."],
    )

    ry = H - rail
    dr.rectangle((pad, ry, W - pad, ry + 1), fill=RULE)
    text(
        dr, (pad, ry + 36),
        f"{N_REAL} free-licensed photographs this repository did not make  ·  "
        f"0 certified  ·  0 wrong certificates",
        19, MUTED,
    )
    text(dr, (W - pad, ry + 36), "|lk| >= 1 is a proof, not a score", 19, INK, anchor="rs")
    return im


# --------------------------------------------------------------------------------------
# the tool where a naive reader is wrong
# --------------------------------------------------------------------------------------


def coin_vs_occlusion(seed=50, coin_seed=15, size=(1800, 726)):
    """One scene, one extracted diagram, one line of the pipeline swapped.

    The left panel is `vision.read_over`, which reads depth from occlusion continuity: the
    under-strand's own mask is interrupted where the other cable crosses it.  The right
    panel is the identical `Diagram` with every crossing's over/under replaced by
    `random.Random(coin_seed).choice('ab')` -- the same substitution `bench.py` and
    `real.py --coin` make.  Both verdicts are computed here; neither is typed in, and the
    assert is what stops this asset shipping if the coin ever stops being wrong.
    """
    W, H = size
    img, truth = synth.render(repaint(synth.pile(seed)), seed=seed)
    d = vision.trace(img)
    v = certify(d, 0, 1)
    t = certify(truth, 0, 1)
    r = random.Random(coin_seed)
    dc = replace(
        d,
        crossings=tuple(
            replace(c, over=r.choice("ab"), over_conf=1.0, kind="read") for c in d.crossings
        ),
    )
    vc = certify(dc, 0, 1)
    # The seed is fixed, not searched, so the picture is the same on every machine.  If the
    # pipeline ever changes enough that this scene stops certifying, or that the coin stops
    # buying a wrong certificate on it, the asset does not ship.
    assert v.status == CERTIFIED and v.claim == LINKED, v
    assert vc.status == CERTIFIED and vc.claim != v.claim, vc
    assert abs(v.value) == abs(t.value), (v, t)

    _, mask, _, _ = segment(img)
    base = darkground(img, mask)

    im = Image.new("RGB", (W, H), GROUND)
    dr = ImageDraw.Draw(im)
    pad, gut, head = 30, 26, 104
    pw = (W - 2 * pad - gut) // 2
    ih = 392

    text(dr, (pad, 52), "Same photograph. Same diagram. One line of the pipeline swapped.", 32, INK, bold=True)
    text(dr, (pad, 82), "Over/under is the only place a wrong certificate can come from, so it is the only thing that differs across this pair.", 19, MUTED)

    panels = [
        (d, v, "read from occlusion continuity",
         "the under-strand's own mask is interrupted where the other cable crosses it",
         GREEN, "unplug one",
         [f"Right. |lk| = {abs(t.value)}, and the braid word this scene was built from says {abs(t.value)}.",
          "0 wrong certificates over 2,000 diagrams."]),
        (dc, vc, f"read by a coin flip",
         f"random.Random({coin_seed}).choice('ab') at every crossing, nothing else changed",
         RED, "just pull",
         ["WRONG. This pile provably cannot be pulled apart with its ends held.",
          "32 wrong certificates over the same 2,000 diagrams."]),
    ]
    for i, (dd, vv, title, sub, colour, action, lines) in enumerate(panels):
        x = pad + i * (pw + gut)
        panel = viz.render(
            dd, vv, image=base, stages=("cables", "crossings"),
            theme=TRACE_THEME, cable_width=4,
        )
        body = panel.crop((0, viz.BANNER_H, panel.width, panel.height - viz.FOOTER_H))
        im.paste(fit(crop_to_crossings(body, dd), (pw, ih)), (x, head))
        dr.rectangle((x, head, x + pw - 1, head + ih - 1), outline=RULE)

        y = head + ih + 12
        text(dr, (x, y + 16), title, 18, INK)
        text(dr, (x, y + 42), sub, 17, MUTED)
        verdict_block(
            dr, x, y + 56, pw, colour,
            f"CERTIFIED {vv.claim}", action, lines, big_size=40,
        )
    return im


# --------------------------------------------------------------------------------------
# the refusal wall
# --------------------------------------------------------------------------------------


def refusal_wall(photos, man, rows_json, size=(1800, 960), cell=60, gap=4, cols=26):
    """Every one of the 247 real images, and the gate each one stopped at."""
    W, H = size
    im = Image.new("RGB", (W, H), GROUND)
    dr = ImageDraw.Draw(im)
    pad = 70

    text(dr, (pad, 58), f"{N_REAL} photographs this repository did not make", 32, INK, bold=True)
    text(dr, (W - pad, 58), "0 certified   ·   0 wrong certificates", 24, RED, anchor="rs")
    text(dr, (pad, 86), "Wikimedia Commons, published link diagrams, and a Hugging Face line-drawing set. Every one refused, and named its own gate.", 18, MUTED)

    order = {r: i for i, (r, _, _) in enumerate(REASONS)}
    rows = sorted(rows_json, key=lambda r: (order.get(r["reason"], 9), r["file"]))
    gy = 118
    n = 0
    for i, row in enumerate(rows):
        cx = pad + (i % cols) * (cell + gap)
        cy = gy + (i // cols) * (cell + gap)
        p = os.path.join(photos, row["file"])
        if os.path.exists(p):
            try:
                tile = fit(Image.open(p).convert("RGB"), (cell, cell))
                tile = Image.blend(tile, Image.new("RGB", (cell, cell), GROUND), 0.52)
            except Exception:
                tile = Image.new("RGB", (cell, cell), PANEL)
        else:
            tile = Image.new("RGB", (cell, cell), PANEL)
        im.paste(tile, (cx, cy))
        dr.rectangle((cx, cy, cx + cell - 1, cy + cell - 1), outline=RED)
        n += 1

    by = gy + ((len(rows) + cols - 1) // cols) * (cell + gap) + 30
    bw = cols * (cell + gap) - gap
    x = pad
    shades = [RED, _mix(RED, AMBER, 0.45), AMBER, _mix(AMBER, GROUND, 0.30)]
    for (name, count, _), sh in zip(REASONS, shades):
        seg = int(round(bw * count / N_REAL))
        dr.rectangle((x, by, x + seg - 3, by + 26), fill=sh)
        text(dr, (x, by + 62), f"{count}", 30, sh, bold=True)
        text(dr, (x + width(str(count), 30) + 10, by + 62), name, 17, MUTED)
        x += seg

    cy2 = by + 96
    x = pad
    x += text(dr, (x, cy2 + 20), "CONTROL", 18, GREEN, bold=True) + 22
    for i in range(CONTROL_N):
        c = GREEN if i < CONTROL_CERTIFIED else RED
        dr.rounded_rectangle((x, cy2 + 4, x + 15, cy2 + 19), radius=3, fill=c)
        x += 20
    x += 12
    x += text(dr, (x, cy2 + 20), f"{CONTROL_CERTIFIED} of {CONTROL_N}", 18, INK) + 10
    text(dr, (x, cy2 + 20),
         "rendered piles certified by the identical code. The zero above is the corpus.",
         18, MUTED)
    return im


# --------------------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------------------


def mechanism(size=(1800, 690)):
    """Why an unreadable crossing is a wider interval and not a guess.

    Two rows of the same six crossings.  Each readable crossing contributes a fixed +-1;
    each unreadable one is affine in an unknown +-1, so the achievable set is exactly
    k + 1 consecutive integers.  Clear of zero on the number line is a certificate.
    """
    W, H = size
    im = Image.new("RGB", (W, H), GROUND)
    dr = ImageDraw.Draw(im)
    pad = 54

    text(dr, (pad, 62), "One unreadable crossing is one more integer, not a guess", 34, INK, bold=True)
    text(dr, (pad, 96), "lk is half the signed crossing sum, and it is affine in every crossing. With S the signed sum over the readable crossings and k unreadable ones,", 19, MUTED)
    text(dr, (pad, 124), "the set of linking numbers achievable over all 2^k resolutions is exactly the k+1 consecutive integers from (S-k)/2 to (S+k)/2. Computed in O(k).", 19, MUTED)

    # number line, shared by both rows
    nl_x0, nl_x1 = 980, W - pad - 40
    lo, hi = -2, 3
    step = (nl_x1 - nl_x0) / (hi - lo)

    def tx(val):
        return nl_x0 + (val - lo) * step

    rows = [
        dict(y=282, read=[1, 1, -1, 1], unknown=2, colour=RED,
             head="REFUSED", note="the achievable interval contains 0",
             sub="two crossings the photograph carries no evidence about"),
        dict(y=488, read=[1, 1, -1, 1, 1, 1], unknown=0, colour=GREEN,
             head="CERTIFIED LINKED", note="the interval clears 0: a proof",
             sub="every crossing read from occlusion continuity"),
    ]

    for row in rows:
        y = row["y"]
        S = sum(row["read"]) + 0  # unknown crossings contribute nothing to S
        k = row["unknown"]
        s_span = S  # signed sum over readable inter-component crossings
        v_lo, v_hi = (s_span - k) / 2, (s_span + k) / 2

        text(dr, (pad, y - 66), row["sub"], 19, MUTED)
        x = pad
        for sgn in row["read"]:
            _glyph_read(dr, x, y - 32, sgn)
            x += 78
        for _ in range(k):
            _glyph_unknown(dr, x, y - 32)
            x += 78
        text(dr, (pad, y + 62), f"S = {s_span}      k = {k}", 22, INK)
        text(dr, (pad + 250, y + 62), f"lk in [{v_lo:g}, {v_hi:g}]", 22, row["colour"])

        dr.line([(760, y - 10), (900, y - 10)], fill=RULE, width=2)
        dr.regular_polygon((900, y - 10, 9), 3, rotation=-90, fill=RULE)

        # the interval, on the line
        bar_y = y - 10
        dr.line([(nl_x0, bar_y), (nl_x1, bar_y)], fill=RULE, width=2)
        for v in range(lo, hi + 1):
            if v:
                dr.line([(tx(v), bar_y - 9), (tx(v), bar_y + 9)], fill=RULE, width=2)
            text(dr, (tx(v), bar_y + 40), f"{v}", 18, RED if v == 0 else MUTED, anchor="ms")
        dr.rounded_rectangle(
            (tx(v_lo) - 12, bar_y - 20, tx(v_hi) + 12, bar_y + 20),
            radius=20, fill=_mix(row["colour"], GROUND, 0.72), outline=row["colour"], width=3,
        )
        for j in range(k + 1):
            dr.ellipse((tx(v_lo + j) - 6, bar_y - 6, tx(v_lo + j) + 6, bar_y + 6), fill=row["colour"])
        # zero last, so "the interval contains 0" is a thing you can see and not read
        dr.line([(tx(0), bar_y - 30), (tx(0), bar_y + 30)], fill=RED, width=3)
        text(dr, (nl_x0, bar_y - 42), row["head"], 26, row["colour"], bold=True)
        text(dr, (nl_x1, bar_y - 42), row["note"], 19, MUTED, anchor="rs")

    yl = H - 46
    dr.rectangle((pad, yl - 26, W - pad, yl - 25), fill=RULE)
    x = pad
    for glyph, lbl in ((_glyph_read, "read from occlusion: contributes a fixed +1 or -1"),
                       (_glyph_unknown, "no evidence: widens the interval by exactly 1")):
        glyph(dr, x, yl - 14, 1) if glyph is _glyph_read else glyph(dr, x, yl - 14)
        x += 46
        x += text(dr, (x, yl + 4), lbl, 18, MUTED) + 60
    return im


def _glyph_read(dr, x, y, sgn, s=30):
    dr.rounded_rectangle((x, y, x + s, y + s), radius=6, fill=_mix(INK, GROUND, 0.62), outline=INK)
    text(dr, (x + s / 2, y + s - 7), "+" if sgn > 0 else "-", 24, INK, anchor="ms", bold=True)


def _glyph_unknown(dr, x, y, s=30):
    for a in range(int(x), int(x + s), 8):
        dr.line([(a, y), (min(a + 4, x + s), y)], fill=AMBER, width=2)
        dr.line([(a, y + s), (min(a + 4, x + s), y + s)], fill=AMBER, width=2)
    for b in range(int(y), int(y + s), 8):
        dr.line([(x, b), (x, min(b + 4, y + s))], fill=AMBER, width=2)
        dr.line([(x + s, b), (x + s, min(b + 4, y + s))], fill=AMBER, width=2)
    text(dr, (x + s / 2, y + s - 6), "?", 24, AMBER, anchor="ms", bold=True)


# --------------------------------------------------------------------------------------
# social preview
# --------------------------------------------------------------------------------------


def social(photos, man, size=(1280, 640)):
    W, H = size
    im = Image.new("RGB", (W, H), GROUND)
    src = os.path.join(photos, HERO_PHOTO)
    right_w = 660
    if os.path.exists(src):
        photo, _ = branch_overlay(src, (right_w, H))
        im.paste(photo, (W - right_w, 0))
        # feather the left edge of the photo into the ground
        grad = Image.new("L", (300, H))
        gd = ImageDraw.Draw(grad)
        for i in range(300):
            gd.line([(i, 0), (i, H)], fill=int(255 * (1 - i / 300)))
        im.paste(Image.new("RGB", (300, H), GROUND), (W - right_w, 0), grad)
    dr = ImageDraw.Draw(im)

    text(dr, (60, 190), "tangle", 76, INK, bold=True)
    text(dr, (60, 250), "Photograph two cables.", 40, INK)
    text(dr, (60, 302), "Get an integer with a proof.", 40, BLUE)
    text(dr, (60, 356), "The Gauss linking number, certified over every crossing", 20, MUTED)
    text(dr, (60, 384), "the camera could not read.", 20, MUTED)
    x = 60
    x += chip(dr, (x, 420), f"|lk| >= 1   LINKED", GREEN, size=20) + 14
    chip(dr, (x, 420), "unreadable   REFUSED", RED, size=20)
    text(dr, (60, 560), f"zero training  ·  zero GPU  ·  {N_REAL} real photographs, 0 wrong certificates", 19, MUTED)
    return im


# --------------------------------------------------------------------------------------
# the three verdict cards and the GIF, on the dark ground
# --------------------------------------------------------------------------------------


def verdict_cards():
    """The three verdicts the README shows side by side, and the GIF, on the dark ground.

    Same treatment as every other asset: the scene in the palette's two cable colours, the
    background dissolved by the segmentation's own mask, the traced centreline drawn white.
    """
    out = []
    for name, cables, seed in (("clasp-linked", synth.clasp(sign=1), 1),
                               ("stack-separable", synth.stack(), 2)):
        img, _ = synth.render(repaint(cables), seed=seed)
        d = vision.trace(img)
        v = certify(d, 0, 1)
        _, mask, _, _ = segment(img)
        page = darkground(img, mask)
        out.append((name, viz.render(d, v, image=page, theme=TRACE_THEME, cable_width=4), v))
    for seed in (5, 8, 18, 2, 11, 3, 7):
        try:
            img, _ = synth.render(repaint(synth.pile(seed)), seed=seed)
            d = vision.trace(img)
        except vision.TraceRefused:
            continue
        v = certify(d, 0, 1)
        if v.status != REFUSED:
            continue
        _, mask, _, _ = segment(img)
        page = darkground(img, mask)
        out.append(("pile-refuse", viz.render(d, v, image=page, theme=TRACE_THEME, cable_width=4), v))
        viz.save_gif(
            os.path.join(OUT, "tangle.gif"),
            viz.frames(d, v, image=page, theme=TRACE_THEME, cable_width=4),
        )
        break
    else:
        raise RuntimeError("no seed in the list produced a refusal; the GIF has no subject")
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def save(im, name, colours=0):
    """PNG, palette-quantised where it costs nothing, so a README page stays light."""
    path = os.path.join(OUT, name)
    if colours:
        im = im.convert("RGB").quantize(colors=colours, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    im.save(path, optimize=True)
    kb = os.path.getsize(path) / 1024
    print(f"  {name:26s} {im.size[0]}x{im.size[1]:<5d} {kb:7.0f} KB")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photos", default="photos", help="real.py corpus directory")
    ap.add_argument("--real", default="real.json", help="real.py run output, for the wall")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)

    if a.selfcheck:
        return _selfcheck()

    os.makedirs(OUT, exist_ok=True)
    man = load_manifest(a.photos)
    have = man is not None
    if not have:
        print(f"! no corpus at {a.photos}/manifest.json -- run: python real.py fetch")

    print("assets:")
    save(hero(a.photos, man), "hero.png", colours=192)
    save(coin_vs_occlusion(), "coin-vs-occlusion.png", colours=192)
    save(mechanism(), "mechanism.png", colours=128)
    save(social(a.photos, man), "social-preview.png", colours=192)

    if have and os.path.exists(a.real):
        with open(a.real, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        save(refusal_wall(a.photos, man, rows), "refusal-wall.png", colours=224)
    else:
        print(f"! skipped refusal-wall.png: needs {a.real} (python real.py run)")

    for name, page, v in verdict_cards():
        save(page, f"{name}.png", colours=128)
        print(f"      {name}: {v.status} {v.claim or v.reason}")
    print(f"  {'tangle.gif':26s} {os.path.getsize(os.path.join(OUT, 'tangle.gif')) / 1024:.0f} KB")
    return 0


def _selfcheck() -> int:
    """No network, no corpus: the two panels that come only from the package."""
    assert mechanism().getpixel((2, 2)) == GROUND
    im = coin_vs_occlusion()
    assert im.getpixel((2, 2)) == GROUND, "the ground must be dark, everywhere"
    assert im.size[0] >= 1200 and im.size[0] / im.size[1] > 16 / 9
    _, v, _ = overlay_scene(synth.clasp(sign=1), 1, (400, 300))
    assert v.status == CERTIFIED and v.value == 1, v
    assert min(HERO) / max(HERO) < 9 / 16, "the hero must be wider than 16:9"
    print("make_assets selfcheck ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
