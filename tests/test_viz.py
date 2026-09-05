"""Structural tests for the overlay.  No byte-exact golden PNG exists and none can.

PNG bytes depend on the zlib build and on Pillow's rasteriser, so determinism is claimed
for `Diagram.digest()` -- which every frame prints -- and the picture is checked the way
the spec says to check it: glyph count and state, banner colour at fixed pixels, footer
string, and the one geometric assertion the whole convention rests on, that a READ crossing
breaks exactly the under-strand and an UNKNOWN one breaks neither.
"""

from __future__ import annotations

import math

import pytest

from tangle import BANNED, CERTIFIED, LINKED, NOT_CERTIFIED, REFUSED, Diagram, certify
from tangle import viz


@pytest.fixture
def hopf():
    """A clasp: two crossings of equal sign, |lk| = 1, CERTIFIED LINKED."""
    return Diagram.from_braid([1, 1])


# --------------------------------------------------------------------------------------
# glyphs
# --------------------------------------------------------------------------------------


def test_glyph_state_per_crossing(hopf):
    v = certify(hopf)
    states = viz.glyph_states(hopf, v)
    assert len(states) == len(hopf.crossings)
    assert set(states.values()) == {viz.READ}

    blurred = hopf.resolve({0: None})
    vb = certify(blurred)
    sb = viz.glyph_states(blurred, vb)
    assert sb[1] == viz.READ
    # crossing 0 is unknown *and* named by the refusal, so it is drawn as the target
    assert sb[0] == viz.TARGET and 0 in vb.look_at


def test_unknown_glyph_without_a_verdict_is_not_a_target(hopf):
    blurred = hopf.resolve({0: None})
    assert viz.glyph_states(blurred, None)[0] == viz.UNKNOWN


def test_read_crossing_breaks_the_under_strand_only(hopf):
    """The one assertion the diagram convention rests on.

    A read crossing puts a gap in exactly one of the two strands.  An unknown crossing puts
    a gap in neither: the picture does not draw a decision that was not made.
    """
    gap = 0.05
    per_cable = [len(viz.breaks(hopf, c.id, gap)) for c in hopf.cables]
    assert sum(per_cable) == len(hopf.crossings)

    blurred = hopf.resolve({cid: None for cid in range(len(hopf.crossings))})
    assert sum(len(viz.breaks(blurred, c.id, gap)) for c in blurred.cables) == 0


def test_breaks_actually_split_the_polyline(hopf):
    """A break interval must remove length, and the removed length must be the gap."""
    cab = hopf.cables[0]
    cuts = viz.breaks(hopf, 0, 0.05)
    assert cuts, "cable 0 is the under-strand somewhere in a clasp"
    whole = viz.pieces(cab.points, cab.closed, [])
    cut = viz.pieces(cab.points, cab.closed, cuts)
    length = lambda ps: sum(math.dist(a, b) for a, b in zip(ps, ps[1:]))
    lost = sum(map(length, whole)) - sum(map(length, cut))
    assert lost == pytest.approx(0.10 * len(cuts), rel=0.02)
    assert len(cut) > len(whole)


def test_pieces_of_an_uncut_line_is_the_line():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert viz.pieces(pts, False, []) == [pts]


# --------------------------------------------------------------------------------------
# the banner
# --------------------------------------------------------------------------------------


def test_banner_colour_at_three_fixed_pixels(hopf):
    v = certify(hopf)
    img = viz.render(hopf, v)
    for xy in ((6, 5), (img.width // 2, 5), (img.width - 6, 5)):
        assert img.getpixel(xy) == viz.BANNER_RGB[CERTIFIED], xy

    blurred = hopf.resolve({0: None})
    ref = viz.render(blurred, certify(blurred))
    assert ref.getpixel((6, 5)) == viz.BANNER_RGB[REFUSED]
    assert viz.BANNER_RGB[REFUSED] != viz.BANNER_RGB[CERTIFIED] != viz.BANNER_RGB[NOT_CERTIFIED]


def test_banner_text_carries_the_verdict_and_the_named_crossing(hopf):
    head, num, action = viz.banner_text(certify(hopf))
    assert head == LINKED and num == "lk = 1" and action

    blurred = hopf.resolve({0: None})
    vb = certify(blurred)
    head, num, _ = viz.banner_text(vb)
    assert head.startswith(REFUSED) and "look at crossing 0" in head

    # a certified bound with k > 0 is a bound, never an exact number
    wide = Diagram.from_braid([1, 1, 1, 1]).resolve({0: None})
    head, num, _ = viz.banner_text(certify(wide))
    assert head == LINKED and num == "|lk| >= 1"


def test_banner_never_prints_a_banned_phrase(hopf):
    for d in (hopf, hopf.resolve({0: None}), Diagram.from_braid([1, -1])):
        v = certify(d)
        blob = " ".join(viz.banner_text(v) + (viz.footer_text(d, v),)).lower()
        for word in BANNED:
            assert word not in blob, (word, blob)


def test_footer_carries_the_digest(hopf):
    v = certify(hopf)
    assert hopf.digest()[:12] in viz.footer_text(hopf, v)
    assert "k = 0" in viz.footer_text(hopf, v)


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------


def test_frames_build_up_and_share_a_size(hopf):
    seq = viz.frames(hopf, certify(hopf))
    assert len(seq) == 4
    assert len({f.size for f in seq}) == 1
    blobs = [f.tobytes() for f in seq]
    assert len(set(blobs)) == 4, "every frame must add something"


def test_save_gif_writes_an_animation(tmp_path, hopf):
    from PIL import Image

    out = str(tmp_path / "t.gif")
    viz.save_gif(out, viz.frames(hopf, certify(hopf)))
    with Image.open(out) as im:
        assert im.n_frames == 4
    with pytest.raises(ValueError):
        viz.save_gif(out, [])


def test_render_over_a_photograph_keeps_its_pixels(hopf):
    """With an image the diagram's coordinates are already its pixels; the photo is pasted
    unscaled between the banner and the footer, so the tracer's numbering lands where it
    was measured."""
    from PIL import Image

    photo = Image.new("RGB", (400, 300), (12, 200, 12))
    img = viz.render(hopf, certify(hopf), photo)
    assert img.size == (400, 300 + viz.BANNER_H + viz.FOOTER_H)
    # the braid's coordinates live in the top-left corner; the far corner is untouched photo
    assert img.getpixel((398, viz.BANNER_H + 298)) == (12, 200, 12)


# --------------------------------------------------------------------------------------
# the font, which replaces a committed binary
# --------------------------------------------------------------------------------------


def test_font_digits_are_tabular():
    """The reason no TTF ships in this repo.

    A 64 px banner integer must not shift between GIF frames when a digit changes, which
    needs equal digit advances -- not a monospace face.  Measured on the face this actually
    picks; the test fails if a future Pillow default stops being tabular, which is the
    condition under which a bundled font becomes necessary again.
    """
    f = viz.font(48)
    widths = {f.getlength(c) for c in "0123456789"}
    assert len(widths) == 1, widths
