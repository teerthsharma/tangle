"""The CLI, its exit codes, and the corpus both it and `bench.py` are built from.

Two things are asserted here that are not asserted anywhere else in the package: that the
seeded corpus is reproducible (same seed, same digest, same pair, same word), and that the
corpus's `lk` agrees with the braid's closed form -- half the signed exponent sum -- so the
number `bench.py` scores against is not this package's own arithmetic.
"""

from __future__ import annotations

import io
import json
import random

import pytest

from tangle import BANNED, CERTIFIED, EXIT, NOT_CERTIFIED, REFUSED, certify, lk_interval
from tangle import cli


def run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli.main(argv, out=buf)
    return code, buf.getvalue()


# --------------------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------------------


def test_synthetic_is_reproducible():
    a, pa, wa = cli.synthetic(7)
    b, pb, wb = cli.synthetic(7)
    assert (a.digest(), pa, wa) == (b.digest(), pb, wb)
    assert cli.synthetic(8)[0].digest() != a.digest()


@pytest.mark.parametrize("seed", range(20))
def test_synthetic_lk_matches_the_braid_closed_form(seed):
    """The control.  lk on two strands is half the signed exponent sum, known before any
    diagram is built, so the bench's ground truth is not the code under test."""
    d, (i, j), word = cli.synthetic(seed)
    assert abs(lk_interval(d, i, j).exact) == abs(sum(1 if w > 0 else -1 for w in word)) // 2


def test_synthetic_refuses_an_odd_word_length():
    with pytest.raises(ValueError):
        cli.synthetic(1, letters=5)


def test_synthetic_refuses_fewer_than_two_strands():
    for strands in (0, 1, -3):
        with pytest.raises(ValueError):
            cli.synthetic(0, strands=strands)


def test_synthetic_refuses_a_point_budget_it_would_hang_on():
    with pytest.raises(ValueError):
        cli.synthetic(0, letters=cli.MAX_WORD_POINTS, strands=2)


# --------------------------------------------------------------------------------------
# a stranger's CLI args never raise -- every bad --synthetic knob is EXIT_INPUT, not a
# traceback.  These reproduce actual crashes found in this module: an odd --letters or a
# --strands < 2 used to propagate uncaught past `main()`.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--synthetic", "--letters", "5"],
        ["--synthetic", "--strands", "1"],
        ["--synthetic", "--strands", "0"],
        ["--synthetic", "--strands", "-3"],
        ["--synthetic", "--letters", "200000"],
        ["--synthetic", "--letters", "1000", "--strands", "64"],
    ],
)
def test_bad_synthetic_args_exit_input_not_crash(argv):
    code, txt = run(argv)
    assert code == cli.EXIT_INPUT
    assert txt.strip()


def test_blur_widens_the_interval_by_exactly_k():
    d, (i, j), _ = cli.synthetic(3)
    base = lk_interval(d, i, j)
    assert base.unknown == 0
    for k in range(len(d.between(i, j)) + 1):
        iv = lk_interval(cli.blur(d, i, j, k, random.Random(k)), i, j)
        assert iv.unknown == k and iv.hi - iv.lo == k
        assert base.exact in iv, "blurring cannot move the truth out of the interval"
    with pytest.raises(ValueError):
        cli.blur(d, i, j, len(d.between(i, j)) + 1, random.Random(0))


# --------------------------------------------------------------------------------------
# exit codes
# --------------------------------------------------------------------------------------


def test_certified_seed_exits_zero():
    code, txt = run(["--synthetic", "--seed", "7"])
    assert code == EXIT[CERTIFIED] == 0
    assert CERTIFIED in txt


def test_straddling_seed_refuses_and_names_a_crossing():
    code, txt = run(["--synthetic", "--seed", "3", "--letters", "8", "--unknown", "3"])
    assert code == EXIT[REFUSED] == 2
    assert "LK_STRADDLES_ZERO" in txt and "look at" in txt


def test_lk_zero_is_not_certified_and_still_prints_the_number():
    d, (i, j), _ = cli.synthetic(5)
    v = certify(d, i, j)
    assert v.status == NOT_CERTIFIED and v.value == 0
    code, txt = run(["--synthetic", "--seed", "5"])
    assert code == EXIT[NOT_CERTIFIED] == 1
    assert "lk = 0" in txt


def test_no_input_is_exit_three():
    assert run([])[0] == cli.EXIT_INPUT == 3
    assert run(["photo.jpg", "--synthetic"])[0] == cli.EXIT_INPUT
    assert run(["--synthetic", "--seed", "7", "--pair", "zero,one"])[0] == cli.EXIT_INPUT
    assert run(["--synthetic", "--seed", "7", "--unknown", "99"])[0] == cli.EXIT_INPUT


def test_a_missing_image_is_an_input_refusal():
    """Without the tracer the CLI must name the missing layer rather than raise
    ImportError at the user; with it, a path that cannot be opened is still exit 3."""
    code, txt = run(["nonexistent.jpg"])
    assert code == cli.EXIT_INPUT
    if cli.vision_entry() is None:
        assert "tangle.vision" in txt and "--synthetic" in txt
    else:
        assert "cannot read" in txt


def test_a_traced_image_produces_a_block_or_a_named_refusal(tmp_path):
    """End to end over a rendered scene: whatever the tracer does, the CLI prints a block
    inside its rules and exits with one of the four codes.  A tracer refusal is a REFUSED
    block with the tracer's own reason code, never a traceback."""
    if cli.vision_entry() is None:
        pytest.skip("the imaging layer is not installed")
    synth = pytest.importorskip("tangle.synth")
    png = str(tmp_path / "scene.png")
    synth.save(png, synth.clasp(1))

    code, txt = run([png])
    assert code in (0, 1, 2), txt
    assert txt.startswith("=" * cli.WIDTH)
    assert "convention" in txt
    if code == 2:
        assert any(w in txt for w in ("REFUSED",))


def test_block_without_a_diagram_prints_no_digest():
    from tangle import Verdict

    v = Verdict(status=REFUSED, reason="NO_INTENSITY_GAP", advice="x", exit_code=2)
    txt = cli.block(v, None, ascii_only=True)
    assert "digest" not in txt and "NO_INTENSITY_GAP" in txt


@pytest.mark.parametrize(
    "make",
    [
        lambda p: p.write_bytes(b""),  # zero-length file
        lambda p: p.write_bytes(b"not an image" * 20),  # corrupt file, plausible extension
    ],
    ids=["zero_length", "corrupt"],
)
def test_a_malformed_image_file_is_an_input_refusal_not_a_crash(tmp_path, make):
    if cli.vision_entry() is None:
        pytest.skip("the imaging layer is not installed")
    p = tmp_path / "bad.jpg"
    make(p)
    code, txt = run([str(p)])
    assert code == cli.EXIT_INPUT
    assert "cannot read" in txt


def test_a_directory_as_the_image_path_is_an_input_refusal(tmp_path):
    if cli.vision_entry() is None:
        pytest.skip("the imaging layer is not installed")
    code, txt = run([str(tmp_path)])
    assert code == cli.EXIT_INPUT
    assert "cannot read" in txt


def test_a_decompression_bomb_is_refused_not_raised(tmp_path, monkeypatch):
    """PIL's own bomb guard raises `DecompressionBombError`, which is not an `OSError` --
    `load_image` must convert it, or a big-enough file crashes past every guard in main()."""
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100 * 100)
    p = tmp_path / "big.png"
    Image.new("RGB", (500, 500), (1, 2, 3)).save(p)
    with pytest.raises(OSError):
        cli.load_image(str(p))
    # and through the CLI it is a named input refusal, not a traceback
    code, txt = run([str(p)])
    assert code == cli.EXIT_INPUT
    assert "cannot read" in txt


def test_load_image_handles_unicode_and_spaces_in_the_path(tmp_path):
    from PIL import Image

    p = tmp_path / "näme with spaces éè 中文.png"
    Image.new("RGB", (30, 30), (5, 5, 5)).save(p)
    arr, img = cli.load_image(str(p))
    assert arr.shape == (30, 30, 3)


def test_load_image_transposes_then_downscales(tmp_path):
    """EXIF first, because orientation moves the frame and the frame pins the four ends."""
    from PIL import Image

    p = str(tmp_path / "big.png")
    Image.new("RGB", (2400, 1200), (9, 9, 9)).save(p)
    arr, img = cli.load_image(p)
    assert max(img.size) == cli.MAX_SIDE == 1024
    assert arr.shape[:2] == (img.height, img.width)


# --------------------------------------------------------------------------------------
# the block
# --------------------------------------------------------------------------------------


def test_block_fits_inside_its_rules_and_carries_the_provenance():
    d, (i, j), _ = cli.synthetic(11)
    txt = cli.block(certify(d, i, j), d, ascii_only=True)
    assert max(len(line) for line in txt.splitlines()) <= cli.WIDTH
    assert txt.splitlines()[0] == "=" * cli.WIDTH
    assert d.digest() in txt and d.provenance in txt
    assert "convention" in txt


def test_ascii_fallback_when_stdout_cannot_carry_box_drawing():
    heavy_u, light_u = cli.rules(False)
    heavy_a, light_a = cli.rules(True)
    assert heavy_u.startswith("━") and light_u.startswith("─")
    assert heavy_a == "=" * cli.WIDTH and light_a == "-" * cli.WIDTH

    class Cp437:
        encoding = "cp437"

    assert cli._ascii_only(Cp437()) is True

    class Utf8:
        encoding = "utf-8"

    assert cli._ascii_only(Utf8()) is False
    # StringIO has encoding None, so the printed blocks in these tests are ASCII
    assert cli._ascii_only(io.StringIO()) is True


def test_printed_output_never_carries_a_banned_phrase():
    for seed in range(12):
        for extra in ([], ["--unknown", "2"]):
            _, txt = run(["--synthetic", "--seed", str(seed)] + extra)
            for word in BANNED:
                assert word not in txt.lower(), (seed, word)


# --------------------------------------------------------------------------------------
# json and files
# --------------------------------------------------------------------------------------


def test_json_is_parseable_and_carries_the_digest():
    code, txt = run(["--synthetic", "--seed", "7", "--json"])
    payload = json.loads(txt)
    d, pair, _ = cli.synthetic(7)
    assert payload["status"] == CERTIFIED
    assert payload["digest"] == d.digest()
    assert payload["pair"] == list(pair)
    assert payload["exit_code"] == code
    assert payload["convention"].startswith("The scene is a ball")


def test_overlay_and_gif_are_written(tmp_path):
    png, gif = tmp_path / "o.png", tmp_path / "o.gif"
    code, txt = run(
        ["--synthetic", "--seed", "7", "--overlay", str(png), "--gif", str(gif)]
    )
    assert code == 0 and png.exists() and gif.exists()
    assert str(png) in txt and str(gif) in txt

    from PIL import Image

    with Image.open(png) as im:
        assert im.size[0] > 0
    with Image.open(gif) as im:
        assert im.n_frames == 4
