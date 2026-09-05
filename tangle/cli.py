"""tangle/cli.py -- `python -m tangle <image>` and `python -m tangle --synthetic --seed N`.

    python -m tangle photo.jpg [--pair 0,1] [--overlay out.png] [--gif out.gif] [--json]
    python -m tangle --synthetic --seed 7 [--letters 6] [--strands 2] [--unknown 2]

    exit 0 CERTIFIED   1 NOT CERTIFIED   2 REFUSED   3 INPUT

Every block this file prints is *generated*: the number, the interval, the reason, the
advice and the convention all come off the `Verdict`, and the only strings the CLI owns are
the field labels.  No example block is hand-typed anywhere in this repository, because a
hand-typed block is how the previous draft of the spec shipped an interval whose width
contradicted its own `k`.

The `--synthetic` corpus lives here rather than in a `corpus.py` the spec deletes.  It is a
closed braid word with a committed seed, so `lk` is the signed exponent sum over the
inter-component letters, halved -- known before anything is drawn and not a fit to anything
this package computes.  `bench.py` imports `synthetic` and `blur` from here; there is one
definition of the corpus and both callers use it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from typing import Callable, Sequence

from .certify import CERTIFIED, EXIT, REFUSED, Verdict, certify
from .diagram import Diagram

EXIT_INPUT = 3  # the fourth code from the spec; EXIT in certify.py owns the other three.

WIDTH = 78


# --------------------------------------------------------------------------------------
# the synthetic corpus: braid word in, known lk out
# --------------------------------------------------------------------------------------


def synthetic(
    seed: int, letters: int = 6, strands: int = 2
) -> tuple[Diagram, tuple[int, int], list[int]]:
    """A seeded closed-braid diagram with at least two components and one shared crossing.

    Returns the diagram (fully read), the pair to certify, and the braid word.  The word is
    returned because it is the *control*: on two strands `lk = (signed exponent sum) / 2`
    in closed form, so the truth the bench scores against is not this package's own
    arithmetic.  `bench.py` asserts the two agree on every corpus entry.
    """
    if letters < 2 or letters % 2:
        raise ValueError("an odd letter count on two strands closes to one component")
    rng = random.Random(seed)
    alphabet = [s * i for i in range(1, strands) for s in (1, -1)]
    for _ in range(200):
        word = [rng.choice(alphabet) for _ in range(letters)]
        d = Diagram.from_braid(word, strands=strands)
        if len(d.cables) < 2:
            continue
        for i in range(len(d.cables)):
            for j in range(i + 1, len(d.cables)):
                if d.between(i, j) and d.validate() is None:
                    return d, (i, j), word
    raise RuntimeError(f"seed {seed} produced no two-component braid in 200 draws")


def blur(d: Diagram, i: int, j: int, k: int, rng: random.Random) -> Diagram:
    """Set `k` of the inter-component crossings of (i, j) to UNKNOWN.

    This is the *only* corruption the interval theorem is proved against, which is exactly
    why "0 wrong certified" on this corpus measures the implementation and not the design.
    The number that could have come out badly is the coverage gap, and that is what
    `bench.py` reports.
    """
    cs = [c.id for c in d.between(i, j)]
    if k > len(cs):
        raise ValueError(f"cannot blur {k} of {len(cs)} inter-component crossings")
    return d.resolve({cid: None for cid in rng.sample(cs, k)}) if k else d


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------


def _ascii_only(stream=None) -> bool:
    enc = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        "━│".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return True
    return False


def rules(ascii_only: bool) -> tuple[str, str]:
    """(heavy, light) horizontal rules.  ASCII when stdout cannot carry box drawing."""
    return ("=" * WIDTH, "-" * WIDTH) if ascii_only else ("━" * WIDTH, "─" * WIDTH)


def _wrap(label: str, value: str) -> list[str]:
    body = textwrap.wrap(value, WIDTH - 15) or [""]
    return [f"  {label:<11} {body[0]}"] + [f"  {'':<11} {b}" for b in body[1:]]


def block(v: Verdict, d: Diagram | None = None, *, ascii_only: bool = False) -> str:
    """The verdict block, inside rules.  Generated from the Verdict, never hand-typed.

    `d` is None when the tracer refused before a diagram existed; there is then no source
    and no digest to print, and printing a placeholder for either would be a lie.
    """
    heavy, light = rules(ascii_only)
    num = ""
    if v.interval is not None and v.interval.unknown:
        num = f"lk in {v.interval}   k = {v.interval.unknown}"
    elif v.value is not None:
        num = f"lk = {v.value}"
    head = f"{v.status}  {v.claim or v.reason}".rstrip()
    rows = [heavy, f"  {head}{num.rjust(max(2, WIDTH - 4 - len(head)))}".rstrip(), heavy]
    if v.witness:
        rows += _wrap("witness", v.witness)
    if v.interval is not None:
        rows += _wrap("interval", f"[{v.interval.lo}, {v.interval.hi}]   S = {v.interval.known_sum}")
    if v.look_at:
        rows += _wrap("look at", "crossing " + ", ".join(str(x) for x in v.look_at))
    if v.advice:
        rows += _wrap("advice", v.advice)
    if d is not None:
        rows += _wrap("source", d.provenance or "(none)")
        rows += _wrap("digest", d.digest())
    rows += [light]
    rows += _wrap("convention", v.convention)
    rows += [heavy]
    return "\n".join(rows)


# --------------------------------------------------------------------------------------
# the imaging layer, if it is installed
# --------------------------------------------------------------------------------------

MAX_SIDE = 1024  # spec stage 1: longest side to 1024, LANCZOS


def vision_entry() -> Callable[..., object] | None:
    """`tangle.vision.trace`, or None when the imaging layer is not installed.

    The certified layer is usable with no pixels at all, so an absent tracer is a named
    input refusal and not an ImportError at the user.
    """
    try:
        from . import vision
    except Exception:
        return None
    return getattr(vision, "trace", None)


def load_image(path: str):
    """Photograph -> (RGB array for the tracer, PIL image for the overlay).

    **EXIF transpose first.**  Orientation moves the frame, and the frame is what pins the
    four ends: a diagram traced from an untransposed portrait photo has its exit points on
    the wrong sides.  Then the longest side goes to 1024, so the overlay's pixels and the
    diagram's coordinates are the same pixels.

    **Transparency is composited onto white, not dropped.**  `convert("RGB")` on an RGBA
    file discards the alpha channel and keeps whatever colour happens to sit under it,
    which for the Wikimedia link renderings is black; the tracer then measures a background
    nobody looking at the file sees, and the border ring reports a spread that belongs to
    the file format rather than the scene.
    """
    import numpy as np
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img.convert("RGBA"))
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return np.asarray(img), img


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tangle",
        description="Photograph two cables and get an integer with a proof, or a refusal "
        "that names the crossing to re-shoot.",
    )
    p.add_argument("image", nargs="?", help="the photograph to certify")
    p.add_argument("--synthetic", action="store_true", help="certify a seeded braid instead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--letters", type=int, default=6, help="braid word length (--synthetic)")
    p.add_argument("--strands", type=int, default=2, help="braid strands (--synthetic)")
    p.add_argument("--unknown", type=int, default=0, help="crossings to blur (--synthetic)")
    p.add_argument("--pair", default=None, help="which two cables, as A,B")
    p.add_argument("--overlay", default=None, metavar="OUT.png")
    p.add_argument("--gif", default=None, metavar="OUT.gif")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, out=None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    out = out or sys.stdout

    if bool(args.image) == bool(args.synthetic):
        print("give one image, or --synthetic --seed N", file=out)
        return EXIT_INPUT

    pair: tuple[int, int] | None = None
    if args.pair:
        try:
            a, b = (int(x) for x in args.pair.split(","))
            pair = (a, b)
        except ValueError:
            print(f"--pair wants two integers, got {args.pair!r}", file=out)
            return EXIT_INPUT

    image = None
    if args.synthetic:
        d, auto, _word = synthetic(args.seed, letters=args.letters, strands=args.strands)
        pair = pair or auto
        if args.unknown:
            try:
                d = blur(d, pair[0], pair[1], args.unknown, random.Random(args.seed))
            except ValueError as e:
                print(str(e), file=out)
                return EXIT_INPUT
    else:
        entry = vision_entry()
        if entry is None:
            print(
                "the imaging layer (tangle.vision) is not installed, so a photograph "
                "cannot be traced. `--synthetic --seed N` certifies a braid with no camera.",
                file=out,
            )
            return EXIT_INPUT
        try:
            arr, image = load_image(args.image)
        except (FileNotFoundError, OSError) as e:
            print(f"cannot read {args.image}: {e}", file=out)
            return EXIT_INPUT
        try:
            d = entry(arr)
        except Exception as e:  # TraceRefused, and anything the tracer did not foresee
            reason = getattr(e, "reason", type(e).__name__)
            v = Verdict(status=REFUSED, reason=reason, advice=str(e), exit_code=EXIT[REFUSED])
            print(block(v, None, ascii_only=_ascii_only(out)), file=out)
            return v.exit_code
        pair = pair or (0, 1)

    v = certify(d, *pair)
    if args.json:
        payload = v.json()
        payload["digest"] = d.digest()
        payload["provenance"] = d.provenance
        payload["pair"] = list(pair)
        print(json.dumps(payload, indent=2), file=out)
    else:
        print(block(v, d, ascii_only=_ascii_only(out)), file=out)

    if args.overlay or args.gif:
        from . import viz

        if args.overlay:
            viz.render(d, v, image, pair=pair).save(args.overlay)
            print(f"overlay -> {args.overlay}", file=out)
        if args.gif:
            viz.save_gif(args.gif, viz.frames(d, v, image, pair=pair))
            print(f"gif -> {args.gif}", file=out)

    return v.exit_code


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    import io

    d, pair, word = synthetic(7)
    assert len(d.cables) >= 2 and d.between(*pair)
    d2, pair2, word2 = synthetic(7)
    assert d.digest() == d2.digest() and pair == pair2 and word == word2, "corpus not reproducible"
    # the closed-form control: lk on two strands is half the signed exponent sum
    from .certify import lk_interval

    assert abs(lk_interval(d, *pair).exact) == abs(sum(1 if w > 0 else -1 for w in word)) // 2

    v = certify(d, *pair)
    txt = block(v, d, ascii_only=True)
    assert "=" * WIDTH in txt and d.digest() in txt and v.convention[:20] in txt
    assert max(len(line) for line in txt.splitlines()) <= WIDTH + 1

    buf = io.StringIO()
    code = main(["--synthetic", "--seed", "7"], out=buf)
    assert code in EXIT.values(), code
    assert v.status in buf.getvalue()

    buf = io.StringIO()
    assert main(["--synthetic", "--seed", "7", "--json"], out=buf) == code
    assert json.loads(buf.getvalue())["status"] == v.status

    buf = io.StringIO()
    assert main([], out=buf) == EXIT_INPUT
    print("ok")


if __name__ == "__main__":  # pragma: no cover
    _demo()
