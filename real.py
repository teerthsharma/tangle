"""real.py -- fetch four corpora nobody in this repository made, then run the tool on them.

    .venv/Scripts/python real.py fetch [--dir photos]
    .venv/Scripts/python real.py run   [--dir photos] [--json real.json]

Every number in `tangle` before this file came out of `tangle.synth`, the package's own
renderer.  A renderer cannot falsify the module that reads its output: `synth` draws a
cable as a constant-width stroke on a flat background with the occlusion gap it was told
to draw, and `vision` looks for exactly that.  This file fetches images nobody here made
and runs the same pipeline on them.

Four corpora, none of them a camera or a renderer of ours:

  commons    photographs on Wikimedia Commons, reached from a pinned list of English
             Wikipedia articles about two-rope bends and links.  Free licences only
             (CC0/CC-BY/CC-BY-SA/public domain); the licence, the author and the file
             page travel into `manifest.json` with every download.  No published number
             says what these photographs contain, so the only thing they can measure is
             the refusal rate and its reasons.

  diagram    link diagrams on Wikimedia Commons whose linking number is published
             *outside this repository* -- in the Thistlethwaite link table (L2a1, L4a1,
             L5a1), in the standard fact that the (2, n) torus link has linking number
             n/2 and the Whitehead link has 0, or, for the six images an outside author
             uploaded as "Linking Number 3.svg" and so on, in that author's own filename.
             This is the corpus that can produce a *wrong* answer, which is the only kind
             of failure that matters: a refusal costs the user a re-shoot, a wrong
             certificate tells them to leave a live cable plugged in.

  cabling    photographs of two-coloured cabling -- booster cables, patch leads, speaker
             wire -- reached by a pinned list of Commons search terms.  This is the
             corpus that matches what the README actually claims to do, and it is here
             because the first three do not: a knot photograph is finite rope with two
             visible ends, and a link diagram is ink, while the tool asks for two cables
             running out of the frame.  Search results drift, so the corpus is pinned by
             the sha256 of every file in `manifest.json` once fetched, not by the query.

  knotcross  `tr33hugg3r/knot-crossings` on Hugging Face (Unlicense), the candidate the
             roadmap named.  Its test split is 49 line drawings of *one* closed curve, so
             it is fetched to be refused: it is the out-of-domain control, and a tracer
             that returns a verdict on any of it is reading something that is not there.

Nothing is committed to the repository except this script.  The images and the manifest
stay in `--dir`, which is gitignored.

Self-check (no network):  .venv/Scripts/python real.py --selfcheck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

WIDTH = 78
UA = {"User-Agent": "tanglekit-real/0.1 (https://github.com/teerthsharma/tangle)"}
THUMB = 1024  # the longest side the CLI resizes to anyway; fetching more is bandwidth

# Free licences only.  Matched against the `LicenseShortName` Commons reports, lowercased.
FREE = ("cc0", "cc by", "cc-by", "public domain", "pd-", "no restrictions")

# The pinned reach.  Articles, not categories: a category is edited under us and the
# corpus changes without the commit changing.  Bends and links are asked for because a
# bend is the one knot photograph that conventionally uses two ropes of different colours,
# which is `vision`'s stated precondition.
ARTICLES = [
    "Sheet bend",
    "Reef knot",
    "Carrick bend",
    "Zeppelin bend",
    "Water knot",
    "Double fisherman's knot",
    "Alpine butterfly bend",
    "Ashley's bend",
    "Flemish bend",
    "Blood knot",
    "Hopf link",
    "Link (knot theory)",
    "Prusik knot",
    "Clove hitch",
    "Girth hitch",
    "Square knot (mathematics)",
    "Granny knot",
    "Surgeon's knot",
    "Bowline",
    "Rope",
]

# (Commons file title, name, published |lk| or None, where that number comes from).
#
# `None` means no number outside this repository states the answer, so the image can only
# be scored on whether the tool refused; it can never be scored as right.  The three
# `truth` strings are the only places a number enters this file from outside, and none of
# them is a measurement of ours:
#
#   table     Thistlethwaite/Rolfsen link table, as published by LinkInfo: L2a1 is the
#             Hopf link with linking number +/-1, L4a1 Solomon's knot with +/-2, L5a1 the
#             Whitehead link with 0.
#   standard  the (2, n) torus link has linking number n/2, and the Whitehead link 0 --
#             the second is the fact this repository's own README is built around.
#   filename  the uploader wrote the linking number into the file name.  Six images by one
#             outside author, and the strongest label in the corpus, because it was
#             written by somebody who had never heard of this tool.
DIAGRAMS = [
    ("File:Orthoknot L2a1 Hopf.svg", "L2a1 Hopf", 1, "table"),
    ("File:Orthoknot L4a1 Solomon.svg", "L4a1 Solomon", 2, "table"),
    ("File:Orthoknot L5a1 Whitehead.svg", "L5a1 Whitehead", 0, "table"),
    ("File:Orthoknot L6a3.svg", "L6a3", None, ""),
    ("File:Orthoknot L6a5.svg", "L6a5", None, ""),
    ("File:Orthoknot L6n1.svg", "L6n1", None, ""),
    ("File:Orthoknot L8a13.svg", "L8a13", None, ""),
    ("File:Orthoknot L8a14.svg", "L8a14", None, ""),
    ("File:Orthoknot L8a21.svg", "L8a21", None, ""),
    ("File:Orthoknot L6a4 borromean.svg", "L6a4 Borromean", None, ""),
    ("File:Orthoknot L12a1882 Brunnian.svg", "L12a1882 Brunnian", None, ""),
    ("File:Linking Number -2.svg", "Linking Number -2", 2, "filename"),
    ("File:Linking Number -1.svg", "Linking Number -1", 1, "filename"),
    ("File:Linking Number 0.svg", "Linking Number 0", 0, "filename"),
    ("File:Linking Number 1.svg", "Linking Number 1", 1, "filename"),
    ("File:Linking Number 2.svg", "Linking Number 2", 2, "filename"),
    ("File:Linking Number 3.svg", "Linking Number 3", 3, "filename"),
    ("File:Hopf Link.png", "Hopf link (tubes)", 1, "standard"),
    ("File:Hopf link.svg", "Hopf link (outline)", 1, "standard"),
    ("File:Hopf link rp.png", "Hopf link (oblique)", 1, "standard"),
    ("File:(2,8)-Torus Link.svg", "(2,8) torus link", 4, "standard"),
    ("File:Whitehead-link.svg", "Whitehead link", 0, "standard"),
    ("File:Whitehead-link-horizontal.svg", "Whitehead link (wide)", 0, "standard"),
    ("File:Symmetric Whitehead link.svg", "Whitehead link (symmetric)", 0, "standard"),
    ("File:Labeled Whitehead Link.svg", "Whitehead link (labelled)", 0, "standard"),
    ("File:Polyamory (Whitehead link).svg", "Whitehead link (heart)", 0, "standard"),
    ("File:Unknotting Whitehead link.svg", "Whitehead link (outline)", 0, "standard"),
]

# Pinned queries, not a category: a category is edited under us.
CABLING = [
    "jumper cables",
    "booster cable",
    "patch cable",
    "speaker wire",
    "extension cord",
    "power cord",
    "ethernet cable",
]

HF_REPO = "tr33hugg3r/knot-crossings"


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _cached(path: str, url: str) -> bytes:
    """Download once.  A re-`fetch` re-reads what is already on disk instead of the wire.

    The manifest records the sha256 of the bytes either way, so a corpus fetched in two
    sittings is the same corpus and says so.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            return f.read()
    blob = _get(url)
    with open(path, "wb") as f:
        f.write(blob)
    return blob


def _api(host: str, **kw) -> dict:
    kw.setdefault("format", "json")
    kw.setdefault("action", "query")
    return json.loads(_get(f"https://{host}/w/api.php?" + urllib.parse.urlencode(kw)))


def _safe(title: str) -> str:
    """A Commons title as a filename that survives a Windows checkout."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title.split(":", 1)[-1])[:96]


def _imageinfo(titles: list[str]) -> dict:
    """Commons imageinfo for up to 50 titles, keyed by title."""
    info = _api(
        "commons.wikimedia.org",
        prop="imageinfo",
        titles="|".join(titles),
        iiprop="url|extmetadata|size",
        iiurlwidth=THUMB,
    )
    return {p["title"]: p for p in info["query"]["pages"].values()}


def _row(title: str, page: dict, dest: str, source: str, **extra) -> dict | None:
    ii = (page.get("imageinfo") or [None])[0]
    if not ii:
        return None
    meta = ii.get("extmetadata", {})
    lic = (meta.get("LicenseShortName", {}).get("value") or "").strip()
    if not any(f in lic.lower() for f in FREE):
        return None
    # An SVG is served as a rendered PNG at `iiurlwidth`, which is what a reader sees; the
    # tracer never touches the vector source, so the corpus is raster either way.
    url = ii.get("thumburl") or ii["url"]
    name = _safe(title)
    if name.lower().endswith(".svg"):
        name = name[:-4] + ".png"
    try:
        blob = _cached(os.path.join(dest, name), url)
    except Exception as e:
        print(f"  ! {name}: {e}")
        return None
    return dict(
        file=name,
        source=source,
        url=url,
        page=ii.get("descriptionurl", ""),
        licence=lic,
        author=re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:120],
        sha256=hashlib.sha256(blob).hexdigest(),
        **extra,
    )


def commons(dest: str, limit: int = 200) -> list[dict]:
    """Every free-licensed still image reachable from ARTICLES, at <= THUMB px."""
    titles: list[str] = []
    for a in ARTICLES:
        try:
            page = _api("en.wikipedia.org", prop="images", titles=a, imlimit=60)
        except Exception as e:  # a dead article is not a reason to abandon the corpus
            print(f"  ! {a}: {e}")
            continue
        for p in page["query"]["pages"].values():
            for im in p.get("images", []):
                t = im["title"]
                if t.lower().endswith((".jpg", ".jpeg", ".png")) and t not in titles:
                    titles.append(t)
    print(f"  {len(titles)} candidate files from {len(ARTICLES)} articles")

    rows = []
    for i in range(0, min(len(titles), limit), 20):
        pages = _imageinfo(titles[i : i + 20])
        for t, p in pages.items():
            r = _row(t, p, dest, "commons")
            if r:
                rows.append(r)
    return rows


def diagrams(dest: str) -> list[dict]:
    """The pinned link diagrams, each carrying the number published for it elsewhere."""
    pages = _imageinfo([t for t, _, _, _ in DIAGRAMS])
    rows = []
    for title, name, lk, truth in DIAGRAMS:
        page = pages.get(title)
        if page is None or "missing" in page:
            print(f"  ! {title}: not on Commons under that title")
            continue
        r = _row(title, page, dest, "diagram", link=name, abs_lk=lk, truth=truth)
        if r:
            rows.append(r)
    return rows


def cabling(dest: str, per_query: int = 12) -> list[dict]:
    """Free-licensed photographs of cabling, from a pinned list of Commons searches."""
    titles: list[str] = []
    for q in CABLING:
        try:
            r = _api(
                "commons.wikimedia.org",
                list="search",
                srsearch="filetype:bitmap " + q,
                srnamespace=6,
                srlimit=per_query,
            )
        except Exception as e:
            print(f"  ! {q}: {e}")
            continue
        for hit in r["query"]["search"]:
            if hit["title"] not in titles:
                titles.append(hit["title"])
    print(f"  {len(titles)} candidate files from {len(CABLING)} queries")
    rows = []
    for i in range(0, len(titles), 20):
        for t, p in _imageinfo(titles[i : i + 20]).items():
            r = _row(t, p, dest, "cabling")
            if r:
                rows.append(r)
    return rows


def knotcross(dest: str) -> list[dict]:
    """The 49-image test split of the Hugging Face candidate.  Unlicense."""
    # The repository holds 18,156 files.  `test/` is not the test split: the whole
    # 9,051-image train split is mirrored a second time under `test/train/`, so a
    # `startswith("test/")` filter downloads 9,101 images and takes two and a half hours.
    # The split the dataset card describes is exactly the depth-two paths
    # `test/<crossing number>/<n>.png`, and there are 49 of them.
    info = json.loads(_get(f"https://huggingface.co/api/datasets/{HF_REPO}"))
    paths = [
        s["rfilename"]
        for s in info["siblings"]
        if re.fullmatch(r"test/(\d+)/[^/]+\.png", s["rfilename"])
    ]
    rows = []
    for path in sorted(paths):
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}"
        name = "knotcross_" + path.replace("test/", "", 1).replace("/", "_")
        try:
            blob = _cached(os.path.join(dest, name), url)
        except Exception as ex:
            print(f"  ! {name}: {ex}")
            continue
        rows.append(
            {
                "file": name,
                "source": "knotcross",
                "url": url,
                "page": f"https://huggingface.co/datasets/{HF_REPO}",
                "licence": "Unlicense",
                "author": "Dranowski, Kabkov, Melamud, Tubbenhauer",
                "sha256": hashlib.sha256(blob).hexdigest(),
                # The directory name is the dataset's own label: the minimum crossing
                # number of the knot drawn.  It is ground truth from outside this
                # repository, and it is the truth for a question `tangle` does not ask --
                # one closed curve has no linking number with anything.  Carried anyway,
                # so the control below can say what was refused.
                "crossing_number": int(path.split("/")[1]),
            }
        )
    return rows


def fetch(dest: str) -> dict:
    os.makedirs(dest, exist_ok=True)
    rows = []
    sources = (
        ("commons", commons),
        ("diagram", diagrams),
        ("cabling", cabling),
        ("knotcross", knotcross),
    )
    for name, fn in sources:
        print(f"{name}:")
        got = fn(dest)
        print(f"  {len(got)} free-licensed images")
        rows += got
    man = {"fetched": time.strftime("%Y-%m-%d"), "images": rows}
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(f"  manifest -> {os.path.join(dest, 'manifest.json')}")
    return man


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def one(path: str, coin: random.Random | None = None) -> dict:
    """Trace and certify one image.  Never raises: a crash is a row, not a stop."""
    from dataclasses import replace

    from tangle.certify import CERTIFIED, REFUSED, certify
    from tangle.cli import load_image
    from tangle.vision import trace

    t0 = time.perf_counter()
    row = {"file": os.path.basename(path)}
    try:
        arr, _ = load_image(path)
        d = trace(arr)
        if coin is not None:
            d = replace(
                d,
                crossings=tuple(
                    replace(c, over=coin.choice("ab"), over_conf=1.0, kind="read")
                    for c in d.crossings
                ),
            )
        v = certify(d, 0, 1)
        row.update(status=v.status, reason=v.reason or "", claim=v.claim or "")
        row["crossings"] = len(d.between(0, 1))
        row["closed"] = [c.closed for c in d.cables]
        row["k"] = v.interval.unknown if v.interval is not None else None
        row["lk"] = v.value
    except Exception as e:
        reason = getattr(e, "reason", type(e).__name__)
        row.update(status=REFUSED, reason=reason, claim="", crossings=None, k=None, lk=None)
        row["detail"] = str(e)[:160]
    row["seconds"] = round(time.perf_counter() - t0, 2)
    row["certified"] = row["status"] == CERTIFIED
    return row


def score(row: dict, abs_lk: int | None) -> str:
    """One labelled image against the number published for it: right, wrong, or refused.

    Only `|lk|` is compared.  The global sign is a stated convention of this repository
    (image coordinates run y-down, which negates every crossing sign), so a signed
    comparison would be scoring the convention rather than the tool.
    """
    if abs_lk is None or not row["certified"]:
        return "unlabelled" if abs_lk is None else "no verdict"
    return "right" if abs(row["lk"]) == abs_lk else "WRONG"


def run(dest: str) -> dict:
    man_path = os.path.join(dest, "manifest.json")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    meta = {r["file"]: r for r in man["images"]}

    rows, coin_rows = [], []
    for r in man["images"]:
        p = os.path.join(dest, r["file"])
        if not os.path.exists(p):
            continue
        row = one(p)
        row["source"] = r["source"]
        row["link"] = r.get("link", "")
        row["abs_lk"] = r.get("abs_lk")
        row["truth"] = r.get("truth", "")
        row["score"] = score(row, r.get("abs_lk"))
        rows.append(row)
        c = one(p, coin=random.Random(20260905))
        c["source"] = row["source"]
        c["score"] = score(c, r.get("abs_lk"))
        coin_rows.append(c)
    return {"rows": rows, "coin": coin_rows, "dir": dest, "images": len(meta)}


def control(n: int = 20) -> dict:
    """The same `one()`, on `n` synthetic piles written to PNG and read back.

    Without this the headline is unreadable: zero certificates on a real corpus is a
    statement about the corpus only if the identical code path certifies something.  The
    scenes go out through `Image.save` and come back through `load_image`, so the control
    carries the same PNG round trip, the same EXIF and alpha handling and the same 1024 px
    resize as every real image above.
    """
    import tempfile

    from tangle import synth

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for seed in range(5, 5 + n):
            path = os.path.join(td, f"pile{seed}.png")
            synth.save(path, synth.pile(seed), seed=seed)
            row = one(path)
            row["source"] = "synthetic control"
            row["score"] = "unlabelled"
            rows.append(row)
    return {"rows": rows, "coin": [], "dir": "<synthetic>", "images": n}


def table(res: dict, ascii_only: bool = False) -> str:
    from tangle.certify import CERTIFIED, NOT_CERTIFIED, REFUSED

    heavy, light = ("=" * WIDTH, "-" * WIDTH) if ascii_only else ("━" * WIDTH, "─" * WIDTH)
    rows = res["rows"]
    synthetic = {r["source"] for r in rows} == {"synthetic control"}
    banner = (
        "  the control: tangle.synth through the same PNG round trip"
        if synthetic
        else "  real images -- no pixel below came out of tangle.synth"
    )
    L = [heavy, banner, heavy]
    for source in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == source]
        st = Counter(r["status"] for r in sub)
        L.append(f"  {source}   n = {len(sub)}")
        for s in (CERTIFIED, NOT_CERTIFIED, REFUSED):
            n = st.get(s, 0)
            L.append(f"    {s:<24} {n:>4}   {100 * n / max(len(sub), 1):5.1f}%")
        why = Counter(r["reason"] for r in sub if r["status"] == REFUSED)
        for reason, n in why.most_common():
            L.append(f"      {reason:<28} {n:>4}")
        L.append(light)

    labelled = [r for r in rows if r.get("abs_lk") is not None]
    if labelled:
        L.append("  against numbers published outside this repository")
        for r in sorted(labelled, key=lambda r: (r["truth"], r["link"])):
            got = "-" if r["lk"] is None else str(abs(r["lk"]))
            L.append(
                f"    {r['link']:<28} |lk| = {r['abs_lk']}  ({r['truth']:<8}) "
                f"got {got:<4} {r['score']}"
            )
        sc = Counter(r["score"] for r in labelled)
        L.append(f"    right {sc['right']}   WRONG {sc['WRONG']}   no verdict {sc['no verdict']}")
        L.append(light)

    cert = [r for r in rows if r["certified"]]
    coin_cert = [r for r in res["coin"] if r["certified"]]
    L += [
        f"  certified verdicts               {len(cert):>4}   of {len(rows)}",
        f"  wrong certificates               "
        f"{sum(1 for r in labelled if r['score'] == 'WRONG'):>4}   of {len(labelled)} labelled",
        "  same images, over/under replaced by a coin flip",
        f"    certified verdicts             {len(coin_cert):>4}   control",
        heavy,
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def _demo() -> None:
    import tempfile

    from tangle.certify import CERTIFIED, NOT_CERTIFIED, REFUSED

    assert _safe("File:Sheet bend.jpg") == "Sheet_bend.jpg"
    assert not any(f in "All rights reserved".lower() for f in FREE)
    assert any(f in "CC BY-SA 4.0".lower() for f in FREE)
    assert len({t for t, _, _, _ in DIAGRAMS}) == len(DIAGRAMS), "a diagram is pinned twice"
    assert len(set(CABLING)) == len(CABLING), "a cabling query is pinned twice"

    # the label is compared on |lk| and a mismatch is loud
    assert score({"certified": True, "lk": -2}, 2) == "right"
    assert score({"certified": True, "lk": 1}, 2) == "WRONG"
    assert score({"certified": False, "lk": None}, 2) == "no verdict"
    assert score({"certified": True, "lk": 1}, None) == "unlabelled"

    # an unreadable file is a row with a reason, not a traceback out of `run`
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "notanimage.png")
        with open(p, "wb") as f:
            f.write(b"not a png")
        r = one(p)
        assert r["status"] == REFUSED and r["reason"] and r["lk"] is None, r

    # one pile through the PNG round trip: the control has to reach a traced diagram, not
    # necessarily a certificate -- whether a given scene certifies is the thing being
    # measured, and pinning it here would make the self-check a second copy of the bench.
    r = control(1)["rows"][0]
    assert r["crossings"] is not None and r["status"] in (CERTIFIED, NOT_CERTIFIED, REFUSED), r

    res = {
        "rows": [dict(source="s", status=REFUSED, reason="X", certified=False, abs_lk=None)],
        "coin": [],
    }
    assert "X" in table(res, ascii_only=True)
    assert CERTIFIED in table(res, ascii_only=True)
    print("ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="real.py", description=__doc__.splitlines()[0])
    p.add_argument("cmd", nargs="?", choices=["fetch", "run", "control"])
    p.add_argument("--dir", default="photos")
    p.add_argument("--json", default=None, metavar="real.json")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        _demo()
        return 0
    if a.cmd == "fetch":
        fetch(a.dir)
        return 0
    if a.cmd in ("run", "control"):
        res = control() if a.cmd == "control" else run(a.dir)
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "━".encode(enc)
            ascii_only = False
        except (UnicodeEncodeError, LookupError):
            ascii_only = True
        print(table(res, ascii_only))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"  rows -> {a.json}")
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
