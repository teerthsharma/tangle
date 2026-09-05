r"""Every math block in the Markdown must survive GitHub's renderer.

GitHub runs a backslash-escape pass over Markdown *before* the math renderer
sees the text, and it runs KaTeX with a macro blocklist. Two failure modes
follow, both measured against `POST /markdown mode=gfm` on 2026-09-05:

  1. Escape loss.  Sent  ``$$ \mathrm{lk}(A,B)\;=\;\sum_{c\,\in\,C} $$``,
     GitHub returned ``\mathrm{lk}(A,B);=;\sum_{c,\in,C}`` -- every
     ``\;`` and ``\,`` consumed, so KaTeX is handed broken source and prints
     "Missing or unrecognized delimiter for \Big". The same LaTeX inside a
     ```math fence came back byte-identical: fenced content is exempt from
     the escape pass. So is the inline form ``$`...`$``.
  2. Macro blocklist.  ``\operatorname`` passes the API untouched and is then
     rejected in the browser with "The following macros are not allowed".
     No round-trip check can see this one; only a substring blocklist can.

The rules below (MD-MATH-01..06) are those two findings made mechanical.
MD-MATH-07 -- the round-trip against GitHub's own renderer -- needs the
network and is opt-in via TANGLE_GH_RENDER=1.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "RESULTS.md"]

# MD-MATH-04. GitHub tests `textContent.includes('\\' + macro)`, so the match is
# a plain substring, not a word boundary: `\operatorname*` and `\phantomx` both
# trip it. `\vphantom` is blocked; `\smash` is not.
BLOCKED = (
    "DeclareMathOperator",
    "DeclarePairedDelimiters",
    "renewtagform",
    "newtagform",
    "colorbox",
    "fcolorbox",
    "hphantom",
    "vphantom",
    "phantom",
    "operatorname",
    "Newextarrow",
    "definecolor",
    "mathchoice",
    "unicode",
    "mmlToken",
)
RE_BLOCKED = re.compile(r"\\(" + "|".join(BLOCKED) + r")")

RE_LONE_DOLLARS = re.compile(r"^[ \t]*\$\$[ \t]*$")
RE_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*(\S*)")
RE_TICK_MATH = re.compile(r"\$`(.+?)`\$")
RE_CODE_SPAN = re.compile(r"`+[^`\n]*`+")
RE_DISPLAY_1L = re.compile(r"\$\$(.+?)\$\$")
RE_INLINE = re.compile(r"\$([^$\n]+?)\$")

# MD-MATH-03: outside a fence, a backslash may only be followed by a letter or
# a space. `\ ` (backslash-space) is measured to survive.
RE_ESCAPED_PUNCT = re.compile(r"\\[^A-Za-z ]")
# MD-MATH-05: constructs with an alphabetic-only spelling must use it.
RE_UNPORTABLE = re.compile(r"\\(?:Big|big|Bigg|bigg)?[lr]?\\?[{}|]|\\[;,!:]")


class Region:
    """One math region: where it came from, and whether it is fenced."""

    def __init__(self, doc: str, line: int, kind: str, text: str) -> None:
        self.doc, self.line, self.kind, self.text = doc, line, kind, text

    @property
    def fenced(self) -> bool:
        return self.kind == "fence"

    @property
    def display(self) -> bool:
        return self.kind in ("fence", "dollars_display")

    def __str__(self) -> str:
        flat = " ".join(self.text.split())
        return f"{self.doc}:{self.line} [{self.kind}] {flat[:90]}"


def _regions(doc: str) -> tuple[list[Region], list[Region]]:
    """Return (math regions, lone-`$$`-line offenders) for one Markdown file."""
    lines = (ROOT / doc).read_text(encoding="utf-8").splitlines()
    math: list[Region] = []
    bad_fence: list[Region] = []
    close: str | None = None  # closing marker of the code fence we are inside
    buf: list[str] = []
    start = 0
    in_dollars = False

    for n, line in enumerate(lines, 1):
        if close is not None:  # inside a ``` block
            if line.strip().startswith(close):
                if buf or start:  # a ```math block, collected
                    math.append(Region(doc, start, "fence", "\n".join(buf)))
                close, buf, start = None, [], 0
            elif start:
                buf.append(line)
            continue

        m = RE_FENCE.match(line)
        if m and not in_dollars:
            close = m.group(1)[:3]
            if m.group(2).lower() == "math":
                start, buf = n, []
            continue

        if in_dollars:
            if RE_LONE_DOLLARS.match(line):
                math.append(Region(doc, start, "dollars_display", "\n".join(buf)))
                in_dollars = False
            else:
                buf.append(line)
            continue

        if RE_LONE_DOLLARS.match(line):
            bad_fence.append(Region(doc, n, "dollars_display", ""))
            in_dollars, start, buf = True, n, []
            continue

        rest = line
        for m in RE_TICK_MATH.finditer(line):
            math.append(Region(doc, n, "inline_tick", m.group(1)))
        rest = RE_TICK_MATH.sub(lambda m: " " * len(m.group(0)), rest)
        rest = RE_CODE_SPAN.sub(lambda m: " " * len(m.group(0)), rest)
        for m in RE_DISPLAY_1L.finditer(rest):
            math.append(Region(doc, n, "dollars_display", m.group(1)))
        rest = RE_DISPLAY_1L.sub(lambda m: " " * len(m.group(0)), rest)
        for m in RE_INLINE.finditer(rest):
            math.append(Region(doc, n, "dollars_inline", m.group(1)))

    return math, bad_fence


ALL: list[Region] = []
BAD_FENCES: list[Region] = []
for _doc in DOCS:
    _m, _b = _regions(_doc)
    ALL += _m
    BAD_FENCES += _b


def _fail(rule: str, why: str, hits: list[Region]) -> None:
    assert not hits, f"{rule}: {why}\n" + "\n".join(f"  {h}" for h in hits)


def test_md_math_01_no_dollar_fences() -> None:
    """Display math uses a ```math fence; `$$` content loses its escapes."""
    _fail("MD-MATH-01", "display math delimited by `$$` (use a ```math fence)", BAD_FENCES)


def test_md_math_02_inline_uses_backticks() -> None:
    """Inline math with an escape must be ``$`...`$``, not bare `$...$`."""
    hits = [
        r for r in ALL
        if r.kind == "dollars_inline" and RE_ESCAPED_PUNCT.search(r.text)
    ]
    _fail("MD-MATH-02", "bare `$...$` inline math carrying a backslash escape", hits)


def test_md_math_03_no_escaped_punctuation_outside_a_fence() -> None:
    r"""Outside a fence, `\;` `\,` `\{` `\}` `\\` and friends are eaten."""
    hits = [r for r in ALL if not r.fenced and RE_ESCAPED_PUNCT.search(r.text)]
    _fail("MD-MATH-03", r"backslash followed by non-letter, non-space", hits)


def test_md_math_04_no_blocked_macros() -> None:
    """GitHub's KaTeX rejects these 15 macros, fenced or not."""
    hits = [r for r in ALL if RE_BLOCKED.search(r.text)]
    _fail("MD-MATH-04", "macro on GitHub's KaTeX blocklist", hits)


def test_md_math_05_portable_spellings() -> None:
    r"""Mandatory outside a fence: `\lbrace` not `\{`, `\quad` not `\;`."""
    hits = [r for r in ALL if not r.fenced and RE_UNPORTABLE.search(r.text)]
    _fail("MD-MATH-05", "non-alphabetic spelling outside a ```math fence", hits)


def test_md_math_06_row_breaks_only_in_a_fence() -> None:
    r"""`\\` as a row separator survives only inside a ```math fence."""
    hits = [r for r in ALL if not r.fenced and "\\\\" in r.text]
    _fail("MD-MATH-06", r"`\\` row break outside a ```math fence", hits)


def test_every_math_block_was_found() -> None:
    """Guard the parser itself: the README's two derivations must be seen."""
    assert len(ALL) >= 2, f"parser found only {len(ALL)} math regions"


# --- KaTeX, when this machine has node ---------------------------------------

KATEX = ROOT / ".donotcommit" / "katex" / "node_modules" / "katex"
_missing = (
    "node not on PATH" if shutil.which("node") is None
    else f"katex not installed at {KATEX}" if not KATEX.is_dir()
    else ""
)


@pytest.mark.skipif(bool(_missing), reason=_missing or "ok")
def test_katex_parses_every_block_in_strict_mode() -> None:
    """Parse each region with KaTeX itself, strict mode, errors fatal."""
    script = """
const katex = require(process.argv[1]);
const blocks = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const bad = [];
for (const b of blocks) {
  try {
    katex.renderToString(b.text, {displayMode: b.display, strict: 'error',
                                  throwOnError: true});
  } catch (e) { bad.push(b.label + '  ->  ' + e.message); }
}
console.log(JSON.stringify(bad));
"""
    payload = ROOT / ".donotcommit" / "katex_blocks.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text(
        json.dumps([{"text": r.text, "display": r.display, "label": str(r)} for r in ALL]),
        encoding="utf-8",
    )
    out = subprocess.run(
        ["node", "-e", script, str(KATEX).replace("\\", "/"), str(payload)],
        capture_output=True, text=True, encoding='utf-8', timeout=120,
    )
    assert out.returncode == 0, out.stderr
    bad = json.loads(out.stdout.strip().splitlines()[-1])
    assert not bad, "KaTeX strict mode rejected:\n" + "\n".join(f"  {b}" for b in bad)


# --- MD-MATH-07: the only check that tests the real pipeline ------------------


@pytest.mark.skipif(
    os.environ.get("TANGLE_GH_RENDER") != "1" or shutil.which("gh") is None,
    reason="opt-in: needs the network and an authenticated gh; set TANGLE_GH_RENDER=1",
)
@pytest.mark.parametrize("doc", DOCS)
def test_md_math_07_round_trips_through_github(doc: str) -> None:
    """POST the file to GitHub and diff what the math renderer was handed.

    The only check here that tests the real pipeline rather than a model of it.
    Measured 2026-09-05: both README blocks came back byte-identical.
    """
    out = subprocess.run(
        ["gh", "api", "markdown", "-X", "POST", "-F", f"text=@{ROOT / doc}",
         "-f", "mode=gfm"],
        capture_output=True, text=True, encoding='utf-8', timeout=120,
    )
    assert out.returncode == 0, out.stderr
    got = [
        m.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip("$\n")
        for m in re.findall(r"<math-renderer[^>]*>(.*?)</math-renderer>", out.stdout, re.S)
    ]
    want = [r.text.strip() for r in ALL if r.doc == doc]
    assert len(got) == len(want), f"{doc}: sent {len(want)} blocks, {len(got)} came back"
    for g, w in zip(got, want):
        assert g.strip() == w, (
            f"{doc}: GitHub altered the LaTeX\n  sent {w!r}\n  got  {g!r}"
        )
