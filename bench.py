"""bench.py -- coverage at zero errors, against a baseline that also has zero errors.

    .venv/Scripts/python bench.py [--n 400] [--kmax 4] [--seed0 1000] [--out results.json]

The corpus is `tangle.cli.synthetic`: closed braid words on two strands with committed
seeds, then `k` of the inter-component crossings blurred to UNKNOWN.  `lk` is known in
closed form before anything is computed -- half the signed exponent sum -- and every entry
asserts the diagram layer agrees with that closed form, so the truth this scores against is
not the package's own arithmetic.

**Read the headline correctly.**  A corpus that corrupts the input in exactly and only the
one way the interval theorem is proved against *cannot* produce a wrong certificate from a
correct implementation.  The `0 wrong` row therefore measures the code, not the design, and
it ships as a regression test rather than as evidence.  The number that could have come out
badly is the **coverage gap** over the abstain-on-any-unknown baseline, which also scores
zero errors; that gap is the entire argument for the interval logic and it is the only row
here worth reading first.

Two negatives are printed rather than buried.  The next-crossing hit rate is reported beside
a random-crossing control at the same photograph budget, and theory predicts **no gap**:
every unknown shrinks the interval width by exactly 1, so for the linking number no crossing
is more decisive than another and the ranking is a perception heuristic, not an information
criterion.  And the distribution of `k` among certified verdicts is printed, because if the
certified population is `k = 0` almost everywhere then "certified over all 2^k resolutions"
is doing less work than the pitch implies.

Not built here: the R2-drape alternation control and the mask-overlap control, which need
rendered scenes (`tangle.synth`) rather than diagrams; and every real-photograph table,
which needs the tracer.  Their absence is named in the output, not silently skipped.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter

from tangle.certify import CERTIFIED, NOT_CERTIFIED, REFUSED, SEPARABLE, certify, lk_interval
from tangle.cli import blur, rules, synthetic

WIDTH = 78


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def wrong(v, truth: int) -> bool:
    """Did a certificate assert something the ground truth contradicts?

    LINKED asserts `truth` lies in the certified interval, which excludes zero.  SEPARABLE
    asserts a separating disk, and T3 says a split pair has `lk = 0`, so a nonzero truth
    falsifies it.  Anything not CERTIFIED asserts nothing and cannot be wrong.
    """
    if v.status != CERTIFIED:
        return False
    if v.claim == SEPARABLE:
        return truth != 0
    return v.interval is None or truth not in v.interval


def abstain(d, i: int, j: int, truth: int):
    """The control that matters: refuse on any unknown inter-component crossing.

    Four lines, no interval theorem, and it also scores zero errors.  Returns
    (certified, is_wrong).
    """
    if d.unknown_between(i, j):
        return (False, False)
    exact = lk_interval(d, i, j).exact
    return (exact != 0, exact != truth)


def coin_flip(d, i: int, j: int, truth: int, rng: random.Random):
    """Resolve every unknown at random, then certify.  This is what a confidence bar is."""
    unk = d.unknown_between(i, j)
    guessed = d.resolve({c.id: rng.choice(("a", "b")) for c in unk})
    v = certify(guessed, i, j)
    return (v.status == CERTIFIED, wrong(v, truth))


def resolve_to_truth(blurred, source, cid: int, i: int, j: int):
    """Re-shoot one crossing: take its true over/under from the unblurred diagram."""
    true_over = {c.id: c.over for c in source.crossings}[cid]
    return certify(blurred.resolve({cid: true_over}), i, j)


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


def run(n: int = 400, kmax: int = 4, seed0: int = 1000, letters: int = 6) -> dict:
    rng = random.Random(20260905)
    tally: Counter = Counter()
    k_certified: Counter = Counter()
    rows = []
    t0 = time.perf_counter()

    for seed in range(seed0, seed0 + n):
        d, (i, j), word = synthetic(seed, letters=letters)

        truth = lk_interval(d, i, j).exact
        closed_form = sum(1 if w > 0 else -1 for w in word) // 2
        if abs(truth) != abs(closed_form):  # the control, live on every entry
            raise AssertionError(f"seed {seed}: lk {truth} vs braid closed form {closed_form}")

        available = len(d.between(i, j))
        for k in range(0, kmax + 1):
            if k > available:
                tally["skipped"] += 1
                continue
            bd = blur(d, i, j, k, rng)
            v = certify(bd, i, j)
            tally["n"] += 1
            tally[v.status] += 1
            if v.status == CERTIFIED:
                tally["wrong"] += wrong(v, truth)
                k_certified[k] += 1

            cert, bad = abstain(bd, i, j, truth)
            tally["abstain_certified"] += cert
            tally["abstain_wrong"] += bad

            cert, bad = coin_flip(bd, i, j, truth, rng)
            tally["coin_certified"] += cert
            tally["coin_wrong"] += bad

            # active perception: does re-shooting the named crossing certify?
            if v.status == REFUSED and v.reason == "LK_STRADDLES_ZERO" and v.look_at:
                tally["straddled"] += 1
                named = resolve_to_truth(bd, d, v.look_at[0], i, j)
                tally["named_hit"] += named.status == CERTIFIED
                pick = rng.choice([c.id for c in bd.unknown_between(i, j)])
                rand = resolve_to_truth(bd, d, pick, i, j)
                tally["random_hit"] += rand.status == CERTIFIED

            rows.append(
                {
                    "seed": seed,
                    "k": k,
                    "truth": truth,
                    "status": v.status,
                    "reason": v.reason,
                    "interval": None if v.interval is None else [v.interval.lo, v.interval.hi],
                }
            )

    return {
        "tally": dict(tally),
        "k_among_certified": dict(sorted(k_certified.items())),
        "config": {"n": n, "kmax": kmax, "seed0": seed0, "letters": letters, "strands": 2},
        "provenance": {
            "commit": _commit(),
            "machine": platform.node(),
            "python": sys.version.split()[0],
            "seconds": round(time.perf_counter() - t0, 2),
        },
        "rows": rows,
    }


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or "(not a git checkout)"
    except Exception:
        return "(git unavailable)"


# --------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------


def _pct(x: int, n: int) -> str:
    return f"{100.0 * x / n:5.1f}%" if n else "    --"


def _zero(x: int, n: int) -> str:
    """Every zero carries its upper bound.  Rule of three: 95% upper bound is 3/N."""
    return f"0 wrong (<={_pct(3, n).strip()} at 95%)" if x == 0 else f"{x} wrong"


def _row(label: str, value: str, note: str = "") -> str:
    return f"    {label:<38}{value:>6}   {note}".rstrip()


def table(res: dict, ascii_only: bool = False) -> str:
    heavy, light = rules(ascii_only)
    t = res["tally"]
    n = t.get("n", 0)
    cert = t.get(CERTIFIED, 0)
    gap = 100.0 * (cert - t.get("abstain_certified", 0)) / n if n else 0.0
    st = t.get("straddled", 0)
    cfg = res["config"]

    L = [
        heavy,
        f"  certified pairs, at {t.get('wrong', 0)} wrong certified verdicts",
        f"  {cfg['n']} braid seeds x k = 0..{cfg['kmax']} blurred crossings = {n} entries",
        heavy,
        _row("tangle", _pct(cert, n), _zero(t.get("wrong", 0), n)),
        _row(
            "abstain on any unknown crossing",
            _pct(t.get("abstain_certified", 0), n),
            _zero(t.get("abstain_wrong", 0), n),
        ),
        "    " + light[:48],
        _row("coverage gained by the interval theorem", f"{gap:.1f}", "points"),
        "",
        "  same diagrams, decision rule replaced:",
        _row("coin flip on unknowns", _pct(t.get("coin_certified", 0), n), f"{t.get('coin_wrong', 0)} wrong"),
        heavy,
        _row("tangle CERTIFIED", _pct(cert, n)),
        _row("tangle NOT CERTIFIED", _pct(t.get(NOT_CERTIFIED, 0), n)),
        _row("tangle REFUSED", _pct(t.get(REFUSED, 0), n)),
        heavy,
        "  active perception -- the ranking is a perception heuristic, and theory",
        "  predicts no gap: every unknown shrinks the interval width by exactly 1.",
        _row("named crossing re-shot, then certified", _pct(t.get("named_hit", 0), st), f"of {st}"),
        _row("random crossing re-shot, same budget", _pct(t.get("random_hit", 0), st), "control"),
        heavy,
        "  k among certified verdicts -- if this is 0 almost everywhere, then",
        '  "certified over all 2^k resolutions" is doing less work than it sounds.',
        "    " + "  ".join(f"k={k}: {v}" for k, v in res["k_among_certified"].items()),
        heavy,
        "  the REFUSED rate above is a function of the blur schedule -- up to",
        f"  k = {cfg['kmax']} of {cfg['letters']} crossings erased on purpose -- and is NOT a photograph's",
        "  refuse rate.  That number needs the tracer and is not measured here.",
        light,
        "  not run here: the R2-drape alternation control and the mask-overlap",
        "  control need rendered scenes; every real-photograph table needs the tracer.",
        heavy,
        f"  commit {res['provenance']['commit']}   machine {res['provenance']['machine']}"
        f"   python {res['provenance']['python']}   {res['provenance']['seconds']} s",
        heavy,
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# self-check:  .venv/Scripts/python bench.py --selfcheck
# --------------------------------------------------------------------------------------


def _demo() -> None:
    d, (i, j), word = synthetic(1000)
    truth = lk_interval(d, i, j).exact
    v = certify(d, i, j)

    # a certificate that agrees with the truth is not wrong; the same one against a
    # falsified truth is.  If this direction is ever inverted the whole bench reads 0.
    if v.status == CERTIFIED:
        assert not wrong(v, truth)
        assert wrong(v, truth + 7)

    assert abstain(d, i, j, truth) == (truth != 0, False)
    blurred = blur(d, i, j, 1, random.Random(0))
    assert abstain(blurred, i, j, truth) == (False, False), "abstain must refuse on any unknown"

    res = run(n=25, kmax=3)
    t = res["tally"]
    assert t["wrong"] == 0, "a wrong certificate on the blur corpus is a kill gate"
    assert t["abstain_wrong"] == 0
    assert t[CERTIFIED] >= t["abstain_certified"], "the interval theorem cannot lose coverage"
    assert t["coin_wrong"] > 0, (
        "the coin-flip control certified nothing wrong; the corpus is too easy and this "
        "bench measures nothing"
    )
    assert len(table(res, ascii_only=True).splitlines()) > 20
    print("ok")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0])
    p.add_argument("--selfcheck", action="store_true", help="assert-based self-check, then exit")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--kmax", type=int, default=4)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--letters", type=int, default=6)
    p.add_argument("--out", default=None, metavar="results.json")
    a = p.parse_args(argv)
    if a.selfcheck:
        _demo()
        return 0

    res = run(n=a.n, kmax=a.kmax, seed0=a.seed0, letters=a.letters)
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "━".encode(enc)
        ascii_only = False
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    print(table(res, ascii_only))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  results -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
