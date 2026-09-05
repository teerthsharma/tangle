# Measured results

Everything below was produced by the commands shown, on one machine, at one commit.
No number here is quoted from a paper, a docstring, or an earlier run.

```
commit    ac84ca0            the last commit that changed code, working tree clean
machine   WIN-16QAL06O9GB    Windows 11, python 3.11.9
library   numpy 2.4.6   scipy 1.17.1   scikit-image 0.26.0   pillow 12.3.0
seeds     20260905 (nuisance + coin flip), 1000..1399 (braid corpus)
```

`bench.py` prints whatever `HEAD` is when it runs, so the hash in the block below is the
commit that produced the numbers, not the commit that records them: the documentation
commits that follow move `HEAD` without moving a number. The seconds on that same line are
wall clock and land between 0.29 and 0.31 s across runs; every other field is exact.

Reproduce all of it:

```
python -m venv .venv
.venv/Scripts/pip install numpy scipy scikit-image pillow pytest
.venv/Scripts/pip install -e .
.venv/Scripts/python -m pytest -q          # 196 passed
.venv/Scripts/python -m pytest -q -s       # the same run, with the tables below printed
.venv/Scripts/python bench.py              # the coverage table
.venv/Scripts/python -m tangle --synthetic --seed 1
```

---

## 1. The headline: coverage at zero wrong certificates

The claim being tested is not "tangle is accurate". It is: **the O(k) interval over the
2^k unknown-crossing resolutions certifies strictly more pairs than refusing on any
unknown, at the same zero error rate.** The abstain baseline is four lines and also scores
zero errors, so the only row that could have come out badly is the gap between them.

```
.venv/Scripts/python bench.py

==============================================================================
  certified pairs, at 0 wrong certified verdicts
  400 braid seeds x k = 0..4 blurred crossings = 2000 entries
==============================================================================
    tangle                                 23.6%   0 wrong (<=0.6% of certified at 95%)
    abstain on any unknown crossing        13.2%   0 wrong (<=1.1% of certified at 95%)
    ------------------------------------------------
    coverage gained by the interval theorem  10.3   points

  same diagrams, decision rule replaced:
    coin flip on unknowns                  70.3%   755 wrong
==============================================================================
    tangle CERTIFIED                       23.6%
    tangle NOT CERTIFIED                    6.2%
    tangle REFUSED                         70.2%
==============================================================================
  active perception -- the ranking is a perception heuristic, and theory
  predicts no gap: every unknown shrinks the interval width by exactly 1.
    named crossing re-shot, then certified 19.9%   of 1404
    random crossing re-shot, same budget   20.0%   control
==============================================================================
  k among certified verdicts -- if this is 0 almost everywhere, then
  "certified over all 2^k resolutions" is doing less work than it sounds.
    k=0: 276  k=1: 147  k=2: 49
==============================================================================
  the REFUSED rate above is a function of the blur schedule -- up to
  k = 4 of 6 crossings erased on purpose -- and is NOT a photograph's
  refuse rate.  That number needs the tracer and is not measured here.
------------------------------------------------------------------------------
  not run here: the R2-drape alternation control and the mask-overlap
  control need rendered scenes; every real-photograph table needs the tracer.
==============================================================================
  commit ac84ca0   machine WIN-16QAL06O9GB   python 3.11.9   0.29 s
==============================================================================
```

**Ground truth is not the package's own arithmetic.** Each entry is the closure of a braid
word on two strands; lk is half the signed exponent sum over the inter-component letters,
known in closed form before any code runs, and every entry asserts the diagram layer agrees
with that closed form.

**The `0 wrong` row measures the code, not the design.** A corpus that corrupts the input
in exactly and only the way the interval theorem is proved against cannot produce a wrong
certificate from a correct implementation. It ships as a regression test, not as evidence.

**The 70.2% REFUSED must not be read against the 40%-refuse kill gate.** Up to 4 of 6
crossings are erased on purpose. A photograph's refuse rate is section 3.

---

## 2. The invariant layer, against closed forms

```
.venv/Scripts/python -m pytest -q -s tests/test_certify.py tests/test_alexander.py

  O(k) interval vs explicit 2^k enumeration:
  0/1000 patterns disagree (192,540 lifts enumerated, k <= 10, seed 20260905)

  per-lift determinant cost, shadow hoisted out of the loop:
    (2,6) torus, 6 crossings:   13.8 us   ->  2^16 lifts =   0.9 s
    (2,10) torus, 10 crossings:   39.0 us   ->  2^16 lifts =   2.6 s
```

The control for the interval is `brute_force_interval`, which resolves every one of the
2^k lifts through `Diagram.resolve` / `Diagram.sign` rather than re-deriving the closed
form, so the two paths share no arithmetic. `K_MAX = 16` for the determinant is set from
that per-lift measurement, not chosen: 2^16 lifts is under 3 s, and `det_values` raises
`DeterminantRefused(K_EXCEEDS_BOUND)` above it rather than hanging. The two microsecond
figures are wall-clock and move a few percent between runs on the same machine; the test
asserts the ceiling that fixes `K_MAX`, not the figure.

Determinant known-answer family, verified for every entry:

| diagram | braid word | det(t = -1) |
|---|---|---|
| unknot / T(2,1) | `[1]`, 2 strands | 1 |
| Hopf link T(2,2) | `[1,1]` | 2 |
| trefoil T(2,3) | `[1,1,1]` | 3 |
| T(2,n), n = 1..10 | `[1]*n` | n |
| figure-eight 4_1 | `[1,-2,1,-2]`, 3 strands | 5 |
| Whitehead link L5a1 | `[1,-2,1,-2,-2]`, 3 strands | 8 |

Three free consistency checks run on every ladder entry: Euler `V - E + F = 2` on the
traced faces, the Torres parity theorem (det odd on one component, even on two or more),
and agreement between the two checkerboard colourings.

---

## 3. Photograph to verdict: the rendered corpus

20 seeded piles x 4 nuisance arms = 80 scenes. Truth is `synth.truth`, computed from the
scene's height functions; over/under is never authored, so this scores the reader against
the scene rather than against itself.

```
.venv/Scripts/python -m pytest -q -s tests/test_vision.py

  over/under on accepted crossings
    clean           46/ 46  100.0%
    blur 1.0 px     48/ 48  100.0%
    blur 3.0 px      8/  8  100.0%
    antialiased     31/ 31  100.0%

  certified / wrong / refused, out of 20 piles per arm
    clean           16     0     4   (80.0% certified)
    blur 1.0 px     18     0     2   (90.0% certified)
    blur 3.0 px      3     0    17   (15.0% certified)
    antialiased      8     0    11   (40.0% certified)
    rule of three: 0 wrong in 45 certified is an upper bound of 0.067, not a rate of 0
    unknown crossings among certified verdicts: {0: 45}

  coin flip on the identical diagrams, seed 20260905
    clean          read  16 certified / 0 wrong    coin  19 certified / 11 wrong
    blur 1.0 px    read  18 certified / 0 wrong    coin  18 certified / 12 wrong
    blur 3.0 px    read   3 certified / 0 wrong    coin   2 certified / 1 wrong
    antialiased    read   8 certified / 0 wrong    coin  12 certified / 8 wrong

  refusal reasons over every arm
    NOT_TWO_COMPONENTS    15
    BRANCHED_SKELETON      7
    OPEN_TRACE             4
```

That histogram counts the 26 scenes the **tracer** refused, where no diagram was ever
built. The 34 refusals in the table above it are those 26 plus the 8 scenes that traced
and then refused in the certified layer with `LK_STRADDLES_ZERO`; the `-s` run of
`tests/test_vision.py` lists those 8 by arm and seed. 45 certified + 34 refused + 1
`NOT CERTIFIED` = 80.

The control is the same extracted diagrams with the over/under reader replaced by a coin
flip: **32 wrong certificates across the four arms**, against 0. The corpus is therefore
not too easy — a guesser fails loudly on it.

The 100% over/under column clears the specification's kill gate (below 90% means the
diagram extraction, not the topology, is the bottleneck) on the crossings the tracer
accepts. It says nothing about the crossings it refused.

Refuse rate per arm, against the 40% kill gate: clean **20%** (pass), blur 1.0 px **10%**
(pass), antialiased **55%** (fail), blur 3.0 px **85%** (fail). Two of four arms lose.

### Noise: a cliff, not a slope

```
.venv/Scripts/python -m pytest -q -s tests/test_vision.py -k intensity_gap

  additive Gaussian noise, ten piles per level
    sigma  6.0/255   traced 10/10
    sigma  8.0/255   traced  0/10   ['NO_INTENSITY_GAP']
```

The threshold is chosen by the widest representable gap in the intensity histogram, so it
does not degrade: between sigma = 6 and sigma = 8 grey levels it goes from every pile to no
pile. The failure direction is a refusal, never a wrong certificate, and both sides are
pinned by a test.

---

## 4. End to end, through the CLI

```
.venv/Scripts/python -m tangle --synthetic --seed 1
  NOT CERTIFIED  OVER_MIXED                                         lk = 0
  interval [0, 0]   S = 0
  look at  crossing 0, 5
  source   from_braid([1, 1, -1, 1, -1, -1], strands=2)             exit 1
```

Exit codes are the verdict, so a shell can branch on them without parsing the block:
`0` CERTIFIED, `1` NOT CERTIFIED, `2` REFUSED, `3` bad input. All four are measured
below; read them with `echo $?` on the process itself, not through a pipe, which reports
the last stage's code instead.

The two scenes are rendered by the pipeline, not shipped:

```
.venv/Scripts/python -c "from PIL import Image; from tangle import synth as s; Image.fromarray(s.render(s.clasp(sign=1), seed=1)[0]).save('clasp.png'); Image.fromarray(s.render(s.pile(5), seed=5)[0]).save('pile5.png')"
```

```
.venv/Scripts/python -m tangle clasp.png          # synth.clasp(sign=+1), seed 1
  CERTIFIED  LINKED                                                 lk = 1
  interval [1, 1]   S = 2      k = 0
  advice   lk = 1 over all 2^0 resolutions.  Unplug one.            exit 0

.venv/Scripts/python -m tangle pile5.png          # synth.pile(5), seed 5
  REFUSED  LK_STRADDLES_ZERO
  look at  crossing 1 at 275,279, camera bearing 152 deg
  advice   the achievable interval [0, 1] contains 0                exit 2
```

The clasp scene is authored with `sign = +1` and the certificate reads `lk = +1`, while the
truth diagram's own lk is `-1`. That is not an error: the tracer walks each centreline from
whichever end it meets first, `_match` reports cable 0 traced backwards, and reversing one
component of a two-component diagram negates lk. In the traced orientation the truth is
`+1`, exactly what was certified. **Only `|lk|` is certified. The sign is a stated
convention** — image coordinates are y-down throughout, which negates every crossing sign
relative to the y-up mathematical convention.

The refusal is the interesting exit. `pile(5)`'s truth is `lk = 0`, and the tool does not
say "unlinked" — it says the achievable interval is `[0, 1]`, names crossing 1, and gives a
camera bearing. `assets/tangle.gif` is that scene's four frames.

---

## 5. Every arm that lost

1. **Active perception ranking loses to random.** 19.9% certified after re-shooting the
   named crossing, 20.0% after re-shooting a uniformly random one at the same photograph
   budget, over 1404 straddles. This was **predicted before it was measured**: lk is affine
   in the crossing signs, so every unknown shrinks the interval width by exactly 1 and no
   crossing is more decisive than any other. The `|sin theta|` ranking is a perception
   heuristic — it picks the crossing a camera can most likely resolve — and it is not an
   information criterion. `bench.py` prints it; it is not buried.

2. **On rendered scenes the interval theorem certified nothing the exact half-sum would
   not have.** All 45 certified verdicts across the four nuisance arms had `k = 0`. The
   10.3-point coverage gap in section 1 exists only on the braid corpus, where unknowns are
   injected on purpose. Every "certified over all 2^k resolutions" claim must be read next
   to `k = 0: 276, k = 1: 147, k = 2: 49` — nothing above `k = 2`, because certification
   needs `|S| > k`.

3. **Two of four nuisance arms fail the 40% refuse gate.** blur 3.0 px refuses 85%,
   antialiased refuses 55%. Antialiasing partly defeats the widest-gap threshold by
   construction: it manufactures intermediate intensities in the gap the threshold needs.

4. **Self-crossings are refused, not handled.** `BRANCHED_SKELETON` on 7 of 80 scenes. On
   real cable piles most crossings are self-crossings, so this is the single largest limit
   on the imaging layer. Per-cable H-contraction is the upgrade path; it is not built.

5. **Same-colour cables are out of scope.** `NOT_TWO_COMPONENTS` on 15 of 80 scenes. Colour
   does segmentation; continuity does over/under. A monochrome pile refuses.

6. **Noise above sigma = 6/255 is a total loss**, 10/10 traced to 0/10 in two grey levels.

7. **No real photographs.** Every number here comes from `tangle.synth`: matte
   constant-colour cables, no contact shadow, no specular highlight, no JPEG, no camera
   model, no lens. The generator also rejects self-crossings, crossings closer than 4 cable
   widths, and crossing angles below 25 degrees. That rejection is a scope condition on the
   corpus, **not** a result about the reader.

8. **The corpus cannot exhibit `|lk| >= 2`.** An arch weaving across an arch cannot wrap
   twice. `|lk| <= 1` everywhere in section 3, asserted in a test so it cannot quietly stop
   being true. `|lk| >= 2` lives only in the `T(2,n)` family, which never goes through a
   camera.

9. **Reidemeister invariance is tested as diagram-move invariance, which is weaker than
   camera invariance.** There is no camera model, so there is no viewpoint-agreement test
   and no multi-view result. `certify.intersect` (the two-view interval intersection)
   exists and is unwired.

10. **R3 is asserted on the pairwise signed sums, not on lk.** Three arcs each crossing the
    others once have interleaved ends, where lk is a half-integer. That refusal is itself
    asserted.

11. **Two controls named in the specification were not run.** The R2-drape alternation
    control and the mask-overlap control need rendered scenes rather than diagrams;
    `bench.py` prints their absence in its own output rather than skipping them silently.

---

## 6. Claims NOT earned

- **`lk = 0` never certifies "unlinked".** The Whitehead link has `lk = 0` and is not
  splittable. The honest output is `NOT CERTIFIED`. The certificate is one-directional,
  `|lk| >= 1 => LINKED`; the converse is not available from the linking number and the
  package never prints it. `SEPARABLE` is issued only by a different, sufficient witness
  (one cable is the over-strand at every inter-component crossing).
- **`det = 1` never certifies "unknotted".** `det = 1` is shared by infinitely many
  nontrivial knots.
- **Determinant collisions are not resolved.** Nothing here distinguishes two diagrams with
  the same determinant. Chirality (reef vs granny) would need the Kauffman bracket, which
  is not implemented.
- **Nothing outside the picture is claimed.** The convention is printed with every verdict:
  the scene is a ball, each cable is an arc properly embedded in it with its two ends fixed
  where it leaves the frame, and the verdict is invariant under any motion inside the scene
  that keeps the cables disjoint and the four ends fixed.
- **No third-party invariant cross-check.** `Diagram.to_pd()` is not implemented and
  Spherogram is not an allowed dependency, so the closed-form ladder in section 2 is the
  only external control on the determinant.

---

## 7. Assets

`assets/` holds overlays reproduced by the full pipeline — `synth.render` ->
`vision.trace` -> `certify` -> `viz.render`, with no hand-authored diagram anywhere:

| file | verdict |
|---|---|
| `clasp-linked.png` | CERTIFIED LINKED, `lk = 1` over 2^0 resolutions |
| `stack-separable.png` | CERTIFIED SEPARABLE by the over-everywhere witness |
| `pile-refuse.png` | REFUSED LK_STRADDLES_ZERO, interval `[0, 1]`, names crossing 1 |
| `tangle.gif` | the four frames of that refusal: bare, cables, crossings, banner |

Every frame carries the diagram digest in its footer, so a GIF cannot be faked frame by
frame.
