<h1 align="center">tangle</h1>

<p align="center"><b>Two cables in one picture: it proves they cannot be pulled apart, or it refuses and names the crossing to re-shoot. It never guesses.</b></p>

<p align="center"><sub>Invented by <b>Teerth Sharma</b> · <a href="mailto:teerths57@gmail.com">teerths57@gmail.com</a> · <a href="https://github.com/teerthsharma/tangle">github.com/teerthsharma/tangle</a> · <a href="https://github.com/teerthsharma/tangle/blob/main/RESULTS.md">RESULTS.md</a></sub></p>

<p align="center">
<img width="640" src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/hero.png" alt="Left: a rendered two-cable scene, traced, its two crossings numbered, CERTIFIED LINKED with lk = 1. Right: a real Wikimedia Commons photograph of booster cables with 29 red rings on the places its own skeleton branches, REFUSED with the reason BRANCHED_SKELETON. A rail underneath reads: 247 free-licensed photographs this repository did not make, 0 certified, 0 wrong certificates.">
</p>

```bash
pip install tanglekit                              # the import and the command stay: tangle
tangle --synthetic --seed 2 --overlay verdict.png  # renders a scene, reads it, prints a verdict
```

**`0` wrong certificates** — over 2,000 rendered diagrams, 80 rendered scenes and 247 real
photographs, on which it also returned **`0` certificates at all**. Both numbers are below.

<p align="center">
<img src="https://img.shields.io/badge/tests-226%20passed-2ea043?style=flat-square" alt="226 tests passed">
<img src="https://img.shields.io/badge/CPU-only-0b7285?style=flat-square" alt="CPU only">
<img src="https://img.shields.io/badge/training-zero-6d28d9?style=flat-square" alt="zero training">
<img src="https://img.shields.io/badge/real%20photos-0%2F247-9b2a2a?style=flat-square" alt="0 of 247 real photos certified">
<img src="https://img.shields.io/badge/unknowns-refused-b45309?style=flat-square" alt="refuses on unknowns">
</p>

---

## What it does for you

Two cables are wound around each other behind your desk. Before you start pulling you want to know
one thing: are they *actually* locked, or do they only look bad? Pull on a genuinely locked pair and
you tighten it into a worse knot.

`tangle` looks at a picture of the two cables and answers that — or tells you it cannot see well
enough, and where to stand for a better shot. No maybe, no percentage: a yes with a proof behind it,
or a refusal with a reason.

**Where it actually works today.** It answers on pictures it renders itself. Point it at a real
photograph and, so far, it says "cannot see" every single time — 247 times out of 247. For 104 of
those 247 the reason is the same one, and fixing it is [the next thing being built](#roadmap).
Nothing here hides that: it is in the hero, in a badge, and measured in [Benchmarks](#benchmarks).

## How it works

1. It finds the two cables in the picture and separates them from the background *(segmentation)*.
2. It thins each cable down to a single line running along its middle *(skeletonisation)*, and every place those two lines touch is a crossing.
3. At each crossing it works out which cable is on top, by seeing whose outline the other one interrupted — the one that got covered up is underneath *(occlusion continuity)*.
4. Crossings that twist one way count as plus one, the other way minus one; it adds them all up and halves the total *(the Gauss linking number)*.
5. If that total is not zero, the two cables are genuinely locked together and no amount of pulling will separate them, so go unplug one.
6. Where the picture is too blurry to tell who is on top, it works out the answer for *both* possibilities at once, and if those answers disagree about whether the total is zero, it says so instead of picking one.

## When it says no

Refusing is a real answer here, not a crash. It has its own exit code, its own reason, and its own
instruction. Five things make it refuse:

| it says | because the picture | what you do |
|---|---|---|
| `NOT_TWO_COMPONENTS` | has only one colour of cable in it, or two too close to tell apart | photograph two cables of different colours |
| `BRANCHED_SKELETON` | shows a cable crossing over *itself*, or two cables touching and merging into one shape | separate them, or re-shoot so they do not overlap themselves |
| `NO_INTENSITY_GAP` | does not separate cable from background clearly enough | more light, or a plainer surface underneath |
| `OPEN_TRACE` | shows a cable that neither runs out of the frame nor comes back to itself | pull back so both ends leave the picture |
| `LK_STRADDLES_ZERO` | is readable, but one crossing is too unclear to call, and that one crossing decides the answer | re-shoot the one crossing it names, from the bearing it gives |

Only the last of those is a refusal *after* the maths ran, and it is the one that tells you where to
stand:

```
REFUSED  LK_STRADDLES_ZERO
look at  crossing 1 at 275,279, camera bearing 152 deg
advice   the achievable interval [0, 1] contains 0
```

The other four stop before any maths happens at all. On real photographs, all 247 refusals were of
that earlier kind — the picture never became a diagram.

**A refusal costs you a second photograph. A wrong certificate tells somebody to leave a live cable
plugged in.** That asymmetry is why the tool is built to refuse.

---

## The exact statement, for readers who want it

**The invariant.** Sum the signs of the crossings *between* the two cables, and halve it:

```math
\mathrm{lk}(A,B)\;=\;\frac{1}{2}\sum_{c\,\in\,C(A,B)}\varepsilon(c),
\qquad
\varepsilon(c)\;=\;\underbrace{\mathop{\mathrm{sgn}}\det\big[\,t_a\;\;t_b\,\big]}_{\mathrm{base}(c)\,:\ \text{in-plane, no depth}}\;\cdot\;\underbrace{x_c}_{\pm 1,\ +1\ \text{iff}\ A\ \text{is over}}
```

Here `t_a` and `t_b` are the directions the two cables are travelling *at that crossing*, measured in
the plane of the photograph. Their determinant needs no depth information at all; the only thing that
needs depth is `x_c`, the `±1` for who is on top.

That integer is an **isotopy invariant**. Bend the cables, drape them, re-route them, walk round and
shoot from the other side — as long as the four ends stay pinned where they leave the picture, `lk`
does not move. It is **camera-angle invariant by theorem**, which is exactly the property a network
trained on rope photographs does not have.

**The unknowns.** Both factors of `ε(c)` are `±1`, so `lk` is **affine** in every crossing the
photograph could not read. With `S` the signed sum over the readable inter-cable crossings and `k`
unreadable ones, the set of linking numbers achievable over all `2^k` ways of resolving them is
*exactly* `k+1` consecutive integers:

```math
\Big\{\,\mathrm{lk}\,\Big\}_{2^k}\;=\;\Big\{\ \tfrac{S-k}{2}+j\ :\ j=0,1,\dots,k\ \Big\},
\qquad
\mathrm{lk}_{\min}=\tfrac{S-k}{2},\qquad \mathrm{lk}_{\max}=\tfrac{S+k}{2}
```

Computed in **`O(k)`. No enumeration. No sampling. No posterior.** Certification is `lk_min > 0` or
`lk_max < 0`, and since the interval is contiguous with step 1, those two plus "straddles zero" are
exhaustive — there is no fourth case hiding.

<p align="center">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/mechanism.png" alt="Two rows of crossing glyphs on one shared number line. Top: four read crossings plus two unknown give S = 2, k = 2, and the interval [0, 2], which the zero bar cuts through, so REFUSED. Bottom: six read crossings give S = 4, k = 0, and the single point 2, clear of zero, so CERTIFIED LINKED.">
</p>

This is an **interval certificate, not a confidence score**. Flipping the over/under at any crossing
of a plane diagram always yields another diagram that a real pair of cables could form, so the `2^k`
orbit is the *exact* set of tangles consistent with what the camera saw.

That is an exact set, not a heuristic superset and not a sample. Checked against explicit `2^k`
enumeration: **`0/1000` patterns disagree**, over 192,540 enumerated lifts.

> **Explainable, minus the AI.** The output is an integer, the crossings it came from, the theorem it
> came from, and a SHA-1 digest of the diagram it came from. Nothing is learned, nothing is fitted,
> nothing is a black box. Run it twice, get the same integer.

---

## What it certifies, and what it refuses

<p align="center">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/clasp-linked.png" width="31%" alt="Verdict card: CERTIFIED LINKED, lk = 1">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/stack-separable.png" width="31%" alt="Verdict card: CERTIFIED SEPARABLE, by the over-everywhere witness">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/pile-refuse.png" width="31%" alt="Verdict card: REFUSED, interval [0,1], look at crossing 1">
</p>

**The certificate is one-directional.** This is the part everyone gets backwards.

| verdict | when | what it means | exit |
|---|---|---|---|
| ✅ **CERTIFIED LINKED** | the interval excludes zero | **Proved.** No motion separates these cables with their ends held. *Unplug one.* | `0` |
| ✅ **CERTIFIED SEPARABLE** | one cable is on top at **every** crossing between them | **Proved**, by a different and independently sufficient witness. *Just pull.* | `0` |
| 🟡 **NOT CERTIFIED** | `lk = 0`, or both cables are on top somewhere | The computation succeeded and **proves nothing** | `1` |
| 🛑 **REFUSED** | the interval straddles zero, or the trace is defective | The computation **declined**, with a cause and a next action | `2` |
| ⚫ **BAD INPUT** | the file cannot be read, or the flags do not parse | Not a verdict — the tool never ran | `3` |

**`lk = 0` is never "unlinked".** The Whitehead link has `lk = 0` and does not come apart. The word
*unlinked* is a banned substring inside this package, enforced by a test that greps the source and
every verdict the example set can produce. `SEPARABLE` is reachable only from the over-everywhere
witness, never from a zero linking number.

There is one more refusal the table above does not need but the code has: `ODD_CROSSING_PARITY`, a
parity theorem that catches every *odd*-cardinality tracer error, for free, with probability 1.

---

## Benchmarks

Every block below is pasted from the command above it. No number in this repository is hand-typed.
**[Full tables, every control, and every arm that lost](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md)**.

### On real images — 247 pictures this repository did not make

The one section whose input came from somewhere else. 99 knot photographs and 72 cabling photographs
from Wikimedia Commons, 27 published link diagrams, and 49 single-curve line drawings from
`tr33hugg3r/knot-crossings` on Hugging Face — all free-licensed, all pinned by sha256 in
`photos/manifest.json`, 51 MB.

```
.venv/Scripts/python real.py fetch --dir photos
.venv/Scripts/python real.py run   --dir photos

certified verdicts                  0   of 247
wrong certificates                  0   of  19 labelled
diagrams built at all               0   of 247
```

Not one real image reached `certify()`. Every one was stopped by a tracer precondition, before any
diagram existed:

| refusal | n | what it means about the picture |
|---|---|---|
| `BRANCHED_SKELETON` | 104 | a cable's own centreline forks: it crosses itself, or two strands merged into one blob |
| `NO_INTENSITY_GAP` | 65 | the two brightness classes of cable-ness separate by `F < 2.0` |
| `NOT_TWO_COMPONENTS` | 61 | one colour cluster, or two inside a just-noticeable difference of each other |
| `OPEN_TRACE` | 17 | the arcs do not chain into one curve, edge to edge or back to the start |

**The control that makes that zero readable.** `real.py control` renders 20 scenes, writes them to
PNG, and reads them back through the *identical* code path — same resize, same alpha handling, same
`one()`:

```
.venv/Scripts/python real.py control

synthetic control   n = 20
  CERTIFIED          13    65.0%
  REFUSED             7    35.0%
```

13 of 20 through the harness, 0 of 247 through the same harness on real input. **The harness is not
what is failing; the corpus is outside the envelope.**

**Against numbers published outside this repository.** 19 of the 27 link diagrams carry a linking
number nobody here wrote — 3 from the Thistlethwaite link table, 10 from standard facts about torus
and Whitehead links, 6 written into the filename by an uploader who has never heard of this tool:

```
right 0   WRONG 0   no verdict 19
```

The only column that could have been a disaster is `WRONG`, and it is empty because the tool never
answered. → [RESULTS.md §4](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md#4-real-images-247-pictures-this-repository-did-not-make)

<p align="center">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/refusal-wall.png" alt="A contact sheet of all 247 real images, every tile bordered in the refusal colour, over a stacked bar showing 104 BRANCHED_SKELETON, 65 NO_INTENSITY_GAP, 61 NOT_TWO_COMPONENTS and 17 OPEN_TRACE, with a control strip of 20 rendered piles below it of which 13 certify green.">
</p>

### On rendered images — every number below is SYNTHETIC

> These come from `tangle.synth`: matte constant-colour cables, no contact shadow, no specular
> highlight, no JPEG, no lens. **None of it has been confirmed by a real photograph.** It is stated
> separately from the section above for exactly that reason.

**The headline claim.** Not "tangle is accurate". The claim is that the `O(k)` interval certifies
strictly more pairs than refusing on any unknown crossing does, at the same zero error rate.

The baseline it is measured against is four lines of code: if any crossing is unreadable, refuse;
otherwise take the half-sum. That baseline also scores zero errors, so the gap between the two rows
is the only thing here that could have come out badly.

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

Ground truth is **not** the package's own arithmetic. Every entry is a closed braid word, and `lk` is
half the signed exponent sum over the inter-cable letters — known in closed form before any code
runs. → [RESULTS.md §1](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md#1-the-headline-coverage-at-zero-wrong-certificates)

**Picture to verdict, on rendered scenes.** 20 seeded piles × 4 nuisance arms = 80 scenes. Truth is
the scene's own height functions, so this scores the reader against the scene rather than against
itself.

```
.venv/Scripts/python -m pytest -q -s tests/test_vision.py

  over/under on accepted crossings
    clean           46/ 46  100.0%
    blur 1.0 px     47/ 47  100.0%
    blur 3.0 px     44/ 44  100.0%
    antialiased     45/ 45  100.0%

  certified / wrong / refused, out of 20 piles per arm
    clean           16     0     4   (80.0% certified)
    blur 1.0 px     17     0     3   (85.0% certified)
    blur 3.0 px     16     0     4   (80.0% certified)
    antialiased     14     0     6   (70.0% certified)
    rule of three: 0 wrong in 63 certified is an upper bound of 0.048, not a rate of 0
    unknown crossings among certified verdicts: {0: 63}

  coin flip on the identical diagrams, seed 20260905
    clean          read  16 certified / 0 wrong    coin  19 certified / 11 wrong
    blur 1.0 px    read  17 certified / 0 wrong    coin  18 certified / 12 wrong
    blur 3.0 px    read  16 certified / 0 wrong    coin  17 certified / 13 wrong
    antialiased    read  14 certified / 0 wrong    coin  19 certified / 11 wrong
```

Same extracted diagrams, over/under reader swapped for a coin flip: **32 wrong certificates against
0.** The corpus is not too easy — a guesser fails loudly on it.
→ [RESULTS.md §3](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md#3-photograph-to-verdict-the-rendered-corpus)

<p align="center">
<img src="https://raw.githubusercontent.com/teerthsharma/tangle/main/assets/coin-vs-occlusion.png" alt="The same rendered scene twice. Read from occlusion continuity it is CERTIFIED LINKED, unplug one, 0 wrong certificates. With the over/under reader replaced by a coin flip the same diagram is CERTIFIED SEPARABLE, just pull, which is wrong: 32 wrong certificates over the same 2,000 diagrams.">
</p>

### Reproduce all of it

```bash
python -m venv .venv && . .venv/*/activate   # .venv\Scripts\activate on Windows
pip install -e ".[test]"
pytest -q                # 226 passed, 2 skipped
pytest -q -s             # the same run, with the tables above printed
python bench.py          # the coverage table
python real.py fetch --dir photos && python real.py run --dir photos && python real.py control
```

`tangle` is zero-shot: no training, no neural network, no dataset, no GPU. Just `numpy`, `scipy`,
`scikit-image`, `pillow` and a theorem from 1833. The certified layer turns out 2,000 verdicts in
0.29-0.31 s on a laptop CPU, over three runs. The imaging layer is not timed yet.

---

## Prior art

Nothing in the certified layer is new. Saying so first is what makes the rest believable.

| work | what they have that this does not | what `tangle` adds |
|---|---|---|
| **Matsuno et al. 2006** ([TMECH](https://doi.org/10.1109/tmech.2006.878557)) | the pipeline itself: photo → topological rope model → knot invariant → verdict, published in 2006 | an explicit UNKNOWN state, certification over the achievable interval, refusal that names a crossing, open code |
| **KnotDLO** ([arXiv:2506.22176](https://arxiv.org/abs/2506.22176)) | it acts — ties knots on real hardware, 50% on overhand from unseen configurations | an invariant, a hard certificate, and a calibrated abstention |
| **Knots-10** ([arXiv:2603.23286](https://arxiv.org/abs/2603.23286)) | real rope photographs at scale, and a 58–69 pp accuracy collapse from studio images to phone photos | an invariant that is material-, colour- and viewpoint-independent **by construction** |
| **HANDLOOM** ([arXiv:2303.08975](https://arxiv.org/abs/2303.08975)) | a learned cable tracer that is better than this one, on real RGB-D images | an exact integer downstream of the trace, and a refusal when the trace is not good enough |
| **pyknotid / Spherogram / SnapPy** ([repo](https://github.com/3-manifolds/Spherogram)) | every invariant here, computed better and more generally — the third-party control this repo owes them | the image half, which is where 100% of the risk lives |

Two honesty notes. **Matsuno's paper is paywalled and has not been read here**, so if either of their
invariants turns out to be the linking number, the novelty scopes down to the unknown-crossing orbit,
the pinned ends, and the refusal. And `tangle` must not claim to beat the Knots-10 studio-to-phone
collapse until it has been run on their images, which it has not.

Also standing on **Gauss 1833** (the linking integral and the half-sum), **Goeritz 1933** and
**Gordon–Litherland 1978** (`det(L) = |Δ(−1)|`), **Habegger–Lin 1990** (string links), and **Lui &
Saxena 2013**, who did the ambiguity orbit as a particle-filter posterior. The one real difference
from them is *exactness and one-sidedness*: a hard certificate plus an abstention, instead of a
posterior plus a threshold.

---

## What we got wrong

Six design verdicts that are now dead, and two arms that lost after they shipped.

- **"`lk = 0` means just pull."** The Whitehead link. Killed, and the word *unlinked* is banned in code.
- **"Closing the diagram at the frame boundary makes `lk` camera-invariant."** It does not: the exit points move with the camera, and an exterior crossing's over/under is a *convention* that does not flip when its planar sign does. **The closure was deleted.** The object is a 2-string tangle with pinned ends, which needs no closure at all.
- **"A width bump at the crossing reads over/under."** For opaque cables the silhouette is *provably* depth-blind — the mask is bitwise identical under swapping which cable is on top. The cue carried zero information and was reading the renderer's drop shadow. There is now a test called `test_silhouette_carries_no_depth`.
- **"A contraction radius fixed in cable widths merges the skeleton's H-pattern."** It cannot: the bridge of an H at crossing angle θ is about `w / sin θ` long, so a constant radius always loses the shallow crossings. Replaced by an angle-aware admissibility rule (`vision.BRIDGE_K`).
- **"`det = 5` certifies a figure-eight tie-in."** A follow-through is tied on a bight, so the doubled rope bounds a band and the traced curve is the *unknot*, `det = 1`. That is why climbing, rigging and every load-bearing verdict are out of scope: the determinant is computed, but no code path turns one into a knot name.
- **"Threshold at the widest empty gap in the histogram."** That rule is satisfiable only by a renderer. A rendered cable is a flat stroke on a flat field, so its histogram is two spikes with nothing between; a photograph's is dense in every bin, because antialiasing, shading and depth of field put real pixels everywhere. Measured on the same 247 real images, the old rule admitted 97 and the Otsu-plus-Fisher rule that replaced it admits 182 (21 against 106 of the 171 photographs). It also widened the synthetic noise envelope from `σ = 6/255` to `σ = 16/255`, with 0 unsound certificates across a 7-level, 20-pile sweep.
- **Active perception lost to random.** Re-shooting the crossing the tool names: 19.9% certified. Re-shooting a uniformly random crossing at the same photograph budget: 20.0%. This was **predicted before it was measured** — `lk` is affine, so every unknown shrinks the interval by exactly 1 and no crossing is more decisive than any other. The `|sin θ|` ranking is a *perception* heuristic, not an information criterion. `bench.py` prints it; it is not buried.
- **On rendered scenes the interval theorem certified nothing the plain half-sum would not have.** All 45 certified verdicts across four nuisance arms had `k = 0`. The 10.3-point coverage gap exists on the braid corpus, where the unknowns are injected on purpose.

---

## Limits

Collected once, here.

- **It has been run on real images and it certified none of them.** 247 free-licensed images produce 0 certified verdicts, 0 wrong certificates, and 0 diagrams built at all. The same harness certifies 13 of 20 rendered piles through the same PNG round trip, so the zero is the corpus, not the plumbing.
- **Two visually distinct cables, or it refuses.** Colour does the segmentation, so a pile of identical black charging cables is one mask and `NOT_TWO_COMPONENTS` — 15 of 80 rendered scenes, and 61 of 247 real ones. The most common real scene is the worst case.
- **Self-crossings are refused, not handled.** `BRANCHED_SKELETON` on 7 of 80 rendered scenes and **104 of 247 real ones**. On a real pile most crossings are self-crossings, which makes this the single largest limit on the imaging layer, and the dominant reason the real number is zero.
- **Noise is a cliff, not a slope.** `σ = 16/255` traces 10/10 piles; `σ = 26/255` traces 0/10. The failure direction is always a refusal, never a wrong certificate, and 20 piles at each of 7 noise levels produce 0 certificates the scene contradicts.
- **Two of four nuisance arms fail the 40% refuse gate.** blur 3.0 px refuses 85%; antialiased refuses 55%.
- **Every benchmark percentage on this page except the real-image table comes from `tangle.synth`.** Coverage, the 133 of 133 crossings read the right way round, the 32 wrong certificates the coin flip buys, `TAU`, `BRIDGE_K`, the blur and antialiasing arms, closed-cable tracing, and every asset. The invariant layer is the one exception, and it is checked against closed forms rather than pictures.
- **No camera model, so no viewpoint-agreement measurement.** Invariance is tested as *diagram-move* invariance (R1/R2/R3), which is strictly weaker than camera invariance. The two-view interval intersection exists and is unwired.
- **The rendered corpus cannot exhibit `|lk| ≥ 2`.** An arch weaving across an arch cannot wrap twice; `|lk| ≥ 2` lives only in the `T(2,n)` family, which never goes through a camera.
- **An even number of tracer errors is uncaught** and can produce a confidently wrong certified integer. Parity catches odd counts; nothing single-view catches pairs. This is the live false-certification path, stated here rather than discovered in a bench → [RESULTS.md §7](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md#7-claims-not-earned).
- **The certificate answers a narrower question than you will ask.** "Cannot be separated with the ends held" is not "will be annoying to untangle". An `lk = 0` pile can still be a nightmare of friction.
- **Only `|lk|` is certified; the sign is a stated convention.** Image coordinates are y-down throughout, which negates every crossing sign relative to the y-up mathematical convention.
- **Never claimed, in any code path or string:** *unlinked* from `lk = 0`; *unknotted* from `det = 1`; a knot name from any determinant; chirality; a probability, confidence or percentage attached to a verdict; any climbing, rigging or safety verdict; anything at all about the cables outside the frame.

---

## Roadmap

- **Resolve a forked centreline instead of refusing on it.** By continuing each cable's direction through the junction, with a decisiveness margin so an ambiguous blob still refuses. This is the one gate that would move the real-image number off zero: it is 104 of the 247 refusals, and nothing downstream of it has ever seen a real picture.
- **More real photographs.** Knots-10 ([arXiv:2603.23286](https://arxiv.org/abs/2603.23286)) and HANDLOOM's annotated RGB-D cable traces are downloadable and need no camera. `real.py` already prints the refuse rate and the refusal-reason histogram, which is what ships before any accuracy claim.
- **Phone.** The certified layer is integer arithmetic with no dependencies — WebAssembly, then a camera app that refuses out loud and tells you where to stand.
- **Multi-view.** `certify.intersect` already intersects two views' intervals, and disjoint intervals *prove* one trace is wrong. It needs a camera model before it is worth wiring up.
- **The Alexander determinant at `t = −1`.** Implemented and checked against a closed-form ladder (`det(T(2,n)) = n`, figure-eight 5, Whitehead 8), but it is genuinely nonlinear in the unknowns: no interval bound of the `O(k)` kind exists, so it needs the `2^k` enumeration under a measured budget (`K_MAX = 16`, set from the 39.0 µs per lift at ten crossings in [RESULTS.md §2](https://github.com/teerthsharma/tangle/blob/main/RESULTS.md#2-the-invariant-layer-against-closed-forms)).

---

<p align="center"><sub>
<a href="https://github.com/teerthsharma/tangle/blob/main/LICENSE">MIT</a> · python ≥ 3.10 · <a href="https://github.com/teerthsharma/tangle/blob/main/RESULTS.md">RESULTS.md</a> ·
Invented by <b>Teerth Sharma</b> · teerths57@gmail.com ·
<a href="https://github.com/teerthsharma/tangle">github.com/teerthsharma/tangle</a><br>
Hero photograph <a href="https://commons.wikimedia.org/wiki/File:Booster_cables.jpg">Booster_cables.jpg</a> by Qurren, CC BY-SA 3.0, via Wikimedia Commons<br>
<code>knot-theory · linking-number · topological-invariant · computer-vision · zero-shot ·
certified · active-perception · explainable-ai</code>
</sub></p>
