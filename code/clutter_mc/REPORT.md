# Does a choppy sea give a heavier clutter tail? Monte Carlo results

2-D MFIE + MOMI (Toporkov 1998) on 1-D Pierson–Moskowitz surfaces, PEC, λ = 0.23 m, backscatter.
Run on 2026-11-08. About 5,300 random seas, each solved as Gaussian and choppy, HH and VV, at 60° and 80°.
That's about 41,000 scattering solutions in total. The code is in this folder, raw per-realization results are in `results/`, and the figures are in `figures/`.

---

## 1. Short answer

1. **Choppy surfaces raise the mean HH level, and the effect is robust.** On the same random seas, choppy is
   +0.6 to +1.8 dB above Gaussian at HH, significant in all 6 HH cases. VV moves less: +0.3 to +0.5 dB at 80°, and nothing at 60°.
2. **Choppy surfaces do not significantly fatten the HH tail near grazing (80°).** The tail is P(σ > τ) normalized
   by each sea's own mean. The paired difference in ⟨σ²⟩/⟨σ⟩² at 80° HH is +0.09 [−0.33, +0.45] at 5 m/s,
   +0.62 [−0.72, +2.45] at 10 m/s, and −0.09 [−0.50, +0.33] at 15 m/s. At 60° HH there is a small, significant fattening at 10 and 15 m/s (+0.18, +0.46).
   **So a clear "no significant difference" at the angle you cared most about, and a modest effect at 60°.**
3. **The heavy HH tail near grazing is already there for the Gaussian sea, and it comes from the long waves.**
   At 80° HH, Gaussian ⟨σ²⟩/⟨σ⟩² is 2.4, 3.0 and 3.0 at 5, 10 and 15 m/s; exponential speckle gives 2. The 12 m radar patch sits on a
   long wave that tilts it toward or away from the radar, and at 10° grazing that swings the echo hard. Removing the long waves (the old
   periodic-on-the-window design) makes this tail measurably lighter (Δm2 = +0.45 [+0.07, +0.83] at 15 m/s).
4. **VV is close to exponential everywhere** (m2 = 2.0–2.2). Its mean is within 0.6 dB of first-order Bragg at 80° and 0.9–1.4 dB above it at 60°.
   HH sits 6–11 dB above Bragg at 80°, which is the known failure of first-order theory for HH near grazing.
5. **The tail curves can be trusted down to P(σ > τ) ≈ 1 %**, and not to 1e-3. Below about 1 % the results also depend on a handful of
   choppy realizations whose value is not resolution-converged (section 4).

---

## 2. What I ran and why

### 2.1 Design: what is held fixed

| quantity | value | why |
|---|---|---|
| footprint (Thorsos g) | **55λ = 12.65 m, same for every case** | 1.11 × Kapp's minimum at 80° (49.4λ). Fixed footprint = same number of "independent scatterers" in every comparison. |
| solve window | L = 8g = 101 m | 6g for Kapp plus margin, because long waves shift where the beam lands (z·tan θ, up to ~10 m at 80°) |
| sampling | **dx = λ/20** (N = 8800), with λ/40 re-solves of the tail (section 4) | λ/10 is fine for Gaussian, **not** for choppy |
| spectral cutoff | K_c = 4.99k | thesis baseline |
| surface generation | P-M on a **942 m periodic domain**, solved on the 101 m window | keeps the long waves (Λp = 23, 91, 205 m at 5, 10, 15 m/s) |
| angles | 60° and 80° from vertical (30° and 10° grazing) | one matrix per surface serves both |
| solver | MFIE + curvature diagonal (eq. 4.1), MOMI × 3 iterations; dense direct solve where MOMI did not converge | |

**On the design issue you raised.** I took your instinct (fix the footprint) and decoupled the two things that were tangled together.
The number of scatterers is set by g and is identical everywhere. How much of the spectrum is represented is set by the generation domain,
which now holds the whole P-M spectrum at every wind. Solve cost depends only on the window, so the long waves come for free. Inside the window
they show up as what they physically are for a 12 m radar cell: a random tilt, a random height offset, and gentle curvature (figure 5).

Represented surface (window averages vs. full-spectrum P-M values):

| U | h_rms in window | P-M h_rms | rms slope in window | P-M rms slope (to K_c) |
|---|---|---|---|---|
| 5 m/s | 0.13 m | 0.133 m | 0.152 | 0.152 |
| 10 m/s | 0.51 m | 0.533 m | 0.169 | 0.170 |
| 15 m/s | 0.99 m | 1.20 m | 0.178 | 0.179 |
| 15 m/s, periodic on window (old design) | 0.88 m | 0.96 m | 0.178 | 0.179 |

The window h_rms falls short at 15 m/s only because the window mean is removed, and a 101 m window can't contain a 205 m wave's full
height swing. The slope is fully represented in every case. **One thing worth noticing:** P-M slope barely grows with wind (0.152 → 0.179),
because the high-K part of the spectrum, α/4K³, does not depend on wind. P-M steepness k_p·h_rms = √(α/6) ≈ 0.037 at *every* wind.
So wind mostly adds long waves; it barely adds sharpness. That is the main reason choppiness doesn't grow much from 5 to 15 m/s.

**Pairing.** Gaussian and choppy versions of a sea share the same random Fourier coefficients, and the same draws are reused across winds.
"Choppy minus Gaussian" is therefore computed seed by seed, with the bootstrap resampling seeds jointly. That gives much tighter CIs on the
difference than comparing two separate intervals.

### 2.2 The choppy surface

`gallery_08_sea_surfaces.py` isn't in the repository, so I rebuilt the choppy surface from `sea_surface_1d.py`'s
Lagrangian approach: x(u) = u + D(u), z = h(u), with D the Hilbert-transform displacement using the crest-sharpening sign. For a single
wave h = a cos Ku this is the Gerstner wave x = u − a sin Ku. My minimum dx/du at 5 m/s is about 0.3, matching your 0.30, so it is very likely the same model.
The earlier 250-realization numbers won't reproduce exactly (different seeds and lengths).

> **Check this on your laptop:** `sea_surface_1d.surface(..., lam=+1)` produces *upside-down* Gerstner waves. Verified with a single
> sinusoid: `lam=+0.5` gives height skewness **−0.54** (sharp troughs, flat crests), and `lam=−0.5` gives +0.54. If `gallery_08` or
> `sea_surface_xband.py` used `lam=+1`, the earlier "choppy" surfaces had the wrong shape.

**Overturning.** No solve window ever contained an overturned crest (min dx/du in any window: 0.20, 0.10, 0.04 at 5, 10, 15 m/s).
On the full 942 m domain, overturning occurred in 0/1527, 1/2304 and 3/1510 seas, always outside the window. So nothing was dropped and
nothing needed to be. Overturning is a non-issue for P-M with this cutoff, for the slope reason above.

### 2.3 Validation

* Thesis tests 1–7: 63/63 checks pass on this machine.
* Fast CPU propagator vs `mfie2d.propagator`: 1e-15. Lean MOMI vs `mfie2d.solve_momi`: 4e-15.
* **GPU port** (PyTorch, complex128) vs CPU on 4 seas × 8 cases: worst difference 3e-12 dB. PyTorch's Bessel functions are only
  about 1e-6 accurate for 1 ≤ kR ≤ 25, so that near-field band is computed with SciPy on the CPU and patched in. CPU throughput was the
  real bottleneck at λ/20: the GPU does one sea (both surfaces, both polarizations, both angles) in about 4 s, against about 35 s on a CPU core.
* Per realization, stored for every one: MOMI iteration 2→3 change, energy ∫σ⁰dθs vs Thorsos's finite-beam prediction, and current at the window edge.
* **Sampling convergence** (18 seas, λ/10, λ/20, λ/40):

| | λ/10 → λ/20, median (max) | λ/20 → λ/40, median (max) |
|---|---|---|
| Gaussian | 0.02–0.17 dB (0.94) | 0.001–0.007 dB (0.16) |
| choppy | 0.23–2.3 dB (**11.9**) | 0.02–0.22 dB (0.89) |

  **λ/10 is unusable for choppy surfaces**, as you suspected. The Gerstner transform is applied to the whole spectrum, so the centimetre
  ripples get sharp crests too: local curvature scales like 1/(dx/du)³ ≈ 37× at dx/du = 0.3. And K_c = 4.99k sits exactly at the λ/10 Nyquist limit.

---

## 3. Results

Figures:
* `fig1_gauss_vs_choppy_U10.png`: exceedance curves with 95% bootstrap bands, 10 m/s, all four pol/angle cases (experiments 1 and 2)
* `fig2_wind_sweep_80HH.png`: 80° HH at 5, 10, 15 m/s, plus the old-design arm (experiment 3)
* `fig3_means_and_m2.png`: mean level minus Bragg, and ⟨σ²⟩/⟨σ⟩², for every case
* `fig4_paired_differences.png`: paired choppy − Gaussian differences
* `fig5_surfaces.png`: what the solver actually sees at each wind
* `fig6`–`fig8`: the problem cases (section 4)

### 3.1 Main table (n = 1527 / 2304 / 1510 seas at 5 / 10 / 15 m/s; 95% bootstrap CIs)

Mean levels in dB. "Δ" = paired choppy − Gaussian. **Bold** = CI excludes 0.

| U | case | Bragg | Gauss ⟨σ⁰⟩ | choppy ⟨σ⁰⟩ | Δ⟨σ⁰⟩ dB | Gauss m2 | choppy m2 | Δm2 |
|---|---|---|---|---|---|---|---|---|
| 5 | HH 60° | −32.1 | −30.4 | −29.7 | **+0.64** [0.35, 0.94] | 1.99 [1.89, 2.11] | 2.09 [2.00, 2.20] | +0.10 [−0.05, 0.25] |
| 5 | HH 80° | −52.2 | −46.4 | −44.8 | **+1.60** [1.25, 1.92] | 2.42 [2.17, 2.83] | 2.51 [2.28, 2.75] | +0.09 [−0.33, 0.45] |
| 5 | VV 60° | −15.2 | −14.3 | −14.2 | +0.15 [−0.17, 0.46] | 2.00 | 2.11 | +0.11 [−0.08, 0.32] |
| 5 | VV 80° | −15.9 | −16.0 | −15.6 | **+0.41** [0.09, 0.73] | 2.05 | 2.03 | −0.02 [−0.17, 0.13] |
| 10 | HH 60° | −32.1 | −29.9 | −29.1 | **+0.76** [0.50, 1.02] | 2.13 [2.03, 2.24] | 2.31 [2.18, 2.46] | **+0.18** [0.01, 0.36] |
| 10 | HH 80° | −52.2 | −43.9 | −42.3 | **+1.56** [1.22, 1.90] | 2.98 [2.48, 3.69] | 3.60 [2.59, 5.24] | +0.62 [−0.72, 2.45] |
| 10 | VV 60° | −15.2 | −14.0 | −13.9 | +0.02 [−0.24, 0.29] | 2.05 | 2.07 | +0.02 [−0.10, 0.14] |
| 10 | VV 80° | −15.9 | −15.7 | −15.3 | **+0.46** [0.22, 0.70] | 2.02 | 2.11 | +0.09 [−0.04, 0.23] |
| 15 | HH 60° | −32.1 | −29.6 | −28.6 | **+0.91** [0.57, 1.26] | 2.21 [2.06, 2.37] | 2.67 [2.32, 3.03] | **+0.46** [0.07, 0.86] |
| 15 | HH 80° | −52.2 | −42.9 | −41.1 | **+1.79** [1.42, 2.15] | 3.04 [2.74, 3.36] | 2.95 [2.64, 3.31] | −0.09 [−0.50, 0.33] |
| 15 | VV 60° | −15.2 | −13.8 | −13.9 | −0.06 [−0.38, 0.24] | 2.12 | 2.02 | −0.10 [−0.29, 0.08] |
| 15 | VV 80° | −15.9 | −15.8 | −15.5 | +0.32 [0.00, 0.65] | 2.11 | 2.17 | +0.06 [−0.14, 0.24] |

Bragg: ⟨σ⁰_HH⟩ = (πα/4)cos⁴θ/sin³θ, ⟨σ⁰_VV⟩ = (πα/4)(1+sin²θ)²/sin³θ, α = 8.1e-3.

Old design (15 m/s, surface periodic on the 101 m window, n = 752, Gaussian): HH 80° m2 = 2.59 [2.36, 2.82] vs 3.04 with long waves,
**Δ = +0.45 [+0.07, +0.83]**. Mean levels are unchanged (+0.31 [−0.21, +0.81] dB). Other cases show no significant difference.

### 3.2 Plain-language interpretation

* **What the sea does to the mean echo.** Choppy (sharp-crested) waves return more HH power, about +1.6 to +1.8 dB at 10° grazing
  and +0.6 to +0.9 dB at 30°. Sharp crests have more small-scale curvature, and HH is the polarization that responds to it. VV, which is
  dominated by Bragg ripples, barely notices. For a radar this shifts the whole clutter distribution up. It is a calibration effect, not a change in shape.
* **What the sea does to the tail shape.** Near grazing, the HH tail is heavier than speckle for *both* seas. The extra spread comes
  from long waves tilting the patch, not from crest sharpness: Gaussian and choppy have the same m2 within error at 80°. In radar
  language, the tilt modulation acts like the "texture" in a K-distribution. Fitting ⟨σ²⟩/⟨σ⟩² = 2(1 + 1/ν) gives ν ≈ 1–5 for HH at
  80°, which means spiky clutter, and mostly ν ≳ 10 for VV, which is almost pure speckle.
* **Wind.** HH-80° tail heaviness goes from m2 ≈ 2.4 at 5 m/s to ≈ 3.0 at 10 and 15 m/s. It rises once the long waves are big
  enough to tilt the patch by several degrees, then saturates. That matches the P-M self-similarity: the slopes hardly grow with wind.
* **Why choppy doesn't add tail at 80°, and what that does and doesn't mean.** In P-M, the dominant waves are only about 4 % steep at
  every wind, so their Gerstner sharpening is mild. The sharpness the choppy model does add sits mostly on the centimetre ripples, and
  at 10° grazing it apparently changes the overall level more than the fluctuations. A real sea with breaking or steeper dominant waves
  (JONSWAP peaks, wave groups) could behave differently. These results say nothing about that.

### 3.3 How far down can P(σ > τ) be trusted?

The per-case "trust floor" (the smallest P whose 95% CI is within a factor of about 1.5) is **P ≈ 0.7–1.6 %**, roughly 10/n. At HH 80°
that corresponds to τ ≈ 7–8 ⟨σ⟩. Below that, the curves in fig 1 and 2 are drawn, but their bands are wide and any ratio there is
noise (the "P(>10⟨σ⟩)" columns in `results/summary.txt` are mostly undefined). Reaching P ≈ 1e-3 with a ±50 % CI would need about 10⁴ seas per case
(≈ 5 GPU-hours per wind on this machine at λ/20, ~20 at λ/40), or importance sampling.

---

## 4. Problems found, what I did about them, and what is still open

1. **MOMI diverged once** (seed 689, 10 m/s, choppy HH): iterations 2 and 3 differed by 4–13 dB, and energy came out at 1.054.
   It was 1 solve out of about 41,000. Every other MOMI 2→3 change was ≤ 0.07 dB. That realization was re-solved with a dense direct solve
   (`fixup.py`), and `analyze.py` logs the substitution.
2. **Under-resolved choppy crests are the real issue (figs 6, 7).** The divergence above came from a near-cusp, a crest radius of
   **1.8 mm** sampled at 11.5 mm. It sits **30 m outside the footprint**, where the current is about 1 % of its peak, and it still changed HH-80°
   backscatter by 29 dB: the under-resolved crest creates a spurious resonance. At λ/20, κ·dx > 1 occurs in 9–23 % of choppy windows;
   Gaussian surfaces never exceed 0.36. Tests at λ/40:
   * **Random 60 seas per wind (unbiased):** the λ/20 → λ/40 shift in the mean is ≤ 0.16 dB, with CIs covering 0, and m2 changes by ≤ 0.08.
     **The bulk statistics and the mean-level conclusions are not resolution artifacts.**
   * **Top-8 HH spikes per case:** most move by < 1.5 dB, but 7 of the 48 records (4 distinct seas) move by 4–49 dB, one of them *up* (seed 1305: +4.7 dB, to 31 × the mean).
     All λ/40 values (656 records) are substituted into the final numbers.
   * **Still open:** whether λ/40 is converged for the sharpest crests. κ·dx is still up to about 2.6 there, and λ/80 (N = 35,200) doesn't fit
     in 8 GB of GPU memory. The 10 m/s choppy HH-80° m2 (3.60) leans on seed 1305; with its λ/20 value it is 3.06. **Treat that cell as unresolved.**
3. **About 10 % of seas at 15 m/s, 80° fail the energy check by more than 2 %** (fig 8). The failures track footprint *tilt*:
   when the long wave slopes away from the radar by more than about 5°, the specular lobe rotates past 90° and leaves the angular range the check
   integrates over. So mostly the check is mis-reading geometry, not catching a solver error. Excluding these seas leaves m2 unchanged (3.04 → 3.08).
   *Not verified:* those patches sit at about 4° local grazing, where Kapp's criterion for the local angle would need a much larger g, so their
   (weak) backscatter values are less certain. They are in the bulk, not the tail.
4. **VV current at the window edge** reaches 10–50 % of the peak in some seas at 10–15 m/s. Energy still balances to < 0.5 % in the median,
   so truncation looks benign, but I didn't test VV backscatter against a longer window.
5. **Multiple comparisons.** The table has 24 paired tests, so about 1 "significant" result is expected by chance at 95 %. The two significant
   Δm2 values at HH 60° (10 and 15 m/s) are consistent in sign and grow with wind, which argues against chance, but neither is strong.
6. **Small MOMI residuals** (13 records with 0.01–0.07 dB between iterations 2 and 3) were left as they were. They are far below the statistical noise.
7. **CPU worker shards** were still running when this snapshot was taken (they hold about 6 % of the seeds). Re-running `analyze.py` later
   will include them; the numbers will move within their CIs.

## 5. What I'd do next

1. **Fix the discretisation for choppy surfaces properly**: arc-length (or curvature-adaptive) nodes, so crests get points where
   they need them. Uniform-x sampling is the wrong tool for Gerstner cusps. Then re-run the choppy arm, and the λ/80-equivalent question goes away.
2. **Apply the choppy transform only to the long waves** (K < a few K_p) as a second model. That isolates "sharp crests of the dominant waves",
   which is closer to what the physical question is about, and it avoids millimetre-radius ripple cusps.
3. **Go after the tail with importance sampling** instead of brute force: bias the draws toward steep, tilted-toward-the-radar
   footprints (the variables the spikes depend on) and reweight. Brute force to 1e-3 is about 10⁴ seas per case.
4. **Vary the footprint deliberately** (g = 55λ, 110λ, 220λ at 80°) to measure how m2 approaches 2 as the patch averages over more long-wave
   tilt. That ties the K-distribution shape parameter ν to the cell size, which is the result a radar engineer would use.
5. Repeat at 85° HH, where shadowing should start to matter, with g scaled per Kapp.

## Files

| file | what |
|---|---|
| `surfaces.py` | Gaussian / choppy P-M generator (long periodic domain, solve window) |
| `fastsolve.py`, `gpusolve.py` | lean CPU and GPU propagator + MOMI (validated against `mfie2d`) |
| `run_mc.py`, `launch.sh` | resumable sharded production run |
| `check_impl.py`, `gpu_check.py`, `spotcheck.py`, `convergence.py`, `conv_summary.py` | validation |
| `fixup.py`, `lambda40_study.py`, `lambda40_summary.py`, `curvature_scan.py`, `problem_cases.py` | problem hunting |
| `analyze.py` | statistics and figures 1–4 (`MC_PATCH40=0` gives the raw λ/20 version, `results/summary_raw20.txt`) |
| `results/*.jsonl` | raw per-realization records: seed, parameters, σ, all diagnostics |
