# Understanding Toporkov (1998) through small test cases

A small, readable 2-D solver ([mfie2d.py](mfie2d.py)) written straight from the thesis's equations,
plus seven test scripts. Each script teaches one idea and checks it against something independent:
exact physics, perturbation theory, a second formulation, or the thesis's own numbers.

Run any test from this folder with `python3 test_0N_*.py`. Figures go to [figures/](figures/).
Runtimes on a laptop: tests 1, 4: seconds; tests 2, 3, 5, 7: 1–2.5 min; test 6: about 3 min.

---

## 1. The thesis in one page

**Question.** What radar echo comes back from a rough, ocean-like, perfectly conducting surface,
especially at **low grazing angles (LGA)**? This is the regime of ship and coastal radars.
Analytical approximations (Kirchhoff, small-perturbation) break down there, so the thesis computes
the answer exactly and numerically, then averages over many random surfaces (Monte Carlo).

**Why 2-D.** The surface is z = ζ(x) with no variation in y. The field then splits into two
independent scalar problems:

| thesis name | radar name | field ψ | boundary condition on PEC | unknown "current" J |
|---|---|---|---|---|
| TE | **HH** (horizontal) | E_y | Dirichlet, E = 0 | ∂E/∂n on the surface |
| TM | **VV** (vertical) | H_y | Neumann, ∂H/∂n = 0 | H on the surface |

Your proposal's Dirichlet problem is the **TE/HH** case, which is also the harder one at LGA.

**The integral equation (MFIE, eq. 2.13).** Green's theorem turns the PDE in the whole half-space
into an equation on the surface only:

    J(x) = J_i(x) + ∫ P(x, x') J(x') dx'          discretized:  J = J_i + P J   (2.18)

* `J_i` = the **Kirchhoff current**: 2 × (incident field or its normal derivative). It is the exact
  answer for a flat plate (image theory) and the "single bounce" answer for gentle surfaces.
* `P` = the **propagator**: how much current at x' induces at x by re-radiating along the surface.
  For a flat surface P ≡ 0 (test 1). All multiple scattering lives in P.

**Solving it (Sec. 2.2).** Dense solve `(I−P)J = J_i`, or iterate. The plain Neumann series adds one
bounce per term and converges slowly or not at all. **MOMI** splits P = L + U (sources to the left /
right of the observer). One forward sweep `(I−L)⁻¹` sums all forward multiple scattering exactly,
because at LGA energy mostly travels one way along the surface. Two iterations are enough (test 4).

**What gets measured (eq. 2.30–2.34).** The far-field amplitude is a cheap integral of J. The
**NRCS** σ⁰(θ_i, θ_s) is |far field|² normalised by incident power. Backscatter is θ_s = −θ_i.
The thesis averages σ⁰ over realizations; **your project wants the whole distribution** of the
single-realization values, P(σ > τ) (test 7).

**Surfaces (Ch. 3).** Gaussian random surfaces built from a spectrum W(K) (Pierson–Moskowitz, eq. 4.2):
a sum of sinusoids with random amplitudes. Scattering at LGA is dominated by the **Bragg** component
K = 2k sin θ_i (test 2), so the high-K tail of W and its cutoff K_c matter (test 5).

**The two main contributions:**
1. **Ch. 4, the curvature term.** The MFIE kernel is finite on the diagonal, but its limit is
   ±ζ''Δx / (2π(1+ζ'²)), not 0. Original MOMI set the diagonal to 0, which made results depend on Δx,
   worst for HH near grazing. Including the term fixes it (tests 3, 6).
2. **Ch. 5, tapers.** A finite surface needs a finite beam. The Gaussian / Thorsos taper only
   approximately satisfies Helmholtz, and near grazing it fails for HH: a spurious jump in backscatter
   and a slowly decaying tail in the current. Fix: beam waist g > 9.4 / ((π/2 − θ_i) k cos θ_i)
   (eq. 5.8), and a surface about 6 g long. Tests 1 and 3 show both effects.

---

## 2. The tests and what they found

| # | script | idea | key result |
|---|---|---|---|
| 1 | [test_01_flat_plate.py](test_01_flat_plate.py) | flat plate: P = 0, J = J_i, energy conservation | ∫σ dθ_s / 2π cos θ_i matches Thorsos's finite-beam correction 1 − (1+2tan²θ)/(2(kg cosθ)²) to 1e-7 |
| 2 | [test_02_bragg_sinusoid.py](test_02_bragg_sinusoid.py) | grating orders, Bragg resonance, HH vs VV | MFIE matches first-order perturbation theory; VV/HH = ((1+sin²θ)/cos²θ)², about 4000× at 80° |
| 3 | [test_03_convergence_efie_vs_mfie.py](test_03_convergence_efie_vs_mfie.py) | curvature term; EFIE vs MFIE; taper residual | operator error ∝ Δx³ with curvature term, ∝ Δx without; Thorsos taper makes EFIE and MFIE disagree by 24%, exact beam by 2.6% |
| 4 | [test_04_momi_vs_neumann.py](test_04_momi_vs_neumann.py) | MOMI vs Neumann series | MOMI: about 1e-5 after 2 iterations at every angle; Neumann: 1e-3 to 4e-2 |
| 5 | [test_05_pm_surfaces.py](test_05_pm_surfaces.py) | P-M surface generation | reproduces thesis numbers (h_rms = 0.133 m, Λ_p = 22.8 m, k·h_rms = 3.6); rms slope grows like log K_c |
| 6 | [test_06_curvature_term_pm.py](test_06_curvature_term_pm.py) | Ch. 4 result on P-M surfaces | HH at 80°: Δx-spread 1.8 dB without curvature term, 0.25 dB with it; VV much less sensitive |
| 7 | [test_07_monte_carlo_tail.py](test_07_monte_carlo_tail.py) | the project deliverable in miniature | 300 surfaces at 60°: single-realization σ is close to exponential (⟨σ²⟩/⟨σ⟩² = 2.2 HH, 2.1 VV); VV mean is 0.4 dB from Bragg theory, HH 1.7 dB above it |

### Lessons that came out of building these (worth knowing for your own solver)

* **Sign conventions bite.** The thesis uses e^{+jωt} and G = H₀⁽²⁾(kR)/(4j) = −(j/4)H₀⁽²⁾. Your proposal
  uses e^{−iωt} and Φ = (i/4)H₀⁽¹⁾. Everything is complex-conjugated between the two. My first draft
  used +j/4, which silently flipped P. Energy conservation did **not** catch it. The EFIE-vs-MFIE
  comparison did. Validate with two independent formulations, not just one.
* **Check the far-field normalization with energy conservation** on every run. For a lossless PEC,
  ∫σ⁰ dθ_s = 2π cos θ_i, minus a known finite-beam correction.
* **Rayleigh anomalies.** If a grating order lands exactly at grazing (sin θ_n = ±1), VV loses energy
  out of the angular window. Avoid exactly-resonant geometries in validation tests.
* **The incident field must solve Helmholtz** if you want formulation-independent answers. The
  Thorsos taper's error scales like 1/(k g cos θ_i)². Your proposal's main route (flat-surface total
  field + compact bump) uses an exact incident field, so it avoids this. That is a real advantage
  to state in the write-up. Caveat on test 3C: the exact beam at the same g is much wider than the
  Thorsos one, so the two runs don't light up the surface identically. The point is only that
  EFIE = MFIE when the incident field is exact.
* **TE current outlives the beam.** On a rough or periodic surface, J at the downstream edge stays at
  a few % of its peak even where the forcing is around 1e-20. Truncating there limits convergence to
  first order whatever the quadrature (test 3B). The surface must extend well past the footprint.

---

## 3. How this maps onto your proposal

| | Toporkov thesis | your proposal |
|---|---|---|
| geometry | whole rough surface, length L, tapered beam | compact bump on [−1,1] in an infinite flat plane |
| incident field | Thorsos taper (approximate) | flat-surface total field, e^{ikd·x} − e^{ikd'·x} (exact) |
| Green's function | free space G | half-space G(x,y) = Φ(x,y) − Φ(x,y'), so G = 0 on the flat part |
| unknown | J = ∂E/∂n (single-layer density, MFIE) | double-layer density φ |
| solver | MOMI | dense, then FMM |
| output | ⟨σ⁰⟩ vs angle | distribution of \|A(θ)\|², P(\|A\|² > τ) |

The proposal's validation ideas fit naturally next to these tests. "Flat Γ₀ gives u ≡ 0" is test 1
in your formulation. The point-source test plays the role of the EFIE cross-check in test 3. Test 2's
SPM comparison works for a small sinusoidal bump too.

## 4. Suggested next steps

1. Reproduce a taper failure from Ch. 5: HH backscatter vs θ_i from 75° to 88° with g below and above
   the Kapp criterion.
2. Add your double-layer, half-space formulation as a second solver here, and compare |A(θ)|² with
   this code on a small bump.
3. Grow test 7: more realizations and a larger L, then look at the tail beyond τ/⟨σ⟩ ≈ 5, where
   spikes and importance sampling come in.
