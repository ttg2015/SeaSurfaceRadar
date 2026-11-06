# Literature notes — sea clutter false-alarm probability via integral solvers

Status key: **READ** = read in the local PDF; **ABSTRACT** = only abstract/metadata seen online (full text not read, and not in the repo).

## Project in one paragraph
2D Helmholtz scattering off a random sea-like curve (PEC => Dirichlet, i.e. TE / horizontal pol), solved with a second-kind
integral equation (dense first, FMM later), Monte Carlo over random surfaces to get P(|A(theta)|^2 > tau), importance sampling
if MC is too slow. Group: Tyler Greiner, Annie Liu, Ellen Persson.

## [1] Toporkov, PhD thesis, Virginia Tech, 1998 — READ (intro, ch.2 method, ch.4 setup, ch.5 + 6 conclusions; skimmed the rest)
`dudes_dissertation.pdf` (204 pp). 1D PEC surfaces with Pierson-Moskowitz (P-M) spectrum, MFIE solved by MOMI. Closest
template for what we are doing; our proposal's Figure 1 is based on its Fig 2-2.

Takeaways that matter for us:
- **Curvature term.** Discretizing the MFIE with zeros on the propagator diagonal gives results that depend on the sampling
  interval (worst for horizontal pol at low grazing). Must include the second-derivative "curvature" term on the diagonal.
  Fix is independent of the solver. Our proposal uses a different (double-layer, second-kind) formulation, but the analogue is
  the treatment of the diagonal / near-singular quadrature, so this is a strong validation target (refine dx, check convergence).
- **Sampling.** Critical interval dx0 = pi/Kc. With Kc = 5k he used dx = lambda/10 (Kc = 4.99k to avoid sampling exactly at critical).
  Our `sea_surface_xband.py` uses dx = lambda/10, k_cut = 2.5k, i.e. already 2x oversampled.
- **Spectral cutoff (important for us).** Average backscatter NRCS is unaffected down to Kc ~ 2.5k, changes below 2k.
  But **single-realization** results keep varying until Kc ~ 3k–4k at low grazing; Kc = 4k gives only a few % rms error in the
  scattered field down to 85 deg. Our deliverable is a *tail probability of single-realization* values, so k_cut = 2.5k is likely
  too low; use >= 4k, or at least check sensitivity of the tail.
- **Surface length / spectrum resolution.** L should be >> dominant wavelength (he used L = 819.2 lambda ~ 8.3 Lambda_p, U = 5 m/s,
  lambda = 23 cm, Lambda_p = 22.8 m) so the spectral spacing dK << K_p and correlation function replicas do not overlap (L > 4 Lambda_p).
- **Spectrum is spiky at high K** (P-M ~ K^-3): mean-square slope/curvature diverge without a cutoff, so the cutoff controls slope statistics.
  Relevant to our Elfouhaily spectrum: check the slope variance against Cox-Munk (the script already prints this).
- **Taper.** Finite surface => need tapered incident field. Thorsos/Gaussian taper fails near grazing for **horizontal (TE)** pol
  (anomalous jump in backscatter NRCS + slowly decaying "tail" in the surface current); vertical (TM) shows no failure up to 86 deg.
  Our Dirichlet problem *is* the TE case, i.e. the hard one. Kapp criterion for the beam waist: `g > A / ((pi/2 - theta_i) * k * cos(theta_i))`
  [thesis eq 5.8]. He determined A = 9.4 for P-M surfaces; earlier heuristic values were smaller (Kapp ~ 2*sqrt(3), Marchand up to ~ 2*sqrt(4) —
  the PDF text extraction garbled these radicals, so verify against the PDF).
  He recommends Thorsos taper + Kapp criterion + longer surface over the "exact" (integral) taper, whose edge suppression is poor.
  Edge levels of -30 to -40 dB were OK for backscatter to ~81–84 deg.
- Our proposal's Remark 1 (Gaussian taper, theta in 80–85 deg) lines up with this chapter. The proposal's main route (compactly supported bump on a flat
  plane with the flat-surface total field as incident field) avoids tapers entirely but is a different physical setup — worth stating the tradeoff explicitly.
- Solver background: MOMI = split P = L + U, iterate with triangular solves, no matrix storage, ~1 new-Born term + 2 iterations enough.
  Competes with our "dense then FMM" plan for the large/low-grazing regime.

## [2] Thorsos 1988, JASA 83(1):78–92 — ABSTRACT
Kirchhoff approximation vs exact integral-equation solution (pressure-release / Dirichlet), Gaussian roughness spectrum. Valid region is controlled
mostly by the surface correlation length away from low grazing. This is the origin of the **Thorsos taper** and of the scattering-strength-vs-angle
curve (Fig 4) the proposal uses as the final validation target. Open question: Is the paper's integral-equation curve the same boundary condition and
spectrum as ours? Need to check before using it as ground truth. Full text not available locally.

## [3] Elfouhaily & Guerin 2004, Waves Random Media 14:R1–R40 — ABSTRACT
Review of approximate scattering theories (Kirchhoff, SPM, SSA, two-scale, etc.), ranked against a list of requirements; motivates why rigorous numerics are
needed. Good background for "why direct simulation" in the intro. Tried to download it: the ifremer mirror returns 403 and the HAL record (hal-00014924) has no direct file link, so it is not in the repo. Needs institutional access.
Same first author as the Elfouhaily 1997 sea spectrum used in `sea_surface_xband.py`.

## [4] Rino & Ngo 1998, IEEE TAP 46(1):133–141 — ABSTRACT
Numerical simulation of low-grazing-angle microwave backscatter from a deep-water *breaking-wave* profile. HH vs VV polarization selectivity comes from
an interference effect (phase dependence of crest backscatter), not Brewster angle; Doppler suggests VV returns come from slower scatterers.
Relevance: sea spikes are a *deterministic crest feature*, so the tail of our random-surface distribution may be dominated by steep/breaking-like realizations.
Gaussian linear surfaces may under-produce these; possible limitation (choppy surface in `sea_surface_1d.py` helps somewhat).

## [5] Toporkov & Sletten 2007, IEEE TGRS 45(5):1181–1197 — ABSTRACT
2D direct numerical simulation of X-band low-grazing *range-resolved* sea backscatter; Gaussian surface with P-M or Elfouhaily spectrum; amplitude
statistics compared with Weibull and K distributions; boundary-integral approach. Probably the closest published precedent to our *deliverable* (clutter
statistics -> false-alarm tails). Top priority to get the full text.

## [6] Melief et al. 2006, IEEE TGRS 44(4):985–993 — ABSTRACT
Empirical, not numerical: three coherent polarimetric sea-clutter datasets. Clutter = low-power diffuse background (tilt-modulated Bragg) + discrete
spikes with high power, high HH/VV ratio, high Doppler (steep/breaking waves, microbreakers). Use to position ours: what physical spikes exist in data that a
Gaussian-surface PEC model can or cannot reproduce.

## Things I noticed in the repo (not from the papers)
- `Electromagnetics_Final_Project.pdf` exists twice and they differ: root = Ellen's early draft ("TODO" list, loose refs), `Tyler_Folder/` = the group
  proposal (three authors, formal problem statement, refs [1]–[6]). Treat the Tyler_Folder one as current.
- Wavelength is unresolved in the proposal (Ellen's margin comment): X-band is 3 cm; the thesis uses 23 cm and 10 cm; proposal says lambda = 10 cm.
  `sea_surface_xband.py` hard-codes 3 cm (10 GHz). Decide and align.
- Proposal references [1]–[6] cited as in the thesis/related; some `[?]` placeholders remain (taper ref, wave spectrum ref).
- The sea-surface scripts model a 1D surface with an Elfouhaily spectrum + choppy (Lagrangian) displacement; Toporkov uses plain Gaussian P-M. Different
  surface statistics would shift the tail probabilities, so decide which is the base case and state the other as a sensitivity study.

## Next reading (priority)
1. Toporkov & Sletten 2007 [5] — full text.
2. Thorsos 1988 [2] — full text, esp. Fig 4 and taper derivation.
3. Elfouhaily & Guerin 2004 [3] — skim, open access.
4. Rino & Ngo [4], Melief [6] — for positioning/discussion only.
