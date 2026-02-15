# Math Fixes (Phase 2) — Changelog

This changelog maps each Phase 2 patch to the original location/claim.

## Fix 11 — Fixed-point space for the map `\Phi`

- Location: `sections/lqg_constructive_solution.tex` (Prop.~`prop:lqg_constructive`, “Fixed-point object”, and proof).
- Original: `\mathcal X := L^\infty_{\mathcal F^0}(\Omega\times[0,T])` with `\|\cdot\|_\infty`.
- Patch: Replace with `\mathcal X := L^2_{\mathcal F^0}(\Omega\times[0,T])` and `\|\bar x\|_2 := (\E\int_0^T|\bar x_t|^2dt)^{1/2}`; rewrite the Lipschitz steps for `p`, `\bar y`, and `B` in the `L^2` norm so the contraction argument matches the mean-square convergence style used elsewhere.
- Downstream text: `sections/sec4_results.tex` stability summary updated to reference `(\mathcal X,\|\cdot\|_2)`.

## Fix 12–13 — Riccati sign/invariance in Prop.~3.7 (Step 1)

- Location: `sections/lqg_constructive_solution.tex` (Prop.~`prop:lqg_constructive`, Step 1).
- Original: Nonnegativity attempted via the invalid inequality “if `\widetilde A_s<0` then `\dot{\widetilde A}_s\ge \kappa^2/\sigma_\eta^2`”.
- Patch: Replace with a correct barrier/uniqueness argument: the RHS is locally Lipschitz; since `f(s,0)=\kappa^2/\sigma_\eta^2>0`, the solution starting at 0 becomes positive immediately and cannot cross into negatives.

## Fix 27 — Lemma A.1 prefactor (handles `\Sigma_0>\Sigma^*`)

- Location: `sections/appx.tex` (Lemma `lem:riccati_convergence`).
- Original: From `\Sigma_t-\Sigma^*=-2\Sigma^*\rho e^{-2at}/(1+\rho e^{-2at})` the bound `|\Sigma_t-\Sigma^*|\le 2\Sigma^*|\rho|e^{-2at}` was claimed; also the derivation used `\operatorname{artanh}(\Sigma_0/\Sigma^*)`.
- Patch:
  - Derive the closed form via the ratio ODE `\rho_t=(1-u_t)/(1+u_t)` (valid for all `\Sigma_0\ge 0`), avoiding `\operatorname{artanh}` domain issues.
  - Replace the bound with a correct piecewise prefactor:
    - if `\rho\ge 0` (equivalently `\Sigma_0\le \Sigma^*`), `|\Sigma_t-\Sigma^*|\le 2\Sigma^*|\rho|e^{-2at}`;
    - if `\rho<0` (equivalently `\Sigma_0>\Sigma^*`), `|\Sigma_t-\Sigma^*|\le \frac{2\Sigma^*|\rho|}{1-|\rho|}e^{-2at}`.

## Fix 5/16 — “not necessarily monotone” tracking-error narrative

- Location: `sections/sec4_results.tex` (Prop.~`prop:tracking_error`) and `sections/sec5_proofs.tex` (proof of Prop.~`prop:tracking_error`).
- Original: “trade-off (not necessarily monotone in `k`)”.
- Patch: Replace with the correct monotonicity statement and add a short derivative argument in the proof: viewing `\Var(e_\infty)` as `g(K)=\frac{\sigma_c^2}{2}K+\frac{\sigma_v^2}{2K}` and using that `K^*(k)` is increasing in `k` and (if `\sigma_c>0`) satisfies `K^*(k)\le \sigma_v/\sigma_c`, implies `\Var(e_\infty)` is strictly decreasing in `k`.

## Fix 8 — Innovation independence under misspecification

- Location: `sections/lqg_constructive_solution.tex` (innovation-form dynamics).
- Original: `I^p` and `I^\xi` stated as “independent Brownian motions” without specifying the modeling measure/filtration.
- Patch: Qualify the statement as holding under the agent’s subjective filtering model (Appendix A); no downstream proof uses objective innovation independence.

