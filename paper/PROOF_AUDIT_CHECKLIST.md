# Proof Audit Checklist (Phase 2)

Use this as an end-to-end sanity checklist after the Phase 2 patches.

## Inequalities and norms

- For every inequality, record whether it is:
  - pathwise (holds for each `(\omega,t)`), or
  - in expectation (`\E[...]`), or
  - in time-integrated mean-square (`\E\int_0^T (\cdot)^2 dt`), or
  - a conditional expectation (`\E[\cdot\mid\mathcal F_t^0]`).
- Confirm the norm matches the proof step:
  - `\|\cdot\|_2 := (\E\int_0^T|\cdot|^2dt)^{1/2}` for the continuous-time fixed-point contraction in Prop.~`prop:lqg_constructive`.
  - Sup norms (`\sup_{t\le T}`) only when BDG/Doob or deterministic bounds justify them.
- If an inequality uses “bounded coefficient”, record the bound and where it comes from (e.g., `\|A\|_\infty\le \kappa^2T/\sigma_\eta^2`).

## Existence/uniqueness and function spaces

- Every “Banach fixed point” invocation specifies:
  - a complete metric space (here `L^2_{\mathcal F^0}(\Omega\times[0,T])`), and
  - an explicit contraction constant `q<1`.
- Every ODE/SDE solution claim specifies:
  - the filtration/adaptedness, and
  - measurability/integrability conditions (e.g., coefficients bounded or square-integrable).

## Riccati/ODE sign and comparison arguments

- Any claim of invariance of a sign region (e.g., `A_t\ge 0`) is justified by:
  - a correct barrier argument (cannot cross a boundary where drift points inward), and
  - uniqueness (local Lipschitz in the state variable).
- Any “comparison” bound explicitly records the comparison ODE and the direction of inequality.

## “Brownian/independent” statements under misspecification

- When “innovation Brownian” or “independent” is stated, confirm the text specifies:
  - the measure/model under which it holds (objective vs subjective), or
  - a weaker property used in the proofs (e.g., martingale increment / quadratic covariation 0).
- Confirm no downstream proof step uses objective independence if only subjective independence is asserted.

## Constants

- For every prefactor `C` in an exponential bound, check:
  - it is valid over the full parameter regime claimed, and
  - it handles sign cases correctly (e.g., Lemma `lem:riccati_convergence` for `\Sigma_0>\Sigma^*`).

