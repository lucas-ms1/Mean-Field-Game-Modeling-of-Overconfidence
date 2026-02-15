"""Belief updating (Kalman filter for a 1D random-walk fundamental)."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .types import AgentBeliefState


def update_belief(
    prev_state: AgentBeliefState,
    observation: float,
    params: Dict[str, float],
) -> AgentBeliefState:
    """Update belief using a 1D Kalman filter.

    Model (discrete time, step size dt):
      v_{t+1} = v_t + mu*dt + eps_v,   eps_v ~ N(0, sigma_v^2 * dt)
      y_t     = v_t + eps_y,           eps_y ~ N(0, sigma_common^2 + sigma_idio^2)

    Overconfidence enters as misperceived observation noise:
      perceived variance is sigma_common^2 + sigma_idio^2 / k (k >= 1).

    Note: if you construct y_t as the normalized increment y_t = (xi_{t+dt}-xi_t)/dt from a signal SDE
    dxi_t = v_t dt + sigma_common dU_t + sigma_idio dB_t, then the effective observation noise scales as
    1/sqrt(dt); i.e., pass sigma_common/sqrt(dt) and sigma_idio/sqrt(dt) here.

    Falls back to a fixed-gain update if params contains "gain".
    """
    if "gain" in params:
        gain = float(params.get("gain", 0.1))
        new_mean = prev_state.v_hat + gain * (observation - prev_state.v_hat)
        new_var = max(prev_state.Sigma * (1.0 - gain), 1e-6)
        return replace(prev_state, v_hat=float(new_mean), Sigma=float(new_var))

    mu = float(params.get("mu", 0.0))
    sigma_v = float(params.get("sigma_v", 0.0))
    sigma_obs = float(params.get("sigma_obs", 0.1))
    sigma_common = float(params.get("sigma_common", 0.0))
    sigma_idio = float(params.get("sigma_idio", sigma_obs))
    k = float(params.get("k", 1.0))
    dt = float(params.get("dt", 1.0))

    k = max(k, 1e-9)
    q = (sigma_v**2) * dt
    # True observation variance includes a common component and an idiosyncratic component.
    # Overconfidence scales only the idiosyncratic noise: perceived variance is
    #   sigma_common^2 + sigma_idio^2 / k.
    r_true = (sigma_common**2 + sigma_idio**2)
    r_perceived = max(sigma_common**2 + (sigma_idio**2) / k, 1e-12)

    mean_pred = prev_state.v_hat + mu * dt
    var_pred = max(prev_state.Sigma + q, 1e-12)

    kalman_gain = var_pred / (var_pred + r_perceived)
    innovation = observation - mean_pred
    mean_post = mean_pred + kalman_gain * innovation
    var_post = (1.0 - kalman_gain) * var_pred

    return replace(prev_state, v_hat=float(mean_post), Sigma=float(max(var_post, 1e-12)))
