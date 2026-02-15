"""Main simulation loop for the scaffold."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np

from .belief import update_belief
from .confidence import update_k
from .logging import Logger
from .market import aggregate_demand, anchored_impact, price_impact
from .memory import compute_salience, update_ema
from .price_model import PriceTransitionModel
from .replay import ReplayBuffer
from .rl import ActorCriticLinearGaussian
from .types import AgentBeliefState, AgentInternalState, MarketState, SimState, SimulationOutput, Transition, state_vector

def _steady_state_posterior_variance_random_walk(q: float, r: float) -> float:
    """Steady-state posterior variance for a 1D random-walk KF.

    Model: v_{t+1}=v_t+eps, eps~N(0,q); y_t=v_t+noise, noise~N(0,r).
    The steady-state posterior variance P solves P^2 + q P - q r = 0.
    """
    q = max(float(q), 0.0)
    r = max(float(r), 0.0)
    if q <= 0.0:
        return 0.0
    disc = q * q + 4.0 * q * r
    return max(0.5 * (-q + math.sqrt(max(disc, 0.0))), 0.0)


def _compute_intertemporal_A_t(
    t: int,
    horizon: int,
    dt: float,
    kappa: float,
    sigma_eta: float,
    sigma_v: float,
    sigma_c: float,
    sigma_eps: float,
    k: float,
) -> float:
    """
    Compute A_t using constant-gain approximation (paper eq:A_closed_form).
    
    A_t ≈ (κ/(σ_η β*)) * tanh(κ β* (T-t) / σ_η)
    where β* = K* * sqrt(σ_c² + σ_ε²) and K* = σ_v / sqrt(σ_c² + σ_ε²/k)
    """
    # Perceived measurement variance
    R_perc = sigma_c**2 + sigma_eps**2 / max(k, 1e-9)
    # Constant-gain Kalman gain
    K_star = sigma_v / max(math.sqrt(R_perc), 1e-9)
    # True measurement variance
    R_true = sigma_c**2 + sigma_eps**2
    # Innovation volatility
    beta_star = K_star * math.sqrt(R_true)
    
    # Time to horizon
    tau = max((horizon - t) * dt, 1e-9)
    
    # Avoid division by zero
    if beta_star < 1e-12 or sigma_eta < 1e-12:
        return 0.0
    
    # Compute A_t using closed-form formula
    arg = (kappa * beta_star * tau) / sigma_eta
    A_t = (kappa / (sigma_eta * beta_star)) * math.tanh(arg)
    
    return max(A_t, 0.0)  # Ensure non-negative


def simulate(
    horizon: int,
    num_agents: int,
    params: Dict[str, Any],
) -> SimulationOutput:
    seed = int(params.get("seed", 0))
    rng = random.Random(seed)
    log_metrics = bool(params.get("log_metrics", True))
    dt = float(params.get("dt", 1.0))
    dt = max(dt, 1e-9)
    sqrt_dt = math.sqrt(dt)
    inv_sqrt_dt = 1.0 / sqrt_dt
    chi = max(float(params.get("chi", 0.0)), 0.0)
    logger = Logger()
    replay = ReplayBuffer(
        capacity=int(params.get("replay_capacity", 1000)),
        h0=params.get("h0", 5.0),
        h1=params.get("h1", 5.0),
        rho_pos=params.get("rho_pos", 0.0),
        rho_neg=params.get("rho_neg", 0.0),
        seed=rng.randint(0, 10_000),
    )
    price_model_cfg = params.get("price_model", None)
    planning_cfg = params.get("planning", None)
    memory_cfg = params.get("memory", None)
    confidence_cfg = params.get("confidence", None)
    rl_cfg = params.get("rl", None)
    market_cfg = params.get("market", None)
    rl_enabled = rl_cfg is not None and getattr(rl_cfg, "enabled", False)

    agents: List[AgentInternalState] = []
    acs: List[ActorCriticLinearGaussian] = []
    price_models: List[PriceTransitionModel] = []

    rl_rng = random.Random(int(getattr(rl_cfg, "seed", 0))) if rl_cfg else random.Random(0)
    for _ in range(num_agents):
        belief = AgentBeliefState(v_hat=0.0, Sigma=1.0)
        # Initialize overconfidence k (can be heterogeneous).
        k_init = float(params.get("k_init", 1.0))
        k_min = float(params.get("k_min", 1.0))
        k_max = float(params.get("k_max", 3.0))
        k_min, k_max = (k_min, k_max) if k_min <= k_max else (k_max, k_min)
        k_dist = str(params.get("k_dist", "fixed")).lower()
        k_std = float(params.get("k_std", 0.0))
        if k_dist == "uniform":
            k0 = rng.uniform(k_min, k_max)
        elif k_dist == "normal":
            # Truncated normal (via clipping) to avoid negative/extreme values.
            k0 = rng.gauss(k_init, max(k_std, 1e-12))
            k0 = float(np.clip(k0, k_min, k_max))
        else:
            k0 = k_init
        agents.append(
            AgentInternalState(
                belief=belief,
                k=k0,
                u=0.0,
                x=0.0,
                wealth=0.0,
            )
        )
        if rl_cfg:
            np_rng = np.random.default_rng(rl_rng.randint(0, 2**31 - 1))
            acs.append(ActorCriticLinearGaussian(dim=7, cfg=rl_cfg, rng=np_rng))  # 7 features including regime_post
        if price_model_cfg and getattr(price_model_cfg, "enabled", False):
            price_models.append(PriceTransitionModel(config=price_model_cfg))

    price = params.get("price_init", 0.0)
    prev_price = price
    fundamental = params.get("fundamental_init", 0.0)
    k_series: List[List[float]] = []
    ema_gs = [0.0 for _ in range(num_agents)]
    
    # Amplification extensions: configuration
    amplification_cfg = params.get("amplification", {})
    state_dependent_kappa = bool(amplification_cfg.get("state_dependent_kappa", False))
    kappa_decay_rate = float(amplification_cfg.get("kappa_decay_rate", 0.1))
    endogenous_fundamental_feedback = bool(amplification_cfg.get("endogenous_fundamental_feedback", False))
    feedback_strength = float(amplification_cfg.get("feedback_strength", 0.05))
    overconfidence_dependent_impact = bool(amplification_cfg.get("overconfidence_dependent_impact", False))
    impact_sensitivity = float(amplification_cfg.get("impact_sensitivity", 0.2))
    volatility_dependent_gamma = bool(amplification_cfg.get("volatility_dependent_gamma", False))
    gamma_volatility_sensitivity = float(amplification_cfg.get("gamma_volatility_sensitivity", 0.1))
    
    # Volatility tracking for volatility-dependent risk aversion
    realized_volatility = 0.0
    price_history: List[float] = [price]
    volatility_window = 20  # Rolling window for volatility estimation
    
    # Regime break parameters
    experiment_cfg = params.get("experiment", None)
    regime_break_enabled = experiment_cfg is not None and getattr(experiment_cfg, "regime_break_enabled", False)
    t_break = getattr(experiment_cfg, "t_break", 0) if experiment_cfg else 0
    
    # Dynamic parameters (updated at regime break)
    current_kappa = market_cfg.kappa if market_cfg else 0.05
    current_impact = market_cfg.impact if market_cfg else 0.01
    current_fundamental_sigma = params.get("fundamental_sigma", 0.1)
    current_observation_sigma = params.get("observation_sigma", 0.1)
    # Pre-generate idiosyncratic observation shocks so that (for fixed seed) the first agents'
    # observation paths are identical across different N. This reduces Monte-Carlo noise in
    # N-convergence checks (common random numbers for the shared subpopulation).
    obs_idio_z = np.random.default_rng(seed + 123_457).standard_normal(size=(num_agents, horizon))
    risk_variance_mode = str(params.get("risk_variance_mode", "belief")).lower()
    risk_variance_sigma_value = float(params.get("risk_variance_sigma_value", 0.0))
    # Optional ablation: freeze Sigma only in the risk denominator of the one-step demand rule.
    # This keeps belief dynamics (the Kalman recursion) unchanged.
    risk_sigma_fixed_by_agent: List[float] | None = None
    if risk_variance_mode in ("fixed_initial", "fixed_steady_state", "fixed_value"):
        sigma_common0 = float(params.get("observation_common_sigma", 0.0))
        sigma_idio0 = float(params.get("observation_sigma", current_observation_sigma))
        sigma_v0 = float(params.get("fundamental_sigma", current_fundamental_sigma))
        q0 = (sigma_v0**2) * dt
        sigma_common0_eff = sigma_common0 * inv_sqrt_dt
        sigma_idio0_eff = sigma_idio0 * inv_sqrt_dt
        risk_sigma_fixed_by_agent = []
        for a in agents:
            if risk_variance_mode == "fixed_initial":
                risk_sigma_fixed_by_agent.append(float(a.belief.Sigma))
            elif risk_variance_mode == "fixed_steady_state":
                r_perceived0 = max(
                    sigma_common0_eff**2 + (sigma_idio0_eff**2) / max(float(a.k), 1e-9),
                    0.0,
                )
                risk_sigma_fixed_by_agent.append(_steady_state_posterior_variance_random_walk(q=q0, r=r_perceived0))
            else:  # fixed_value
                risk_sigma_fixed_by_agent.append(float(risk_variance_sigma_value))

    for t in range(horizon):
        # Apply regime break at t_break
        if regime_break_enabled and t == t_break:
            if experiment_cfg:
                if getattr(experiment_cfg, "post_kappa", None) is not None:
                    current_kappa = experiment_cfg.post_kappa
                if getattr(experiment_cfg, "post_impact", None) is not None:
                    current_impact = experiment_cfg.post_impact
                if getattr(experiment_cfg, "post_sigma_v", None) is not None:
                    current_fundamental_sigma = experiment_cfg.post_sigma_v
                if getattr(experiment_cfg, "post_obs_noise", None) is not None:
                    current_observation_sigma = experiment_cfg.post_obs_noise
            
            # Reset price model covariance on break (if enabled)
            if price_model_cfg and getattr(price_model_cfg, "reset_on_break", False):
                for model in price_models:
                    model.reset_covariance()
            
            # Reset critic weights on break (if enabled)
            if rl_cfg and getattr(rl_cfg, "reset_critic_on_break", False):
                for ac in acs:
                    ac.reset_critic()
        
        regime_post = regime_break_enabled and t >= t_break
        p_t = price
        mu_v = float(params.get("fundamental_mu", 0.0))
        
        # Endogenous fundamental feedback: dv_t = μ_v dt + σ_v dW_t + β_feedback * (p_t - v_t) dt
        fundamental_drift = mu_v * dt
        if endogenous_fundamental_feedback:
            mispricing_feedback = feedback_strength * (p_t - fundamental) * dt
            fundamental_drift += mispricing_feedback
        
        fundamental += fundamental_drift + rng.gauss(0.0, 1.0) * (current_fundamental_sigma * (dt**0.5))
        
        # Update realized volatility (rolling window)
        price_history.append(p_t)
        if len(price_history) > volatility_window:
            price_history.pop(0)
        if len(price_history) > 1:
            returns = np.diff(price_history)
            realized_volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
        demands: List[float] = []
        actions: List[float] = []
        a_terms: List[float] = []
        b_terms: List[float] = []
        myopic_terms: List[float] = []
        hedging_terms: List[float] = []
        vhat_xs: List[float] = []
        sigma_xs: List[float] = []
        saliences: List[float] = []
        correctness_flags: List[float] = []
        next_agents: List[AgentInternalState] = []
        rewards: List[float] = []
        mispricings: List[float] = []
        rho_xs: List[float] = []
        states_for_action: List[AgentInternalState] = []

        mean_belief = sum(a.belief.v_hat for a in agents) / max(len(agents), 1)
        k_mean = sum(a.k for a in agents) / max(len(agents), 1)
        bar_x_pre = sum(a.x for a in agents) / max(len(agents), 1)

        sigma_common = float(params.get("observation_common_sigma", 0.0))
        # Under the normalized increment y=(xi_{t+dt}-xi_t)/dt, observation noise scales with 1/sqrt(dt).
        sigma_common_eff = sigma_common * inv_sqrt_dt
        sigma_idio_eff = float(current_observation_sigma) * inv_sqrt_dt
        common_obs = rng.gauss(0.0, sigma_common_eff)
        use_price_in_filter = bool(params.get("use_price_in_filter", False))
        price_obs_sigma = float(params.get("price_obs_sigma", 1.0))

        for agent_id, agent in enumerate(agents):
            observation = fundamental + common_obs + float(obs_idio_z[agent_id, t]) * sigma_idio_eff
            new_belief = update_belief(
                agent.belief,
                observation,
                {
                    "mu": mu_v,
                    "sigma_v": current_fundamental_sigma,
                    "sigma_common": sigma_common_eff,
                    "sigma_idio": sigma_idio_eff,
                    "k": agent.k,
                    "dt": dt,
                },
            )
            vhat_xs.append(float(new_belief.v_hat))
            sigma_xs.append(float(new_belief.Sigma))

            correctness = 1.0 if abs(new_belief.v_hat - fundamental) < abs(agent.belief.v_hat - fundamental) else -1.0
            if memory_cfg and memory_cfg.salience_mode == "return":
                g = abs(p_t - prev_price)
            else:
                g = abs(agent.x * (p_t - prev_price))
            ema_alpha = memory_cfg.ema_alpha if memory_cfg else 0.2
            eps = memory_cfg.eps if memory_cfg else 1.0e-6
            ema_gs[agent_id] = update_ema(ema_gs[agent_id], g, ema_alpha)
            salience_raw = compute_salience(g=g, ema_g=ema_gs[agent_id], eps=eps)
            memory_enabled = True if confidence_cfg is None else bool(confidence_cfg.memory_enabled)
            salience = salience_raw if memory_enabled else 0.0
            h0 = memory_cfg.h0 if memory_cfg else 5.0
            h1 = memory_cfg.h1 if memory_cfg else 5.0
            h1_effective = h1 if memory_enabled else 0.0
            new_k, new_conf_u = update_k(
                prev_k=agent.k,
                prev_u=agent.u,
                correctness=correctness,
                salience=salience,
                k_mean=k_mean,
                params={
                    "k_min": params.get("k_min", 0.5),
                    "k_max": params.get("k_max", 3.0),
                    "k_bar": confidence_cfg.k_bar if confidence_cfg else 1.0,
                    "psi": confidence_cfg.psi if confidence_cfg else 1.0,
                    "alpha_u": confidence_cfg.alpha_u if confidence_cfg else 0.1,
                    "lambda_herd": confidence_cfg.lambda_herd if confidence_cfg else 0.0,
                    "h0": h0,
                    "h1": h1_effective,
                },
            )
            # Preserve initial heterogeneous k when comparing distributions at fixed mean (no confidence drift)
            if str(params.get("k_dist", "fixed")).lower() in ("uniform", "normal"):
                new_k = agent.k

            mispricing = new_belief.v_hat - p_t
            state_for_action = AgentInternalState(
                belief=new_belief,
                k=new_k,
                u=new_conf_u,
                x=agent.x,
                wealth=agent.wealth,
            )
            market_state = MarketState(
                p=p_t,
                p_prev=prev_price,
                v_true=fundamental,
                aggregate_demand=0.0,
                regime_post=regime_post,
            )

            # Action selection happens before demand aggregation.
            if rl_enabled:
                s_vec = state_vector(state_for_action, market_state)
                demand = acs[agent_id].act(s_vec)
                if rl_cfg and rl_cfg.action_clip > 0.0:
                    demand = float(np.clip(demand, -rl_cfg.action_clip, rl_cfg.action_clip))
                demands.append(demand)
                actions.append(demand)
            else:
                # Myopic CARA best-response under Gaussian price increments (paper Eq. (14)),
                # solved in closed form for the mean-field consistency at each step.
                # Optionally augmented with intertemporal hedging term (paper eq:opt_control_lqg).
                sigma_p2 = float(params.get("sigma_p2", 1.0))
                gamma_base = float(params.get("gamma", 1.0))
                
                # Volatility-dependent risk aversion: γ_eff = γ_base * (1 + β_γ * realized_volatility)
                gamma = gamma_base
                if volatility_dependent_gamma:
                    gamma = gamma_base * (1.0 + gamma_volatility_sensitivity * realized_volatility)
                    gamma = max(gamma, gamma_base * 0.1)  # Floor at 10% of base
                
                alpha0 = float(params.get("alpha_0", 1.0))
                use_intertemporal = params.get("use_intertemporal_policy", False)
                # Discrete-time risk adjustment: uncertainty in v translates into uncertainty in the next price move
                # through the anchoring term kappa * (v - p). Using belief variance increases the sensitivity of
                # overconfidence (lower perceived Sigma -> larger demand).
                sigma_for_risk = float(new_belief.Sigma)
                if risk_sigma_fixed_by_agent is not None:
                    sigma_for_risk = float(risk_sigma_fixed_by_agent[agent_id])
                eff_var = max(sigma_p2 + (current_kappa**2) * sigma_for_risk * dt, 1e-9)
                rho_xs.append((current_kappa**2) * sigma_for_risk * dt / max(sigma_p2, 1e-9))
                inv_denom = alpha0 / max(chi + gamma * eff_var, 1e-9)
                myopic_i = inv_denom * current_kappa * mispricing
                b_i = inv_denom * float(current_impact)
                hedging_i = 0.0
                a_i = myopic_i
                
                # Add intertemporal hedging correction if enabled (paper eq:opt_control_lqg)
                if use_intertemporal:
                    # Use noise_sigma from market config, fallback to sqrt(sigma_p2)
                    if market_cfg:
                        sigma_eta = max(float(market_cfg.noise_sigma), 1e-9)
                    else:
                        sigma_eta = max(float(params.get("sigma_p2", 1.0)) ** 0.5, 1e-9)
                    sigma_v = current_fundamental_sigma
                    sigma_c = float(params.get("observation_common_sigma", 0.0))
                    sigma_eps = current_observation_sigma
                    A_t = _compute_intertemporal_A_t(
                        t=t,
                        horizon=horizon,
                        dt=dt,
                        kappa=current_kappa,
                        sigma_eta=sigma_eta,
                        sigma_v=sigma_v,
                        sigma_c=sigma_c,
                        sigma_eps=sigma_eps,
                        k=agent.k,
                    )
                    # Intertemporal hedging term: (A_t * y_i) / gamma, where y_i = mispricing
                    hedging_i = (A_t * mispricing) / max(gamma, 1e-9)
                    a_i = a_i + hedging_i
                
                myopic_terms.append(myopic_i)
                hedging_terms.append(hedging_i)
                demand = a_i  # temporary; corrected after mean-field closure
                demands.append(demand)
                actions.append(demand)
                a_terms.append(a_i)
                b_terms.append(b_i)
            states_for_action.append(state_for_action)

            next_state = AgentInternalState(
                belief=new_belief,
                k=new_k,
                u=new_conf_u,
                x=demand,
                wealth=agent.wealth,
            )
            next_agents.append(next_state)
            saliences.append(salience)
            correctness_flags.append(correctness)
            mispricings.append(mispricing)

        # Apply stock–flow mean-field closure for the non-RL policy:
        # x_{i,t} = a_{i,t} + b_{i,t} * \bar{u}_t, with \bar{u}_t = (\bar{x}_t - \bar{x}_{t^-})/dt.
        if not rl_enabled:
            mean_A = float(sum(a_terms) / max(len(a_terms), 1)) if a_terms else 0.0
            mean_B = float(sum(b_terms) / max(len(b_terms), 1)) if b_terms else 0.0
            denom_fp = 1.0 - mean_B / dt
            if denom_fp > 1e-9:
                bar_x = (mean_A - (mean_B / dt) * bar_x_pre) / denom_fp
            else:
                # Fallback: if the per-step fixed point fails (mean_B >= dt), freeze inventories.
                bar_x = bar_x_pre
            bar_u = (bar_x - bar_x_pre) / dt
            demands = [a_terms[j] + b_terms[j] * bar_u for j in range(len(a_terms))]
            actions = list(demands)
            for j in range(len(next_agents)):
                next_agents[j] = AgentInternalState(
                    belief=next_agents[j].belief,
                    k=next_agents[j].k,
                    u=next_agents[j].u,
                    x=demands[j],
                    wealth=next_agents[j].wealth,
                )

        # Cross-sectional diagnostics at decision time (before rewards)
        xs_action_std = float(np.std(np.array(actions, dtype=float))) if actions else 0.0
        xs_action_mean_abs = float(np.mean(np.abs(np.array(actions, dtype=float)))) if actions else 0.0
        xs_belief_std = float(np.std(np.array(vhat_xs, dtype=float))) if vhat_xs else 0.0
        xs_sigma_mean = float(np.mean(np.array(sigma_xs, dtype=float))) if sigma_xs else 0.0
        xs_rho_mean = float(np.mean(np.array(rho_xs, dtype=float))) if rho_xs else 0.0
        xs_rho_max = float(np.max(np.array(rho_xs, dtype=float))) if rho_xs else 0.0

        # Trading rates implied by inventory changes (flow-based impact input)
        trade_rates = [(demands[j] - states_for_action[j].x) / dt for j in range(len(demands))]
        xs_flow_std = float(np.std(np.array(trade_rates, dtype=float))) if trade_rates else 0.0
        xs_flow_mean_abs = float(np.mean(np.abs(np.array(trade_rates, dtype=float)))) if trade_rates else 0.0
        total_flow = aggregate_demand(trade_rates)
        total_inventory = aggregate_demand(demands)
        bar_u = float(total_flow) / max(num_agents, 1)
        bar_x = float(total_inventory) / max(num_agents, 1)

        if market_cfg and market_cfg.price_rule == "anchored_impact":
            anchor = fundamental if fundamental is not None else mean_belief
            noise_sigma = float(market_cfg.noise_sigma) * (1.0 + float(getattr(market_cfg, "noise_vol_scale", 0.0)) * xs_flow_mean_abs)
            price = anchored_impact(
                price=p_t,
                anchor=anchor,
                total_demand=total_flow,
                num_agents=num_agents,
                kappa=current_kappa,
                impact=current_impact,
                noise_sigma=noise_sigma,
                rng=rng,
                dt=dt,
                state_dependent_kappa=state_dependent_kappa,
                kappa_decay_rate=kappa_decay_rate,
                overconfidence_dependent_impact=overconfidence_dependent_impact,
                impact_sensitivity=impact_sensitivity,
                k_mean=k_mean,
            )
        else:
            price = price_impact(total_flow, p_t, params.get("lambda_price", 0.1), dt=dt)

        # Optional robustness: incorporate a price-consistent observation of v_t.
        # Under the discrete-time price update p_{t+1} = p_t + kappa*(v_t-p_t)*dt + impact*bar_u*dt + sigma_eta*sqrt(dt)*eps,
        # we can form v_obs = p_t + (p_{t+1}-p_t-impact*bar_u*dt)/(kappa*dt) = v_t + (sigma_eta/(kappa*sqrt(dt)))*eps.
        if use_price_in_filter:
            denom = max(current_kappa * dt, 1e-12)
            v_obs = p_t + (price - p_t - float(current_impact) * bar_u * dt) / denom
            sigma_eta = float(market_cfg.noise_sigma) if market_cfg else float(params.get("sigma_eta", 0.0))
            sigma_v_obs = price_obs_sigma * (sigma_eta / max(current_kappa * (dt**0.5), 1e-12))
            for j in range(len(next_agents)):
                b = next_agents[j].belief
                next_agents[j] = AgentInternalState(
                    belief=update_belief(
                        b,
                        v_obs,
                        {"mu": 0.0, "sigma_v": 0.0, "sigma_common": 0.0, "sigma_idio": sigma_v_obs, "k": 1.0, "dt": 0.0},
                    ),
                    k=next_agents[j].k,
                    u=next_agents[j].u,
                    x=next_agents[j].x,
                    wealth=next_agents[j].wealth,
                )
        market_state = MarketState(
            p=p_t,
            p_prev=prev_price,
            v_true=fundamental,
            aggregate_demand=total_flow,
            regime_post=regime_post,
        )
        market_state_next = MarketState(
            p=price,
            p_prev=p_t,
            v_true=fundamental,
            aggregate_demand=total_flow,
            regime_post=regime_post,
        )

        real_deltas: List[float] = []
        mean_correctness = sum(correctness_flags) / max(len(correctness_flags), 1) if correctness_flags else 0.0
        for agent_id, agent in enumerate(agents):
            state = SimState(agent=states_for_action[agent_id], market=market_state)
            # Compute reward based on reward_mode
            if rl_cfg and rl_cfg.reward_mode == "expected_utility":
                # Expected utility reward using decision-time belief
                mu_hat = state.agent.belief.v_hat
                var_hat = state.agent.belief.Sigma if rl_cfg.use_belief_var else 0.0
                x = actions[agent_id]
                p_t = state.market.p
                x_sq = x * x
                if not math.isfinite(x_sq):
                    x_sq = 1e300
                reward = x * (mu_hat - p_t) - (rl_cfg.gamma / 2.0) * x_sq * (rl_cfg.sigma_p2 + var_hat)
                reward -= 0.5 * chi * x_sq * dt
                if not math.isfinite(reward):
                    reward = -1e300 if reward < 0 else 1e300
            else:
                # Realized PnL reward (default)
                reward = actions[agent_id] * (price - p_t)
                if rl_cfg:
                    a_sq = actions[agent_id] * actions[agent_id]
                    if not math.isfinite(a_sq):
                        a_sq = 1e300
                    reward -= (rl_cfg.gamma / 2.0) * a_sq * rl_cfg.sigma_p2
                a_sq = actions[agent_id] * actions[agent_id]
                if not math.isfinite(a_sq):
                    a_sq = 1e300
                reward -= 0.5 * chi * a_sq * dt
                if not math.isfinite(reward):
                    reward = -1e300 if reward < 0 else 1e300
            rewards.append(reward)
            next_agents[agent_id] = AgentInternalState(
                belief=next_agents[agent_id].belief,
                k=next_agents[agent_id].k,
                u=next_agents[agent_id].u,
                x=next_agents[agent_id].x,
                wealth=next_agents[agent_id].wealth + reward,
            )
            next_state = SimState(agent=next_agents[agent_id], market=market_state_next)
            phi = PriceTransitionModel.features(p_t, mean_belief, total_flow, regime_post).tolist()
            transition = Transition(
                t=t,
                agent_id=agent_id,
                state=state,
                action=actions[agent_id],
                reward=reward,
                next_state=next_state,
                info={
                    "salience": saliences[agent_id],
                    "mispricing": mispricings[agent_id],
                    "correctness": correctness_flags[agent_id],
                    "t": float(t),
                    "m_t": float(mean_belief),
                    "D_t": float(total_flow),
                    "phi_t": phi,
                },
            )
            replay.add(transition=transition, weight=saliences[agent_id] + 1e-6)
            if rl_enabled:
                diag = acs[agent_id].update_from_transition(transition, synthetic=False)
                real_deltas.append(abs(diag["delta"]))

        # TODO: model-based price learning happens after price update.
        if price_model_cfg and getattr(price_model_cfg, "enabled", False):
            for agent_id, model in enumerate(price_models):
                if getattr(price_model_cfg, "salience_weighting", False):
                    weight = 1.0 + float(price_model_cfg.w_scale) * saliences[agent_id]
                else:
                    weight = 1.0
                model.update(p_t, mean_belief, total_flow, price, weight=weight, regime_post=regime_post)

        mean_salience = sum(saliences) / max(len(saliences), 1) if saliences else 0.0
        logger.log_step(price=price, fundamental=fundamental, mean_belief=mean_belief, total_demand=total_flow)
        prev_price = p_t
        agents = next_agents
        k_series.append([a.k for a in agents])

        # TODO: replay sampling happens after transition logging.
        if (planning_cfg and getattr(planning_cfg, "enabled", False)) and log_metrics:
            planning_deltas: List[float] = []
            
            # Pre-break gate: skip planning entirely before regime break if only_post_break is enabled
            if regime_break_enabled and getattr(planning_cfg, "only_post_break", False) and t < t_break:
                # Skip planning pre-break when only_post_break is enabled
                pass
            else:
                # Post-break warmup: skip planning during initial adaptation window
                warmup_steps = getattr(planning_cfg, "warmup_steps", 0)
                if regime_break_enabled and t_break <= t < t_break + warmup_steps:
                    # Skip planning during warmup period after regime break
                    pass
                else:
                    # Recency gating: compute min_t for sampling
                    recent_window = getattr(planning_cfg, "recent_window", 150)
                    min_t = t - recent_window
                    if regime_break_enabled and t_break <= t <= t_break + getattr(planning_cfg, "gate_after_break", 200):
                        min_t = max(min_t, t_break)  # Do not sample pre-break transitions during adaptation window
                    
                    batch = replay.sample(batch_size=int(planning_cfg.K), t=t, rng=rng, min_t=min_t)
                    for tr in batch:
                        if not getattr(planning_cfg, "use_model", True):
                            continue
                        model = price_models[tr.agent_id] if price_models else None
                        if model is None:
                            continue
                        m_t = float(tr.info.get("m_t", mean_belief))
                        D_t = float(tr.info.get("D_t", total_flow))
                        tr_regime_post = tr.state.market.regime_post if hasattr(tr.state.market, "regime_post") else regime_post
                        p_pred_next = model.predict(tr.state.market.p, m_t, D_t, regime_post=tr_regime_post)
                        market_hat = MarketState(
                            p=p_pred_next,
                            p_prev=tr.state.market.p,
                            v_true=tr.state.market.v_true,
                            aggregate_demand=D_t,
                            regime_post=regime_post,  # Use current regime_post for next state
                        )
                        next_hat = SimState(agent=tr.next_state.agent, market=market_hat)
                        
                        # Planning reward must match rl.reward_mode
                        if rl_cfg and rl_cfg.reward_mode == "expected_utility":
                            # Use decision-time belief (from tr.state), NOT p_pred_next
                            mu_hat = tr.state.agent.belief.v_hat
                            var_hat = tr.state.agent.belief.Sigma if rl_cfg.use_belief_var else 0.0
                            r_hat = tr.action * (mu_hat - tr.state.market.p)
                            a_sq = tr.action * tr.action
                            if not math.isfinite(a_sq):
                                a_sq = 1e300
                            r_hat -= (rl_cfg.gamma / 2.0) * a_sq * (rl_cfg.sigma_p2 + var_hat)
                        else:
                            # realized_pnl: reward based on predicted price change
                            r_hat = tr.action * (p_pred_next - tr.state.market.p)
                            if rl_cfg:
                                a_sq = tr.action * tr.action
                                if not math.isfinite(a_sq):
                                    a_sq = 1e300
                                r_hat -= (rl_cfg.gamma / 2.0) * a_sq * rl_cfg.sigma_p2
                        tr_hat = Transition(
                            t=tr.t,
                            agent_id=tr.agent_id,
                            state=tr.state,
                            action=tr.action,
                            reward=r_hat,
                            next_state=next_hat,
                            info=tr.info,
                        )
                        if rl_enabled:
                            update_actor = getattr(planning_cfg, "update_actor", True)
                            lr_scale = getattr(planning_cfg, "synthetic_lr_scale", 1.0)
                            diag = acs[tr.agent_id].update_from_transition(tr_hat, synthetic=True, update_actor=update_actor, lr_scale=lr_scale)
                            planning_deltas.append(abs(diag["delta"]))
            k_vals = [a.k for a in agents]
            xs_k_std_t = float(np.std(k_vals)) if len(k_vals) > 1 else 0.0
            logger.log_metrics(
                {
                    "t": float(t),
                    "mean_abs_delta_real": sum(real_deltas) / max(len(real_deltas), 1),
                    "mean_abs_delta_planning": sum(planning_deltas) / max(len(planning_deltas), 1),
                    "mean_action": sum(actions) / max(len(actions), 1),
                    "price": price,
                    "v_true": fundamental,
                    "total_demand": total_flow,
                    "total_inventory": total_inventory,
                    "bar_x_pre": bar_x_pre,
                    "bar_x": bar_x,
                    "bar_u": bar_u,
                    "mean_k": sum(k_vals) / max(len(agents), 1),
                    "xs_k_std": xs_k_std_t,
                    "mean_belief": mean_belief,
                    "mean_correctness": mean_correctness,
                    "mean_salience": mean_salience,
                    "regime_post": regime_post,
                    "xs_action_std": xs_action_std,
                    "xs_action_mean_abs": xs_action_mean_abs,
                    "xs_flow_std": xs_flow_std,
                    "xs_flow_mean_abs": xs_flow_mean_abs,
                    "xs_belief_std": xs_belief_std,
                    "xs_sigma_mean": xs_sigma_mean,
                    "xs_rho_mean": xs_rho_mean,
                    "xs_rho_max": xs_rho_max,
                    "mean_myopic_term": sum(myopic_terms) / max(len(myopic_terms), 1) if myopic_terms else 0.0,
                    "mean_hedging_term": sum(hedging_terms) / max(len(hedging_terms), 1) if hedging_terms else 0.0,
                }
            )
        elif log_metrics:
            k_vals = [a.k for a in agents]
            xs_k_std_t = float(np.std(k_vals)) if len(k_vals) > 1 else 0.0
            logger.log_metrics(
                {
                    "t": float(t),
                    "mean_abs_delta_real": sum(real_deltas) / max(len(real_deltas), 1),
                    "mean_abs_delta_planning": 0.0,
                    "mean_action": sum(actions) / max(len(actions), 1),
                    "price": price,
                    "v_true": fundamental,
                    "total_demand": total_flow,
                    "total_inventory": total_inventory,
                    "bar_x_pre": bar_x_pre,
                    "bar_x": bar_x,
                    "bar_u": bar_u,
                    "mean_k": sum(k_vals) / max(len(agents), 1),
                    "xs_k_std": xs_k_std_t,
                    "mean_belief": mean_belief,
                    "mean_correctness": mean_correctness,
                    "mean_salience": mean_salience,
                    "regime_post": regime_post,
                    "xs_action_std": xs_action_std,
                    "xs_action_mean_abs": xs_action_mean_abs,
                    "xs_flow_std": xs_flow_std,
                    "xs_flow_mean_abs": xs_flow_mean_abs,
                    "xs_belief_std": xs_belief_std,
                    "xs_sigma_mean": xs_sigma_mean,
                    "xs_rho_mean": xs_rho_mean,
                    "xs_rho_max": xs_rho_max,
                    "mean_myopic_term": sum(myopic_terms) / max(len(myopic_terms), 1) if myopic_terms else 0.0,
                    "mean_hedging_term": sum(hedging_terms) / max(len(hedging_terms), 1) if hedging_terms else 0.0,
                }
            )

    return SimulationOutput(
        prices=logger.prices,
        fundamentals=logger.fundamentals,
        mean_beliefs=logger.mean_beliefs,
        demands=logger.total_demands,
        k_series=k_series,
        metrics=logger.metrics,
    )
