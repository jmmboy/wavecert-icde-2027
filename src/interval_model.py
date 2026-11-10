"""Regularized reversible panel CTMC, shared by event representations.

States are 0 (no common ADL difficulty), 1 (difficulty), and 2 (death).
The four rates (0->1, 0->2, 1->0, 1->2) are constant in time conditional
on the *baseline* covariates and have separate log-linear predictors.
Observed living states have interval/panel likelihood exp(Q * delta)[i,j].
Death discovered in an interval uses exp(Q * delta)[i,2], which integrates
all hidden deterioration/recovery paths before death. An exact death time
uses sum(exp(Q * delta)[i,k] * q[k,2], k=0,1), a density per year.

This is a deliberately small CTMC statistical baseline, NOT a reproduction
of HIDeM or C-TPM. Informative visits/dropout are not corrected by this model.
Use fit weights only when the corresponding observation assumptions hold.
Health recovery is allowed; state occupation is different from first-event
cumulative incidence and is scored separately below.

API
---
model.fit(X, panel_obs, sample_weight=None, initial_params=None, include_death=None)
X has one row per person. panel_obs is a dict of arrays, a DataFrame, or
a list of dicts, with person (integer X-row), delta (years), start, end.
Each row starts from an observed living state. Nonoverlapping adjacent
intervals conditional on actual observed states form a panel likelihood.
end=-1 denotes a living person with unknown health (mask {0,1}); -2 denotes
fully unknown status and contributes 1. Alternatively pass end_mask [n,3].
Rows with unknown starting state are rejected, rather than treated healthy.
Do not concatenate rows starting from an imputed previous unknown state.
Optional exact_death (bool) selects density instead of interval probability.

predict_state(X, t=3, start_state=0) returns [n,3] occupation probabilities.
predict_cif_first(X,t=3) is first entry to state 1 before death from state 0.
observed_log_likelihood is identified from the actual panel observations;
oracle_state_metrics requires independently generated exact latent states.
partial_state_brier_bounds reports the identified interval for a masked
horizon state; it never imputes a candidate model's own outcome labels.

For cohorts without complete death ascertainment, include_death=False fixes
both death rates to EXACTLY zero, fits only the 0<->1 two-state working
model, and rejects death-labelled intervals. This is a model of health in
the supplied alive-observed panels, NOT an estimate of unconditional death
or population health risk. Use actual elapsed visit time for predictions.
Conditioning on attendance/survival needs explicit interpretation and any
IPW sensitivity must state its exchangeability/positivity assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

RATE_NAMES = ("0_to_1", "0_to_2", "1_to_0", "1_to_2")


def _transient_exp(rates, delta, derivative=False):
    """Vectorized analytic exp of a two-state substochastic generator.

    Derivatives are with respect to the four *rates*, not log rates.
    Rates must be strictly positive. The exp eigenvalues are nonpositive,
    avoiding overflow from an exp-times-cosh implementation.
    """
    rates = np.asarray(rates, dtype=float)
    dt = np.broadcast_to(np.asarray(delta, dtype=float), rates.shape[:-1])
    a, b, c, d = np.moveaxis(rates, -1, 0)
    m = -.5 * (a + b + c + d)
    h = .5 * (c + d - a - b)
    z = np.sqrt(h*h + a*c)
    ep = np.exp((m + z)*dt)
    em = np.exp((m - z)*dt)
    C = .5 * (ep + em)
    # expm1 protects the eigenvalue difference for short intervals.
    F = ep * (-np.expm1(-2*z*dt)) / (2*z)
    T = np.empty(rates.shape[:-1] + (2, 2))
    T[..., 0, 0] = C + F*h
    T[..., 0, 1] = F*a
    T[..., 1, 0] = F*c
    T[..., 1, 1] = C - F*h
    if not derivative:
        return T
    dm = np.full(rates.shape, -.5)
    dh = np.broadcast_to(np.array([-.5, -.5, .5, .5]), rates.shape)
    dac = np.zeros_like(rates)
    dac[..., 0] = c
    dac[..., 2] = a
    dz = (h[..., None]*dh + .5*dac) / z[..., None]
    dep = ep[..., None]*dt[..., None]*(dm + dz)
    dem = em[..., None]*dt[..., None]*(dm - dz)
    dC = .5*(dep + dem)
    # dF = [(dep-dem)/2 - F dz] / z. Cancellation at tiny dt is
    # harmless at the tested practical panel scale (>=1e-6 years).
    dF = (.5*(dep-dem) - F[..., None]*dz) / z[..., None]
    D = np.empty(rates.shape[:-1] + (2, 2, 4))
    D[..., 0, 0, :] = dC + dF*h[..., None] + F[..., None]*dh
    D[..., 0, 1, :] = dF*a[..., None]
    D[..., 0, 1, 0] += F
    D[..., 1, 0, :] = dF*c[..., None]
    D[..., 1, 0, 2] += F
    D[..., 1, 1, :] = dC - dF*h[..., None] - F[..., None]*dh
    return T, D


def transition_matrix(rates, delta):
    """Full [n,3,3] transition matrix; rates ordered as RATE_NAMES."""
    rates = np.atleast_2d(np.asarray(rates, dtype=float))
    if np.any(rates < 0) or np.any(rates[:, [0, 2]] <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("Rates must be finite/nonnegative, with positive living transition rates")
    if np.any(np.asarray(delta) < 0):
        raise ValueError("delta must be nonnegative")
    T = _transient_exp(rates, delta)
    P = np.zeros((len(rates), 3, 3))
    P[:, :2, :2] = T
    P[:, :2, 2] = 1 - T.sum(axis=-1)
    P[:, 2, 2] = 1
    # Only remove floating point excursions, never renormalize fitted labels.
    return np.clip(P, 0, 1)


@dataclass
class _Panel:
    person: np.ndarray
    delta: np.ndarray
    start: np.ndarray
    mask: np.ndarray
    exact: np.ndarray


def _as_panel(obs: Any, n: int) -> _Panel:
    if isinstance(obs, (list, tuple)):
        if not obs:
            raise ValueError("Empty panel observations")
        keys = set().union(*(row.keys() for row in obs))
        obs = {k: np.array([r.get(k, False if k == 'exact_death' else None)
                                 for r in obs]) for k in keys}
    def get(names, default=None):
        for key in names:
            if key in obs:
                return np.asarray(obs[key])
        return default
    person = get(("person", "subject_index", "person_index"))
    delta = get(("delta", "dt", "duration"))
    start = get(("start", "currentstate", "start_state"))
    end = get(("end", "nextstate", "end_state"))
    mask = get(("end_mask", "health_mask"))
    if person is None or delta is None or start is None or (end is None and mask is None):
        raise ValueError("panel_obs needs person, delta, start, and end/end_mask")
    if not np.all(np.isfinite(person.astype(float))) or np.any(person.astype(float) != person.astype(int)):
        raise ValueError("person must be integer X-row indices")
    person = person.astype(int)
    delta = delta.astype(float)
    if np.any(person < 0) or np.any(person >= n):
        raise ValueError("person is outside X row range")
    if not np.all(np.isfinite(start.astype(float))) or np.any(start.astype(float) != start.astype(int)):
        raise ValueError("Unknown/invalid start; merge intervals or use a forward model")
    start = start.astype(int)
    if np.any(~np.isin(start, [0, 1])):
        raise ValueError("Each panel interval must start from observed state 0 or 1")
    if np.any(~np.isfinite(delta)) or np.any(delta <= 0):
        raise ValueError("All observation deltas must be positive finite years")
    if mask is None:
        end = end.astype(float)
        if np.any(~np.isfinite(end)) or np.any(~np.isin(end, [-2, -1, 0, 1, 2])):
            raise ValueError("end must be 0/1/2, -1 living health unknown, or -2 unknown")
        mask = np.zeros((len(person), 3), dtype=float)
        for j in range(3):
            mask[:, j] = (end == j) | (end == -2) | ((end == -1) & (j < 2))
    mask = np.asarray(mask, dtype=float)
    if mask.shape != (len(person), 3) or np.any(~np.isin(mask, [0, 1])) or np.any(mask.sum(axis=1) == 0):
        raise ValueError("end_mask must be [n_observations,3] nonempty binary masks")
    exact = get(("exact_death", "death_exact"), np.zeros(len(person), dtype=bool)).astype(bool)
    if np.any(exact & ~(np.all(mask == [0, 0, 1], axis=1))):
        raise ValueError("exact_death requires an observed terminal death")
    if len(delta) != len(person) or len(start) != len(person) or len(exact) != len(person):
        raise ValueError("Observation columns have different lengths")
    return _Panel(person, delta, start, mask, exact)


def _row_likelihood(rates, panel, derivative=False):
    T, D = _transient_exp(rates, panel.delta, derivative=True)
    row = np.arange(len(panel.person))
    ts = T[row, panel.start, :]
    ds = D[row, panel.start, :, :]
    # sum_{k living} P_ik * (mask_k - mask_dead) + mask_dead
    contrast = panel.mask[:, :2] - panel.mask[:, 2, None]
    value = (ts*contrast).sum(axis=1) + panel.mask[:, 2]
    grad = np.einsum("nk,nkj->nj", contrast, ds)
    if np.any(panel.exact):
        death_rates = rates[:, [1, 3]]
        density = (ts*death_rates).sum(axis=1)
        densgrad = np.einsum("nk,nkj->nj", death_rates, ds)
        densgrad[:, 1] += ts[:, 0]
        densgrad[:, 3] += ts[:, 1]
        value = np.where(panel.exact, density, value)
        grad = np.where(panel.exact[:, None], densgrad, grad)
    if derivative:
        return np.maximum(value, 1e-300), grad
    return np.maximum(value, 1e-300)


class ReversiblePanelCTMC:
    """Ridge regularized baseline CTMC with analytic likelihood gradients.

    penalty is on standardized non-intercept coefficients, summed across all
    four rates. The NLL is averaged per *weighted observed person*, so long
    panels contribute their valid joint likelihood, not a fictitious set of
    independent persons. Repeated intervals share that person's baseline X.
    """
    def __init__(self, penalty=.01, max_iter=150, tol=1e-7,
                 standardize=True, lograte_bounds=(-14., 5.), include_death=True):
        self.penalty = float(penalty)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.standardize = standardize
        self.lograte_bounds = tuple(lograte_bounds)
        self.include_death_ = bool(include_death)
        if self.penalty < 0:
            raise ValueError("penalty must be nonnegative")

    def _features(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be [n_persons,n_features]")
        X = np.where(np.isfinite(X), X, np.nan)
        if fit:
            self.median_ = np.nanmedian(X, axis=0)
            self.median_ = np.nan_to_num(self.median_, nan=0.)
        if X.shape[1] != len(self.median_):
            raise ValueError("Prediction feature count differs from training")
        X = np.where(np.isnan(X), self.median_, X)
        if fit:
            self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1])
            self.scale_ = X.std(axis=0) if self.standardize else np.ones(X.shape[1])
            self.scale_[self.scale_ < 1e-8] = 1
        Z = (X - self.mean_) / self.scale_
        return np.column_stack((np.ones(len(X)), Z))

    def _rates(self, Z, theta=None):
        if theta is None:
            theta = self.params_
        eta = Z @ theta.T
        clipped = np.clip(eta, *self.lograte_bounds)
        rates = np.exp(clipped)
        active = ((eta > self.lograte_bounds[0]) & (eta < self.lograte_bounds[1]))
        if not self.include_death_:
            rates[:, [1, 3]] = 0.
            active[:, [1, 3]] = False
        return rates, active

    def fit(self, X, panel_obs, sample_weight=None, initial_params=None, include_death=None):
        tick = time.perf_counter()
        Z = self._features(X, fit=True)
        panel = _as_panel(panel_obs, len(Z))
        if include_death is not None:
            self.include_death_ = bool(include_death)
        if not self.include_death_ and np.any((panel.mask[:, :2].sum(axis=1) == 0) | panel.exact):
            raise ValueError("include_death=False cannot fit death observations; restrict to alive-observed intervals")
        if not len(panel.person):
            raise ValueError("No panel observations")
        w_person = np.ones(len(Z)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w_person.shape != (len(Z),) or np.any(~np.isfinite(w_person)) or np.any(w_person < 0):
            raise ValueError("sample_weight must be nonnegative per-person weights")
        w = w_person[panel.person]
        denominator = w_person[np.unique(panel.person)].sum()
        if denominator <= 0:
            raise ValueError("No positively weighted observed persons")
        Zobs = Z[panel.person]
        shape = (4, Z.shape[1])
        if initial_params is None:
            initial = np.zeros(shape)
            expo = np.array([panel.delta[panel.start == j].sum() for j in [0, 1]])
            ending = panel.mask.argmax(axis=1)
            init_rates = [
                ((panel.start == 0) & (ending == 1)).sum() / max(expo[0], 1),
                ((panel.start == 0) & (ending == 2)).sum() / max(expo[0], 1),
                ((panel.start == 1) & (ending == 0)).sum() / max(expo[1], 1),
                ((panel.start == 1) & (ending == 2)).sum() / max(expo[1], 1),
            ]
            initial[:, 0] = np.log(np.clip(init_rates, .005, 1.))
        else:
            initial = np.asarray(initial_params, dtype=float).reshape(shape).copy()

        def objective(flat):
            theta = flat.reshape(shape)
            rates, active = self._rates(Zobs, theta)
            value, dvalue = _row_likelihood(rates, panel, True)
            nll = -(w*np.log(value)).sum()/denominator
            d_eta = -(w/value)[:, None]*dvalue*rates*active/denominator
            grad = d_eta.T @ Zobs
            nll += .5*self.penalty*np.square(theta[:, 1:]).sum()
            grad[:, 1:] += self.penalty*theta[:, 1:]
            return float(nll), grad.ravel()

        result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.max_iter, "ftol": self.tol,
                                   "gtol": max(self.tol, 1e-6), "maxls": 30})
        self.params_ = result.x.reshape(shape)
        self.intercept_ = self.params_[:, 0].copy()
        self.coef_ = self.params_[:, 1:].copy()
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.optimizer_message_ = str(result.message)
        self.objective_ = float(result.fun)
        self.gradient_max_ = float(np.abs(result.jac).max())
        self.fit_seconds_ = time.perf_counter() - tick
        self.n_observations_ = len(panel.person)
        self.n_persons_ = len(np.unique(panel.person))
        rate_active = self._rates(Zobs)[1]
        self.rate_clipping_fraction_ = float((~rate_active[:, [0, 2] if not self.include_death_ else [0, 1, 2, 3]]).mean())
        self.active_rate_names_ = RATE_NAMES if self.include_death_ else (RATE_NAMES[0], RATE_NAMES[2])
        return self

    def predict_rates(self, X):
        return self._rates(self._features(X))[0]

    def predict_transition(self, X, t=3.):
        P = transition_matrix(self.predict_rates(X), t)
        if not self.include_death_:
            P[:, :2, 2] = 0.
            P[:, :2, :2] /= P[:, :2, :2].sum(axis=2, keepdims=True)
        return P

    def predict_state(self, X, t=3., start_state=0):
        P = self.predict_transition(X, t)
        state = np.broadcast_to(np.asarray(start_state, dtype=int), (len(P),))
        if np.any(~np.isin(state, [0, 1, 2])):
            raise ValueError("start_state must be 0/1/2")
        return P[np.arange(len(P)), state]

    def predict_alive_state(self, X, t=3., start_state=0):
        """Health conditional on survival; requires justified conditioning."""
        p = self.predict_state(X, t=t, start_state=start_state)[:, :2]
        return p/np.maximum(p.sum(axis=1, keepdims=True), 1e-300)

    def predict_cif_first(self, X, t=3.):
        """P(first 0->1 by t before 0->2), from healthy baseline only.

        Recovery after first entry does not reverse the first-entry event.
        This quantity is NOT the state-1 occupancy returned above.
        """
        r = self.predict_rates(X)
        total = r[:, 0] + r[:, 1]
        return (r[:, 0]/total)*(-np.expm1(-total*np.asarray(t)))

    def observed_log_likelihood(self, X, panel_obs, per_row=False, per_person=False):
        Z = self._features(X)
        panel = _as_panel(panel_obs, len(Z))
        r = self._rates(Z[panel.person])[0]
        ll = np.log(_row_likelihood(r, panel))
        if per_row:
            return ll
        if per_person:
            return np.bincount(panel.person, weights=ll, minlength=len(Z))
        return float(ll.sum())

    def observed_nll(self, X, panel_obs, sample_weight=None):
        """Per-observed-person NLL; same protocol required across models.

        Exact-death density scores depend on time units; all models must use
        the same years convention. Scores from different observation schedules
        are not directly comparable as if they were one common endpoint.
        """
        panel = _as_panel(panel_obs, len(X))
        ll = self.observed_log_likelihood(X, panel_obs, per_row=True)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)
        return float(-(w[panel.person]*ll).sum()/w[np.unique(panel.person)].sum())


def oracle_state_metrics(predicted, true_states):
    """Simulation-only scores using independently available exact truth.

    multiclass_brier is the sum over all 3 classes (range 0..2), per plan.
    Do not supply midpoint-coded/estimated latent states as oracle truth.
    """
    p = _check_probabilities(predicted)
    y = np.asarray(true_states)
    if y.shape != (len(p),) or np.any(~np.isin(y, [0, 1, 2])):
        raise ValueError("Exact independent latent truth must be 0/1/2 per person")
    yi = y.astype(int)
    onehot = np.eye(3)[yi]
    b = np.square(p-onehot).sum(axis=1)
    return {"multiclass_brier": float(b.mean()),
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(p)), yi], 1e-300)).mean()),
            "brier_per_person": b}


def _check_probabilities(predicted):
    p = np.asarray(predicted, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or np.any(~np.isfinite(p)) or np.any(p < -1e-8) or np.any(p > 1+1e-8):
        raise ValueError("predicted must be finite [n,3] probabilities")
    if not np.allclose(p.sum(axis=1), 1., atol=1e-7):
        raise ValueError("State probabilities do not sum to 1")
    return p


def partial_state_brier_bounds(predicted, horizon_state_mask, sample_weight=None):
    """Logical Brier bounds from independently constructed horizon masks.

    mask[i,k]=1 only if state k is compatible with observed data at horizon.
    For a reversible process, healthy before and impaired after a horizon do
    not rule out any intervening living state. Event-time midpoint labels are
    invalid. Fully unobserved status permits {0,1,2}. Bounds may be wide; that
    is uncertainty in observation, not a defect to remove by self-imputation.
    """
    p = _check_probabilities(predicted)
    mask = np.asarray(horizon_state_mask, dtype=bool)
    if mask.shape != p.shape or np.any(mask.sum(axis=1) == 0):
        raise ValueError("horizon_state_mask must be [n,3] and nonempty")
    all_losses = 1 + np.square(p).sum(axis=1, keepdims=True) - 2*p
    lo = np.where(mask, all_losses, np.inf).min(axis=1)
    hi = np.where(mask, all_losses, -np.inf).max(axis=1)
    w = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.shape != (len(p),) or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid sample weights")
    return {"brier_lower": float(np.average(lo, weights=w)),
            "brier_upper": float(np.average(hi, weights=w)),
            "exact_horizon_state_fraction": float(np.average(mask.sum(axis=1) == 1, weights=w)),
            "lower_per_person": lo, "upper_per_person": hi}


# Short stable alias for the experiment runner.
PanelCTMC = ReversiblePanelCTMC
