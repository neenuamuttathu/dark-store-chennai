"""Dark-store location MILP.

Profit-maximising facility location with (a) at most P open stores and (b) a maximum
road distance D_max between a customer and the store that serves them.

    max  sum_ij pi_ij y_ij - sum_j F_j x_j,        pi_ij = w_i (m - c d_ij)
    s.t. sum_j y_ij <= 1                  for all i      (1) serve each customer at most once
         y_ij <= x_j                      for all i, j   (2) only open stores serve
         sum_i w_i y_ij <= C x_j          for all j      (3) capacity (optional)
         sum_j x_j <= P                                  (4) store budget
         y_ij = 0 if d_ij > D_max                        (5) max distance (new)
         x_j in {0,1}, 0 <= y_ij <= 1

Without capacity the model is solved in an equivalent compact form (see solve_compact).
With capacity the pair-level model above is solved directly (see solve_pairs).
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

EARTH_RADIUS_KM = 6371.0

STATUS_TEXT = {
    0: "Optimal",
    1: "Time limit reached (best solution found)",
    2: "Infeasible",
    3: "Unbounded",
    4: "Solver error",
}


@dataclass(frozen=True)
class Params:
    max_stores: int = 10                # P
    max_distance_km: float = 5.0        # D_max, road km
    order_value: float = 500.0          # average order value, Rs
    margin_pct: float = 25.0            # gross margin on order value, %
    cost_per_km: float = 10.0           # last-mile cost per order per road-km, Rs
    fixed_cost_min: float = 80_000.0    # Rs/month for the cheapest site
    fixed_cost_max: float = 140_000.0   # Rs/month for the most expensive site
    circuity: float = 1.3               # road distance / straight-line distance
    capacity: float | None = None       # orders/month per store; None = uncapacitated
    single_sourcing: bool = False       # capacitated only: force y binary
    time_limit_s: float = 60.0
    mip_gap: float = 1e-4

    @property
    def margin_per_order(self):
        return self.order_value * self.margin_pct / 100


@dataclass
class Solution:
    status: str
    open_stores: np.ndarray   # bool per candidate
    fixed_cost: np.ndarray    # Rs/month per candidate
    flows: pd.DataFrame       # one row per served (customer, store) pair
    objective: float
    mip_gap: float
    solve_time_s: float
    n_vars: int
    n_constraints: int
    pairs_total: int          # N * M
    pairs_feasible: int       # pairs left after the max-distance / profitability filter
    formulation: str


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def road_distance_matrix(customers, candidates, circuity):
    return circuity * haversine_km(
        customers["lat"].to_numpy()[:, None], customers["lon"].to_numpy()[:, None],
        candidates["lat"].to_numpy()[None, :], candidates["lon"].to_numpy()[None, :],
    )


def site_fixed_costs(candidates, p):
    return p.fixed_cost_min + (p.fixed_cost_max - p.fixed_cost_min) * candidates["rent_index"].to_numpy()


def _run_milp(c, integrality, a, ub, p):
    t0 = time.perf_counter()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, -np.inf, ub),
        options={"time_limit": p.time_limit_s, "mip_rel_gap": p.mip_gap, "presolve": True},
    )
    gap = getattr(res, "mip_gap", None)
    return res, time.perf_counter() - t0, float("nan") if gap is None else gap


def solve(customers, candidates, p: Params) -> Solution:
    d = road_distance_matrix(customers, candidates, p.circuity)
    w = customers["orders"].to_numpy(dtype=float)
    contrib = w[:, None] * (p.margin_per_order - p.cost_per_km * d)
    # Constraint (5): a pair (i, j) can only be used when d_ij <= D_max. Pairs with
    # pi_ij <= 0 are dropped as well - serving them at a loss never helps.
    feasible = (d <= p.max_distance_km) & (contrib > 0)
    fixed = site_fixed_costs(candidates, p)
    if p.capacity is None:
        return solve_compact(d, w, contrib, feasible, fixed, p)
    return solve_pairs(d, w, contrib, feasible, fixed, p)


def solve_compact(d, w, contrib, feasible, fixed, p) -> Solution:
    """Uncapacitated case, exact compact reformulation.

    For customer i, sort its feasible stores nearest-first (so pi_i(1) >= pi_i(2) >= ...).
    With S the set of open stores, the profit from i is

        max_{j in S} pi_ij = sum_k (pi_i(k) - pi_i(k+1)) * [S contains one of the k nearest]

    Each indicator is linearised as z <= sum of x over that set of k nearest stores.
    Every (customer, k) that shares the same set shares one z variable (coefficients add
    up), which shrinks ~N*M assignment variables to a few thousand without changing the
    optimum. The LP relaxation is as tight as the strong y_ij <= x_j formulation.
    """
    n, m = d.shape
    order = np.argsort(np.where(feasible, d, np.inf), axis=1, kind="stable")  # nearest first
    n_feas = feasible.sum(axis=1)
    pi_sorted = np.take_along_axis(np.where(feasible, contrib, 0.0), order, axis=1)
    delta = pi_sorted - np.c_[pi_sorted[:, 1:], np.zeros(n)]
    rank = np.arange(m)[None, :]
    use = (rank < n_feas[:, None]) & (delta > 1e-9)

    # Bitmask of the k nearest stores for every (i, k): cumulative OR along the sorted row.
    one = np.uint64(1)
    lo = np.where(order < 64, one << np.minimum(order, 63).astype(np.uint64), 0).astype(np.uint64)
    hi = np.where(order >= 64, one << np.maximum(order - 64, 0).astype(np.uint64), 0).astype(np.uint64)
    keys = np.column_stack([np.bitwise_or.accumulate(lo, axis=1)[use],
                            np.bitwise_or.accumulate(hi, axis=1)[use]])
    sets, set_id = np.unique(keys, axis=0, return_inverse=True)
    coef = np.bincount(set_id.ravel(), weights=delta[use], minlength=len(sets))
    members = np.unpackbits(sets.astype("<u8").view(np.uint8), axis=1, bitorder="little")[:, :m]

    t = len(sets)
    n_vars = m + t  # [x_0..x_{m-1}, z_0..z_{t-1}]
    rows, cols = np.nonzero(members)
    a_link = sparse.csr_array(
        (np.r_[np.ones(t), -np.ones(len(rows))], (np.r_[np.arange(t), rows], np.r_[m + np.arange(t), cols])),
        shape=(t, n_vars),
    )
    a_card = sparse.csr_array((np.ones(m), (np.zeros(m, dtype=int), np.arange(m))), shape=(1, n_vars))
    a = sparse.vstack([a_link, a_card], format="csr")
    ub = np.r_[np.zeros(t), [p.max_stores]]
    c = np.r_[fixed, -coef]
    integrality = np.r_[np.ones(m), np.zeros(t)]

    res, solve_time, gap = _run_milp(c, integrality, a, ub, p)
    status = STATUS_TEXT.get(res.status, res.message)
    if res.x is None:
        return _empty(status, m, fixed, solve_time, n_vars, a.shape[0], n * m, int(feasible.sum()), "compact")

    x = res.x[:m] > 0.5
    # Each customer goes to its nearest open feasible store.
    open_sorted = x[order] & (rank < n_feas[:, None])
    served = open_sorted.any(axis=1)
    i = np.flatnonzero(served)
    j = order[i, open_sorted[i].argmax(axis=1)]
    flows = pd.DataFrame({"customer_idx": i, "store_idx": j, "share": 1.0, "orders": w[i], "distance_km": d[i, j]})
    return Solution(status, x, fixed, flows, -res.fun, gap, solve_time, n_vars, a.shape[0],
                    n * m, int(feasible.sum()), "compact")


def solve_pairs(d, w, contrib, feasible, fixed, p) -> Solution:
    """Capacitated case: the pair-level model with one y variable per feasible pair."""
    n, m = d.shape
    ii, jj = np.nonzero(feasible)
    k = len(ii)
    y_cols = m + np.arange(k)  # variable vector is [x_0..x_{m-1}, y_0..y_{k-1}]
    n_vars = m + k

    # (1) assignment: one row per customer that has at least one reachable store
    _, cust_row = np.unique(ii, return_inverse=True)
    a1 = sparse.csr_array((np.ones(k), (cust_row, y_cols)), shape=(cust_row.max(initial=-1) + 1, n_vars))
    # (3) capacity sum_i w_i y_ij - C x_j <= 0. It also forces y_ij = 0 when x_j = 0, so
    # the per-pair rows (2) are left out: they tighten the LP but make it far slower to solve.
    a3 = sparse.csr_array(
        (np.r_[w[ii], -p.capacity * np.ones(m)], (np.r_[jj, np.arange(m)], np.r_[y_cols, np.arange(m)])),
        shape=(m, n_vars),
    )
    # (4) at most P stores
    a4 = sparse.csr_array((np.ones(m), (np.zeros(m, dtype=int), np.arange(m))), shape=(1, n_vars))
    a = sparse.vstack([a1, a3, a4], format="csr")
    ub = np.r_[np.ones(a1.shape[0]), np.zeros(m), [p.max_stores]]

    c = np.r_[fixed, -contrib[ii, jj]]  # milp minimises, so negate profit
    integrality = np.r_[np.ones(m), np.full(k, 1.0 if p.single_sourcing else 0.0)]

    res, solve_time, gap = _run_milp(c, integrality, a, ub, p)
    status = STATUS_TEXT.get(res.status, res.message)
    if res.x is None:
        return _empty(status, m, fixed, solve_time, n_vars, a.shape[0], n * m, k, "pairs")

    x = res.x[:m] > 0.5
    y = res.x[m:]
    keep = (y > 1e-6) & x[jj]
    flows = pd.DataFrame({
        "customer_idx": ii[keep],
        "store_idx": jj[keep],
        "share": y[keep],
        "orders": w[ii[keep]] * y[keep],
        "distance_km": d[ii[keep], jj[keep]],
    })
    return Solution(status, x, fixed, flows, -res.fun, gap, solve_time, n_vars, a.shape[0], n * m, k, "pairs")


def _empty(status, m, fixed, solve_time, n_vars, n_cons, pairs_total, pairs_feasible, formulation):
    flows = pd.DataFrame(columns=["customer_idx", "store_idx", "share", "orders", "distance_km"])
    return Solution(status, np.zeros(m, bool), fixed, flows, float("nan"), float("nan"), solve_time,
                    n_vars, n_cons, pairs_total, pairs_feasible, formulation)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return values[order][np.searchsorted(cum, q * cum[-1])]


def summarize(customers, sol: Solution, p: Params) -> dict:
    """Raw KPI numbers for one solution (money in Rs/month)."""
    f = sol.flows
    km = f["distance_km"].to_numpy(dtype=float)
    wts = f["orders"].to_numpy(dtype=float)
    orders = wts.sum()
    n_open = int(sol.open_stores.sum())
    gross_margin = orders * p.margin_per_order
    delivery = p.cost_per_km * float(wts @ km)
    fixed = float(sol.fixed_cost[sol.open_stores].sum())
    per_customer = f.groupby("customer_idx").size()
    has_flows = len(f) > 0
    return {
        "profit": gross_margin - delivery - fixed,
        "revenue": orders * p.order_value,
        "gross_margin": gross_margin,
        "delivery_cost": delivery,
        "fixed_cost": fixed,
        "stores_open": n_open,
        "customers_served": int(len(per_customer)),
        "customers_total": len(customers),
        "customers_split": int((per_customer > 1).sum()),
        "orders_served": orders,
        "orders_total": float(customers["orders"].sum()),
        "avg_km": float(wts @ km / orders) if has_flows else float("nan"),
        "p90_km": float(_weighted_quantile(km, wts, 0.9)) if has_flows else float("nan"),
        "max_km": float(km.max()) if has_flows else float("nan"),
        "orders_per_store": orders / n_open if n_open else float("nan"),
        "utilization": orders / (n_open * p.capacity) if n_open and p.capacity else float("nan"),
        "status": sol.status,
        "mip_gap": sol.mip_gap,
        "solve_time_s": sol.solve_time_s,
        "n_vars": sol.n_vars,
        "n_constraints": sol.n_constraints,
        "pairs_total": sol.pairs_total,
        "pairs_feasible": sol.pairs_feasible,
    }


def store_table(candidates, sol: Solution, p: Params) -> pd.DataFrame:
    """One row per opened store."""
    f = sol.flows.assign(
        contribution=lambda t: t["orders"] * (p.margin_per_order - p.cost_per_km * t["distance_km"]),
        order_km=lambda t: t["orders"] * t["distance_km"],
    )
    g = f.groupby("store_idx").agg(
        customers=("customer_idx", "nunique"),
        orders=("orders", "sum"),
        order_km=("order_km", "sum"),
        contribution=("contribution", "sum"),
    )
    idx = np.flatnonzero(sol.open_stores)
    g = g.reindex(idx, fill_value=0)
    out = candidates.iloc[idx][["store", "locality", "lat", "lon"]].reset_index(drop=True)
    out["customers"] = g["customers"].to_numpy()
    out["orders"] = g["orders"].to_numpy()
    out["avg_km"] = (g["order_km"] / g["orders"].replace(0, np.nan)).to_numpy()
    if p.capacity:
        out["utilization_pct"] = 100 * out["orders"] / p.capacity
    out["fixed_cost"] = sol.fixed_cost[idx]
    out["contribution"] = g["contribution"].to_numpy()
    out["store_profit"] = out["contribution"] - out["fixed_cost"]
    return out
