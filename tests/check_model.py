"""Correctness checks for the dark-store model. Run: python tests/check_model.py"""

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import generate_instance  # noqa: E402
from model import Params, road_distance_matrix, site_fixed_costs, solve, summarize  # noqa: E402


def brute_force_profit(customers, candidates, p):
    """Enumerate every set of at most P stores; customers take their best feasible open store."""
    d = road_distance_matrix(customers, candidates, p.circuity)
    w = customers["orders"].to_numpy(float)
    contrib = w[:, None] * (p.margin_per_order - p.cost_per_km * d)
    contrib[(d > p.max_distance_km) | (contrib < 0)] = 0
    fixed = site_fixed_costs(candidates, p)
    best = 0.0  # opening nothing is always feasible
    m = len(candidates)
    for size in range(1, p.max_stores + 1):
        for s in itertools.combinations(range(m), size):
            s = list(s)
            best = max(best, contrib[:, s].max(axis=1).sum() - fixed[s].sum())
    return best


def check_brute_force():
    # Fixed costs scaled to the small customer base so that opening stores pays off.
    cust, cand = generate_instance(400, 9, seed=7)
    for dmax, P in [(3.0, 2), (5.0, 3), (8.0, 4), (5.0, 9)]:
        p = Params(max_stores=P, max_distance_km=dmax, fixed_cost_min=4_000, fixed_cost_max=7_000)
        sol = solve(cust, cand, p)
        bf = brute_force_profit(cust, cand, p)
        assert bf > 0
        assert abs(sol.objective - bf) <= 1e-6 * bf, (dmax, P, sol.objective, bf)
        print(f"  brute force D_max={dmax} P={P}: MILP {sol.objective:,.0f} = enumeration {bf:,.0f} "
              f"({sol.open_stores.sum()} open)")


def check_capacitated_matches_compact():
    # With capacity too large to bind, the pair-level model must reach the same optimum.
    cust, cand = generate_instance(1500, 20, seed=3)
    p = Params(max_stores=6, max_distance_km=5.0)
    a = solve(cust, cand, p)
    b = solve(cust, cand, replace(p, capacity=1e9))
    assert abs(a.objective - b.objective) <= 1e-3 * abs(a.objective), (a.objective, b.objective)
    print(f"  compact {a.objective:,.0f} vs pair-level {b.objective:,.0f} (non-binding capacity)")


def check_invariants():
    cust, cand = generate_instance(10_000, 50, seed=42)
    prev = -np.inf
    for P in (2, 5, 10, 20, 25):
        p = Params(max_stores=P, max_distance_km=5.0)
        sol = solve(cust, cand, p)
        k = summarize(cust, sol, p)
        assert k["stores_open"] <= P
        assert k["max_km"] <= p.max_distance_km + 1e-9, k["max_km"]
        assert abs(k["profit"] - sol.objective) <= 1e-6 * abs(sol.objective), (k["profit"], sol.objective)
        assert sol.flows["store_idx"].isin(np.flatnonzero(sol.open_stores)).all()
        assert sol.flows["customer_idx"].is_unique
        assert k["profit"] >= prev - 1e-6, "profit must be non-decreasing in P"
        prev = k["profit"]
        print(f"  P={P:2d}: profit Rs {k['profit'] / 1e5:6.2f} lakh, {k['stores_open']} open, "
              f"max distance {k['max_km']:.2f} km, {k['solve_time_s']:.2f}s")

    profits = [summarize(cust, solve(cust, cand, Params(max_stores=10, max_distance_km=r)),
                         Params())["profit"] for r in (2.0, 3.0, 5.0, 8.0)]
    assert all(b >= a - 1e-6 for a, b in zip(profits, profits[1:])), profits
    print("  profit non-decreasing in D_max:", [f"{x / 1e5:.2f}" for x in profits])


def check_capacity_respected():
    cust, cand = generate_instance(2000, 20, seed=5)
    p = Params(max_stores=8, max_distance_km=5.0, capacity=1500, time_limit_s=30)
    sol = solve(cust, cand, p)
    load = sol.flows.groupby("store_idx")["orders"].sum()
    assert (load <= p.capacity + 1e-6).all(), load.max()
    assert sol.flows["distance_km"].max() <= p.max_distance_km + 1e-9
    print(f"  capacitated: max load {load.max():.0f} <= {p.capacity:.0f}, status '{sol.status}', "
          f"gap {sol.mip_gap:.4f}, {sol.solve_time_s:.1f}s")


if __name__ == "__main__":
    for check in (check_brute_force, check_capacitated_matches_compact, check_invariants, check_capacity_respected):
        print(check.__name__)
        check()
    print("All checks passed.")
