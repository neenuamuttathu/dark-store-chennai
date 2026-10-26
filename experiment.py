"""Offline run of the required experiment: profit for P = 2, 5, 10, 20, 25 with
10,000 customers and 50 candidate sites. Writes results/profit_vs_P.csv and .png.

Run: python experiment.py [--max-distance 5] [--seed 42]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from data import generate_instance  # noqa: E402
from model import Params, solve, summarize  # noqa: E402

P_VALUES = (2, 5, 10, 20, 25)

parser = argparse.ArgumentParser()
parser.add_argument("--max-distance", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

customers, candidates = generate_instance(10_000, 50, args.seed)
base = Params(max_distance_km=args.max_distance)
rows = []
for P in P_VALUES:
    p = replace(base, max_stores=P)
    k = summarize(customers, solve(customers, candidates, p), p)
    rows.append({
        "P": P,
        "stores_opened": k["stores_open"],
        "profit_lakh": round(k["profit"] / 1e5, 3),
        "customers_served_pct": round(100 * k["customers_served"] / k["customers_total"], 2),
        "avg_distance_km": round(k["avg_km"], 3),
        "max_distance_km": round(k["max_km"], 3),
        "status": k["status"],
        "mip_gap": k["mip_gap"],
        "solve_time_s": round(k["solve_time_s"], 3),
    })
    print(rows[-1])

out = Path(__file__).parent / "results"
out.mkdir(exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "profit_vs_P.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax.plot(df["P"], df["profit_lakh"], color="#2a78d6", linewidth=2, marker="o", markersize=7,
        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
for n, r in df.iterrows():
    first = n == 0  # keep the first label clear of the y-axis
    ax.annotate(f"{r['stores_opened']} open", (r["P"], r["profit_lakh"]), textcoords="offset points",
                xytext=(8, -12) if first else (-6, 7), ha="left" if first else "right",
                fontsize=8, color="#52514e")
ax.set_xticks(P_VALUES)
ax.set_ylim(bottom=0)
ax.set_xlabel("Max dark stores allowed to open (P)", color="#52514e")
ax.set_ylabel("Profit (₹ lakh / month)", color="#52514e")
ax.set_title(f"Profit vs P: 10,000 customers, 50 candidates, max distance {args.max_distance:g} km",
             fontsize=10, color="#0b0b0b", loc="left")
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#898781")
fig.tight_layout()
fig.savefig(out / "profit_vs_P.png")
print(f"Saved {out / 'profit_vs_P.csv'} and {out / 'profit_vs_P.png'}")
