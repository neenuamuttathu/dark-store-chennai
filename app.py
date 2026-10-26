"""Chennai dark-store location optimizer (Streamlit app)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from data import KM_PER_DEG_LAT, KM_PER_DEG_LON, generate_instance
from model import Params, solve, store_table, summarize

SWEEP_P = (2, 5, 10, 20, 25)
SWEEP_CUSTOMERS, SWEEP_CANDIDATES = 10_000, 50
SWEEP_DMAX = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)

# Categorical palette (8 slots, fixed order) and chart ink, per theme.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {"light": "#0b0b0b", "dark": "#f0efec"}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MUTED = "#898781"

st.set_page_config(page_title="Chennai Dark Store Optimizer", page_icon="🏬", layout="wide")


def rgb(hex_color, alpha=255):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def lakh(x):
    return f"₹{x / 1e5:,.2f} lakh"


def km(x):
    return "–" if np.isnan(x) else f"{x:.2f} km"


@st.cache_data(show_spinner=False)
def get_instance(n_customers, n_candidates, seed):
    return generate_instance(n_customers, n_candidates, seed)


@st.cache_data(show_spinner=False, max_entries=128)
def run_model(n_customers, n_candidates, seed, params):
    customers, candidates = get_instance(n_customers, n_candidates, seed)
    return solve(customers, candidates, params)


def catchment_colors(lat, lon, n_colors):
    """Give each open store a palette slot, keeping same-colour stores as far apart as possible."""
    pos = np.c_[lat * KM_PER_DEG_LAT, lon * KM_PER_DEG_LON]
    slots = []
    for s in range(len(pos)):
        best, best_dist = 0, -1.0
        for c in range(n_colors):
            same = [t for t in range(s) if slots[t] == c]
            dist = min(np.hypot(*(pos[s] - pos[t])) for t in same) if same else np.inf
            if dist > best_dist:
                best, best_dist = c, dist
        slots.append(best)
    return slots


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Inputs")
    with st.form("inputs"):
        n_candidates = st.number_input("Dark store candidate sites", 5, 100, 50, step=5)
        n_customers = st.number_input("Customers", 500, 20_000, 10_000, step=500)
        max_dist = st.slider("Max distance store ↔ customer (road km)", 1.0, 15.0, 5.0, 0.5,
                             help="No customer may be served from a store farther than this.")
        max_stores = st.number_input("Max dark stores allowed to open (P)", 1, 100, 10)
        seed = st.number_input("Random seed (data)", 0, 9_999, 42)

        with st.expander("Economic assumptions"):
            order_value = st.number_input("Average order value (₹)", 100, 5_000, 500, step=50)
            margin_pct = st.number_input("Gross margin (% of order value)", 1.0, 80.0, 25.0, step=1.0)
            cost_per_km = st.number_input("Delivery cost (₹ per order per road km)", 0.0, 100.0, 10.0, step=1.0)
            fixed_min = st.number_input("Fixed cost, cheapest site (₹/month)", 0, 2_000_000, 80_000, step=10_000)
            fixed_max = st.number_input("Fixed cost, priciest site (₹/month)", 0, 2_000_000, 140_000, step=10_000)
            circuity = st.number_input("Road circuity factor (road ÷ straight-line)", 1.0, 2.0, 1.3, step=0.05)

        with st.expander("Store capacity (optional)"):
            use_capacity = st.checkbox("Limit orders per store", value=False,
                                       help="Switches to the larger pair-level model. Slower: large instances "
                                            "may stop at the time limit with a gap.")
            capacity = st.number_input("Capacity (orders / month / store)", 100, 200_000, 4_000, step=500)
            single_sourcing = st.checkbox("Single sourcing (one store per customer)", value=False)
            time_limit = st.number_input("Solver time limit (s)", 5, 300, 60, step=5)

        st.form_submit_button("Run optimization", type="primary", width="stretch")

params = Params(
    max_stores=int(min(max_stores, n_candidates)),
    max_distance_km=float(max_dist),
    order_value=float(order_value),
    margin_pct=float(margin_pct),
    cost_per_km=float(cost_per_km),
    fixed_cost_min=float(min(fixed_min, fixed_max)),
    fixed_cost_max=float(max(fixed_min, fixed_max)),
    circuity=float(circuity),
    capacity=float(capacity) if use_capacity else None,
    single_sourcing=bool(single_sourcing),
    time_limit_s=float(time_limit),
)
theme = (st.context.theme.type or "light") if hasattr(st.context, "theme") else "light"
palette, ink = PALETTE[theme], INK[theme]

st.title("Dark store location optimizer: Chennai")
st.caption(
    "Chooses which candidate sites to open as dark stores, and which store serves each customer, to maximise "
    "monthly profit, with at most P stores and no delivery longer than the max distance. Solved exactly as a "
    "mixed-integer linear program (HiGHS)."
)

tab_opt, tab_sweep, tab_model = st.tabs(["Optimize", "Profit vs. number of stores", "Model & assumptions"])

# ---------------------------------------------------------------- tab 1: optimize
with tab_opt:
    customers, candidates = get_instance(int(n_customers), int(n_candidates), int(seed))
    with st.spinner("Solving the MILP…"):
        sol = run_model(int(n_customers), int(n_candidates), int(seed), params)
    k = summarize(customers, sol, params)

    if sol.status != "Optimal":
        st.warning(f"Solver status: {sol.status}. MIP gap {k['mip_gap']:.2%} - profit may be below the true optimum.")
    if max_stores > n_candidates:
        st.info(f"P was capped at the number of candidate sites ({n_candidates}).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net profit / month", lakh(k["profit"]), border=True)
    c2.metric("Stores opened", f"{k['stores_open']} of {params.max_stores}", border=True)
    c3.metric("Customers served", f"{k['customers_served'] / k['customers_total']:.1%}", border=True)
    c4.metric("Avg delivery distance", km(k["avg_km"]), border=True)

    left, right = st.columns([0.38, 0.62], gap="medium")
    with left:
        st.subheader("KPIs")
        rows = [
            ("Net profit", f"{lakh(k['profit'])} / month"),
            ("Revenue (GMV)", lakh(k["revenue"])),
            ("Gross margin", lakh(k["gross_margin"])),
            ("Delivery cost", lakh(k["delivery_cost"])),
            ("Fixed store cost", lakh(k["fixed_cost"])),
            ("Dark stores opened", f"{k['stores_open']} of max {params.max_stores}"),
            ("Customers served", f"{k['customers_served']:,} of {k['customers_total']:,} "
                                 f"({k['customers_served'] / k['customers_total']:.1%})"),
            ("Orders served / month", f"{k['orders_served']:,.0f} of {k['orders_total']:,.0f} "
                                      f"({k['orders_served'] / k['orders_total']:.1%})"),
            ("Avg distance (order-weighted)", km(k["avg_km"])),
            ("90th percentile distance", km(k["p90_km"])),
            ("Max delivery distance", f"{km(k['max_km'])} (limit {params.max_distance_km:g} km)"),
            ("Orders per open store / month", f"{k['orders_per_store']:,.0f}"),
        ]
        if params.capacity:
            rows += [
                ("Avg store utilization", f"{k['utilization']:.1%} of {params.capacity:,.0f}"),
                ("Customers split across 2+ stores", f"{k['customers_split']:,}"),
            ]
        rows += [
            ("Solver status", sol.status),
            ("MIP gap", f"{k['mip_gap']:.3%}"),
            ("Solve time", f"{k['solve_time_s']:.2f} s"),
            ("Model size", f"{k['n_vars']:,} vars · {k['n_constraints']:,} rows ({sol.formulation})"),
            ("Pairs within max distance",
             f"{k['pairs_feasible']:,} of {k['pairs_total']:,} ({k['pairs_feasible'] / k['pairs_total']:.1%})"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["KPI", "Value"]), hide_index=True, width="stretch",
                     height=35 * len(rows) + 38,
                     column_config={"KPI": st.column_config.TextColumn(width="medium"),
                                    "Value": st.column_config.TextColumn(width="large")})

    with right:
        st.subheader("Solution map")
        t1, t2 = st.columns(2)
        show_lines = t1.toggle("Assignment lines", value=True)
        show_radius = t2.toggle("Max-distance radius", value=True)

        stores = store_table(candidates, sol, params)
        slots = catchment_colors(stores["lat"].to_numpy(), stores["lon"].to_numpy(), len(palette))
        store_rgb = {j: rgb(palette[s]) for j, s in zip(np.flatnonzero(sol.open_stores), slots)}

        primary = sol.flows.sort_values("share").drop_duplicates("customer_idx", keep="last")
        cust = customers.copy()
        cust["store_idx"] = -1
        cust.loc[primary["customer_idx"].to_numpy(), "store_idx"] = primary["store_idx"].to_numpy()
        cust["distance_km"] = np.nan
        cust.loc[primary["customer_idx"].to_numpy(), "distance_km"] = primary["distance_km"].to_numpy()
        served = cust["store_idx"] >= 0
        cust["color"] = [store_rgb[j][:3] + [190] if j >= 0 else rgb(MUTED, 120) for j in cust["store_idx"]]
        store_names = candidates["store"].to_numpy()
        cust["tip"] = np.where(
            served,
            cust["customer"] + " · " + cust["orders"].astype(str) + " orders/mo<br>served by "
            + store_names[cust["store_idx"].clip(lower=0)] + " · " + cust["distance_km"].round(2).astype(str) + " km",
            cust["customer"] + " · " + cust["orders"].astype(str) + " orders/mo<br>not served (no open store in range)",
        )

        layers = []
        if show_radius and len(stores):
            layers.append(pdk.Layer(
                "ScatterplotLayer", stores, get_position="[lon, lat]",
                get_radius=1000 * params.max_distance_km / params.circuity,
                filled=False, stroked=True, get_line_color=rgb(ink, 70), line_width_min_pixels=1,
            ))
        if show_lines and len(primary):
            lines = pd.DataFrame({
                "from": cust.loc[served, ["lon", "lat"]].to_numpy().tolist(),
                "to": candidates.iloc[cust.loc[served, "store_idx"]][["lon", "lat"]].to_numpy().tolist(),
                "color": [store_rgb[j][:3] + [70] for j in cust.loc[served, "store_idx"]],
            })
            layers.append(pdk.Layer("LineLayer", lines, get_source_position="from", get_target_position="to",
                                    get_color="color", get_width=1))
        layers.append(pdk.Layer(
            "ScatterplotLayer", cust[["lon", "lat", "color", "tip"]], get_position="[lon, lat]",
            get_fill_color="color", get_radius=35, radius_min_pixels=1.5, pickable=True,
        ))
        closed = candidates.loc[~sol.open_stores].assign(
            tip=lambda t: t["store"] + " · " + t["locality"] + "<br>candidate, not opened")
        layers.append(pdk.Layer(
            "ScatterplotLayer", closed[["lon", "lat", "tip"]], get_position="[lon, lat]", get_radius=160,
            radius_min_pixels=4, filled=True, get_fill_color=rgb(SURFACE[theme], 200), stroked=True,
            get_line_color=rgb(MUTED), line_width_min_pixels=1.5, pickable=True,
        ))
        if len(stores):
            stores_map = stores.assign(
                color=[store_rgb[j] for j in np.flatnonzero(sol.open_stores)],
                tip=lambda t: (t["store"] + " · " + t["locality"] + "<br>" + t["customers"].astype(str)
                               + " customers · " + t["orders"].round(0).astype(int).astype(str) + " orders/mo<br>"
                               + "store profit ₹" + (t["store_profit"] / 1e5).round(2).astype(str) + " lakh/mo"),
            )
            layers.append(pdk.Layer(
                "ScatterplotLayer", stores_map, get_position="[lon, lat]", get_radius=230, radius_min_pixels=7,
                get_fill_color="color", stroked=True, get_line_color=rgb(SURFACE[theme]),
                line_width_min_pixels=2, pickable=True,
            ))
            layers.append(pdk.Layer(
                "TextLayer", stores_map, get_position="[lon, lat]", get_text="store", get_size=13,
                get_color=rgb(ink), get_pixel_offset=[0, -16], font_weight=700,
                outline_width=2, outline_color=rgb(SURFACE[theme]), font_settings={"sdf": True},
            ))
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=13.03, longitude=80.205, zoom=10.2),
                tooltip={"html": "{tip}", "style": {"fontSize": "12px"}},
            ),
            height=640,
            alt="Map of Chennai showing customers coloured by the dark store that serves them, open stores "
                "as large labelled dots and unopened candidate sites as hollow circles.",
        )
        st.caption(
            "Large dots: open dark stores, with ID labels. Hollow circles: candidate sites that were not opened. "
            "Small dots: customers, coloured by their serving store; grey means not served. Rings show the max "
            f"distance as a straight-line radius ({params.max_distance_km:g} km road ÷ {params.circuity:g} "
            "circuity). Hover over any point for details."
        )

    st.subheader("Opened dark stores")
    show = stores.drop(columns=["lat", "lon"]).assign(
        fixed_cost=lambda t: t["fixed_cost"] / 1e5,
        contribution=lambda t: t["contribution"] / 1e5,
        store_profit=lambda t: t["store_profit"] / 1e5,
    )
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "store": "Store", "locality": "Nearest locality", "customers": "Customers",
            "orders": st.column_config.NumberColumn("Orders / month", format="%.0f"),
            "avg_km": st.column_config.NumberColumn("Avg distance (km)", format="%.2f"),
            "utilization_pct": st.column_config.NumberColumn("Utilization (%)", format="%.1f"),
            "fixed_cost": st.column_config.NumberColumn("Fixed cost (₹ lakh)", format="%.2f"),
            "contribution": st.column_config.NumberColumn("Margin − delivery (₹ lakh)", format="%.2f"),
            "store_profit": st.column_config.NumberColumn("Store profit (₹ lakh)", format="%.2f"),
        },
    )
    assignments = cust[["customer", "lat", "lon", "orders"]].assign(
        store=np.where(served, store_names[cust["store_idx"].clip(lower=0)], ""),
        distance_km=cust["distance_km"].round(3),
    )
    st.download_button("Download customer assignments (CSV)", assignments.to_csv(index=False),
                       file_name="assignments.csv", mime="text/csv")

# ---------------------------------------------------------------- tab 2: profit vs P
with tab_sweep:
    st.subheader(f"Profit when at most P dark stores may open ({SWEEP_CUSTOMERS:,} customers, "
                 f"{SWEEP_CANDIDATES} candidate sites)")
    st.caption(
        f"All runs use the same instance (seed {seed}) and the max distance ({params.max_distance_km:g} km) and "
        "economics from the sidebar. Only P changes."
    )
    run_now = params.capacity is None or st.button(
        "Run sweep", type="primary",
        help=f"Capacity mode is on: each of the {len(SWEEP_P)} solves can take up to the time limit.")
    if not run_now:
        st.info("Capacity mode is slower, so the sweep runs on demand.")
    else:
        sweep_cust, _ = get_instance(SWEEP_CUSTOMERS, SWEEP_CANDIDATES, int(seed))
        progress = st.progress(0.0, text="Solving…")
        rows = []
        for n, P in enumerate(SWEEP_P):
            p = replace(params, max_stores=P)
            s = run_model(SWEEP_CUSTOMERS, SWEEP_CANDIDATES, int(seed), p)
            r = summarize(sweep_cust, s, p)
            rows.append({
                "Max stores allowed (P)": P,
                "Stores opened": r["stores_open"],
                "Profit (₹ lakh/month)": r["profit"] / 1e5,
                "Customers served (%)": 100 * r["customers_served"] / r["customers_total"],
                "Avg distance (km)": r["avg_km"],
                "Fixed cost (₹ lakh)": r["fixed_cost"] / 1e5,
                "Solver status": r["status"],
                "MIP gap (%)": 100 * r["mip_gap"],
                "Solve time (s)": r["solve_time_s"],
            })
            progress.progress((n + 1) / len(SWEEP_P), text=f"Solved P = {P}")
        progress.empty()
        sweep = pd.DataFrame(rows)

        fig = go.Figure(go.Scatter(
            x=sweep["Max stores allowed (P)"], y=sweep["Profit (₹ lakh/month)"],
            mode="lines+markers", line=dict(color=palette[0], width=2),
            marker=dict(size=10, color=palette[0], line=dict(width=2, color=SURFACE[theme])),
            customdata=sweep[["Stores opened", "Customers served (%)"]].to_numpy(),
            hovertemplate="P ≤ %{x}<br>Profit ₹%{y:.2f} lakh/month<br>%{customdata[0]} stores opened"
                          "<br>%{customdata[1]:.1f}% customers served<extra></extra>",
        ))
        plateau = sweep[sweep["Stores opened"] < sweep["Max stores allowed (P)"]]
        if len(plateau):
            first = plateau.iloc[0]
            fig.add_annotation(
                x=first["Max stores allowed (P)"], y=first["Profit (₹ lakh/month)"],
                text=f"Plateau: only {first['Stores opened']} stores are worth opening",
                showarrow=True, arrowhead=0, ax=0, ay=40, font=dict(color=ink, size=12),
            )
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Max dark stores allowed to open (P)", tickvals=list(SWEEP_P), showgrid=False),
            yaxis=dict(title="Profit (₹ lakh / month)", rangemode="tozero"),
        )
        st.plotly_chart(fig, theme="streamlit")
        st.dataframe(
            sweep, hide_index=True, width="stretch",
            column_config={
                "Profit (₹ lakh/month)": st.column_config.NumberColumn(format="%.2f"),
                "Customers served (%)": st.column_config.NumberColumn(format="%.1f"),
                "Avg distance (km)": st.column_config.NumberColumn(format="%.2f"),
                "Fixed cost (₹ lakh)": st.column_config.NumberColumn(format="%.2f"),
                "MIP gap (%)": st.column_config.NumberColumn(format="%.3f"),
                "Solve time (s)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        gains = sweep["Profit (₹ lakh/month)"].diff().iloc[1:]
        steps = [f"{a}→{b}: ₹{g / (b - a):.2f} lakh" for a, b, g in zip(SWEEP_P[:-1], SWEEP_P[1:], gains)]
        st.markdown(
            "**Reading the curve.** Raising P only enlarges the feasible set, so the optimal profit can never "
            "fall. The gain per extra allowed store shrinks as P grows (" + ", ".join(steps) + "): the first "
            "stores take the densest areas, and later ones reach thinner suburbs. Once the allowance exceeds the "
            "number of stores that pay for their fixed cost, the extra slots go unused and the curve flattens."
        )
        st.download_button("Download sweep results (CSV)", sweep.to_csv(index=False),
                           file_name="profit_vs_P.csv", mime="text/csv")

        if params.capacity is None:
            st.subheader(f"Effect of the max-distance constraint (P = {params.max_stores})")
            rows = []
            for dmax in SWEEP_DMAX:
                p = replace(params, max_distance_km=dmax)
                r = summarize(sweep_cust, run_model(SWEEP_CUSTOMERS, SWEEP_CANDIDATES, int(seed), p), p)
                rows.append({"Max distance (km)": dmax, "Profit (₹ lakh/month)": r["profit"] / 1e5,
                             "Stores opened": r["stores_open"],
                             "Customers served (%)": 100 * r["customers_served"] / r["customers_total"]})
            dsweep = pd.DataFrame(rows)
            fig2 = go.Figure(go.Scatter(
                x=dsweep["Max distance (km)"], y=dsweep["Profit (₹ lakh/month)"], mode="lines+markers",
                line=dict(color=palette[0], width=2),
                marker=dict(size=9, color=palette[0], line=dict(width=2, color=SURFACE[theme])),
                customdata=dsweep[["Stores opened", "Customers served (%)"]].to_numpy(),
                hovertemplate="Max distance %{x} km<br>Profit ₹%{y:.2f} lakh/month<br>%{customdata[0]} stores "
                              "opened<br>%{customdata[1]:.1f}% customers served<extra></extra>",
            ))
            fig2.add_vline(x=params.max_distance_km, line=dict(color=MUTED, width=1, dash="dot"))
            fig2.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Max distance store ↔ customer (road km)", showgrid=False),
                yaxis=dict(title="Profit (₹ lakh / month)", rangemode="tozero"),
            )
            st.plotly_chart(fig2, theme="streamlit")
            st.caption(
                f"A tighter radius shrinks the feasible set, so profit falls as the limit tightens. Beyond "
                f"m ÷ c = {params.margin_per_order / max(params.cost_per_km, 1e-9):.1f} km a delivery loses money "
                "anyway, so the constraint stops binding there. The dotted line marks the current setting."
            )

# ---------------------------------------------------------------- tab 3: model
with tab_model:
    st.subheader("Mathematical model")
    st.markdown(
        "**Sets.** Customers $i \\in I$ and candidate sites $j \\in J$.  \n"
        "**Parameters.** $w_i$ orders/month; margin per order $m$ = order value × margin %; delivery cost $c$ "
        "per order per km; road distance $d_{ij}$; fixed cost $F_j$; store limit $P$; max distance "
        "$D_{\\max}$; optional capacity $C$. The monthly contribution of serving $i$ from $j$ is "
        "$\\pi_{ij} = w_i\\,(m - c\\,d_{ij})$.  \n"
        "**Variables.** $x_j \\in \\{0,1\\}$ opens site $j$; $y_{ij} \\in [0,1]$ is the share of customer "
        "$i$'s demand served by $j$."
    )
    st.latex(r"\max \; Z=\sum_{i\in I}\sum_{j\in J}\pi_{ij}\,y_{ij}-\sum_{j\in J}F_j\,x_j")
    st.latex(r"""\begin{aligned}
    &\textstyle\sum_{j} y_{ij} \le 1 && \forall i && \text{(1) each customer served at most once}\\
    &y_{ij} \le x_j && \forall i,j && \text{(2) only open stores serve}\\
    &\textstyle\sum_{i} w_i\,y_{ij} \le C\,x_j && \forall j && \text{(3) capacity (optional)}\\
    &\textstyle\sum_{j} x_j \le P && && \text{(4) at most } P \text{ stores}\\
    &y_{ij} = 0 && \forall (i,j):\ d_{ij} > D_{\max} && \text{(5) max distance (enhancement)}\\
    &x_j \in \{0,1\},\ 0 \le y_{ij} \le 1
    \end{aligned}""")
    st.markdown(
        """
**The enhancement (5).** A customer may only be served by a store within $D_{\\max}$. It is implemented by
*variable elimination*: $y_{ij}$ is never created for pairs beyond $D_{\\max}$. This is equivalent to
$y_{ij} \\le a_{ij} x_j$ with $a_{ij} = [d_{ij} \\le D_{\\max}]$, and it shrinks the model. Note that the
aggregated form $\\sum_j d_{ij} y_{ij} \\le D_{\\max}$ would be **wrong** when $y$ is fractional, because it only
limits the *average* distance. Constraint (1) uses $\\le 1$ rather than $= 1$. That keeps the model feasible for
any $P$ and $D_{\\max}$: customers with no open store in range simply go unserved. With $= 1$, P = 2 would be
infeasible for a city the size of Chennai.

**Pre-processing.** Pairs with $\\pi_{ij} \\le 0$ (deliveries farther than $m/c$ km) are dropped too, since
serving them can never raise profit.

**How it is solved.**
* *Uncapacitated (default): exact compact reformulation.* Sort customer $i$'s feasible stores nearest-first, so
  that $\\pi_{i(1)} \\ge \\pi_{i(2)} \\ge \\dots$. If $S$ is the set of open stores, the customer's profit is
  $\\max_{j \\in S}\\pi_{ij} = \\sum_k (\\pi_{i(k)} - \\pi_{i(k+1)})\\,[S \\cap \\{(1),\\dots,(k)\\} \\ne \\emptyset]$.
  Each indicator becomes a variable $z \\le \\sum_{j \\in \\text{k nearest}} x_j$. Every customer–rank pair
  that shares the same set of nearest stores shares one $z$, with the coefficients summed. This shrinks
  about $N \\times M$ assignment variables to a few thousand, keeps the optimum unchanged, and gives an LP
  relaxation as tight as the strong $y_{ij} \\le x_j$ formulation. 10,000 × 50 instances solve to proven
  optimality in well under a second.
* *Capacitated (optional): pair-level model (1), (3), (4), (5).* Capacity (3) already forces $y_{ij} = 0$
  when $x_j = 0$, so the per-pair rows (2) are left out. They would tighten the LP, but they make it far slower
  to solve. Large capacitated instances can therefore stop at the time limit, and the app reports the MIP gap.
* Both are solved with **HiGHS** via `scipy.optimize.milp`, using branch-and-bound on the LP relaxation with
  presolve, cutting planes and primal heuristics. The MIP gap is |best bound − incumbent| / |incumbent|.

**Properties.** Profit is non-decreasing in $P$ and in $D_{\\max}$, because each increase only enlarges the
feasible set. Because (4) says *at most* $P$, the optimum may open fewer than $P$ stores. That is why the
profit curve flattens. Facility location is NP-hard (set cover reduces to it), which is why we need MILP
rather than enumeration: $\\binom{50}{10} \\approx 10^{10}$ subsets.
"""
    )
    st.subheader("Assumptions")
    st.dataframe(pd.DataFrame([
        ("Geography", "Chennai: lat 12.86–13.20, lon 80.09 to the coastline; sea points rejected"),
        ("Customers", "75% clustered (σ ≈ 1.2 km) around 28 residential localities, 25% uniform; random seed fixed"),
        ("Demand", "2–10 orders per customer per month (uniform, mean 6)"),
        ("Candidate sites", "Drawn near localities (σ ≈ 1.5 km), independent of the customer draw"),
        ("Distance", "Haversine great-circle distance × road circuity factor (default 1.3)"),
        ("Margin per order", f"₹{params.order_value:,.0f} order value × {params.margin_pct:g}% = "
                             f"₹{params.margin_per_order:,.0f}"),
        ("Delivery cost", f"₹{params.cost_per_km:g} per order per road km (rider, fuel)"),
        ("Fixed cost", f"₹{params.fixed_cost_min:,.0f}–₹{params.fixed_cost_max:,.0f} per month "
                       "(rent, staff, utilities), higher near the city centre"),
        ("Capacity", "Unlimited by default (a dark store handles far more orders than this demand); optional limit"),
        ("Planning horizon", "One month; all money in ₹ per month"),
    ], columns=["Item", "Assumption"]), hide_index=True, width="stretch")
