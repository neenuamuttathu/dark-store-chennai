# Dark store location optimizer: Chennai

A Streamlit app that chooses which candidate sites to open as dark stores, and which store serves each
customer, so that monthly profit is as high as possible. It respects two limits: **at most P stores**, and
**no customer served from a store farther than a maximum distance**. The model is a mixed-integer linear
program solved exactly with HiGHS (`scipy.optimize.milp`).

## Model

Profit-maximising facility location, enhanced with a maximum-distance constraint (5):

```
max  Σ_ij π_ij y_ij − Σ_j F_j x_j                 π_ij = w_i (m − c·d_ij)
s.t. Σ_j y_ij ≤ 1            ∀i                    (1) each customer served at most once
     y_ij ≤ x_j              ∀i,j                  (2) only open stores serve
     Σ_i w_i y_ij ≤ C x_j    ∀j                    (3) capacity (optional)
     Σ_j x_j ≤ P                                   (4) at most P stores
     y_ij = 0  if d_ij > D_max                     (5) max distance (enhancement)
     x_j ∈ {0,1}, 0 ≤ y_ij ≤ 1
```

- **(5)** is implemented by variable elimination: no `y_ij` is created for pairs beyond `D_max`.
- **Without capacity (the default)**, the model is solved in an exact compact form. Customers' sorted
  "nearest-k store sets" are merged, so the 10,000 × 50 model shrinks to about 2,000 variables and solves to
  proven optimality in under a second.
- **With capacity**, the pair-level model is solved directly. It is slower, and large instances may stop at
  the time limit; the app reports the MIP gap.

Full theory and assumptions are in the app's **Model & assumptions** tab.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: inputs, KPI table, map, profit-vs-P sweep |
| `model.py` | MILP formulations, solver call, KPIs |
| `data.py` | Synthetic Chennai customers and candidate sites (fixed seed) |
| `experiment.py` | Offline run of the required P sweep; writes `results/profit_vs_P.csv` and `.png` |
| `tests/check_model.py` | Brute-force optimality check and invariants (max distance, P, monotonicity) |
| `tests/check_app.py` | Headless smoke test of the app |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Run `python tests/check_model.py` and `python tests/check_app.py` for the checks. `experiment.py` also
needs `matplotlib`.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app → Deploy a public app
   from GitHub**.
3. Pick the repository, branch `main` and main file `app.py`. Under Advanced settings, choose Python 3.12.
   Then deploy.
