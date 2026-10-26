"""Headless smoke test of the Streamlit app. Run: python tests/check_app.py"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run(label, **inputs):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    if inputs:
        for key, value in inputs.items():
            widgets = [*at.number_input, *at.slider, *at.checkbox]
            widget = next(w for w in widgets if w.label.startswith(key))
            widget.set_value(value)
        at.button[0].click()  # form submit
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    kpis = at.dataframe[0].value
    profit = kpis.loc[kpis["KPI"] == "Net profit", "Value"].iloc[0]
    status = kpis.loc[kpis["KPI"] == "Solver status", "Value"].iloc[0]
    print(f"  {label}: profit {profit}, status {status}, {len(at.dataframe)} tables, warnings {len(at.warning)}")
    return at


if __name__ == "__main__":
    run("defaults")
    run("tiny + tight radius", Customers=500, **{"Dark store candidate sites": 5, "Max distance": 1.0})
    run("P larger than candidates", **{"Dark store candidate sites": 10, "Max dark stores": 40})
    run("capacitated", Customers=2000, **{"Limit orders per store": True, "Capacity": 1500})
    print("App checks passed.")
