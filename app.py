import math
import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Helpers
# ----------------------------
def _to_float(x):
    """Extracts the first numeric token from a cell; returns NaN if none."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan


def load_units_from_excel(file) -> pd.DataFrame:
    """
    Loads the unit library from the provided Excel and normalizes fields.
    Handles columns like Pricing / Pricing.1 and Term / Term.1.
    """
    df = pd.read_excel(file, sheet_name=0, engine="openpyxl")

    # Consolidate rate + term if sheet has split columns
    if "Pricing.1" in df.columns:
        df["Rate_raw"] = df["Pricing"].where(df["Pricing"].notna(), df["Pricing.1"])
    else:
        df["Rate_raw"] = df["Pricing"] if "Pricing" in df.columns else np.nan

    if "Term.1" in df.columns:
        df["Term_raw"] = df["Term"].where(df["Term"].notna(), df["Term.1"])
    else:
        df["Term_raw"] = df["Term"] if "Term" in df.columns else np.nan

    df["Rate_USD_per_month"] = df["Rate_raw"].apply(_to_float)
    df["Term_months"] = df["Term_raw"].apply(_to_float)

    # Clean text fields
    for col in ["Unit Type", "Drive", "Vendor"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "Advantages" in df.columns:
        df["Advantages"] = df["Advantages"].fillna("Standard").astype(str).str.strip()
    else:
        df["Advantages"] = "Standard"

    # Build Unit_ID
    df["Unit_ID"] = (
        df.get("Unit Type", "").astype(str)
        + " | " + df.get("Drive", "").astype(str)
        + " | " + df.get("Vendor", "").astype(str)
        + " | " + df["Advantages"].astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Ensure numeric flow columns
    for col in ["Volume Min (MMSCFD)", "Volume Max (MMSCFD)",
                "Suction Low psi", "Suction High psi",
                "Discharge Low psi", "Discharge High psi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "Unit_ID", "Unit Type", "Drive", "Vendor", "Advantages",
        "Rate_raw", "Rate_USD_per_month", "Term_months",
        "Availability",
        "Suction Low psi", "Suction High psi",
        "Discharge Low psi", "Discharge High psi",
        "Volume Min (MMSCFD)", "Volume Max (MMSCFD)",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def filter_eligible_units(units: pd.DataFrame, suction_psig: float, discharge_psig: float,
                          strict_envelopes: bool) -> pd.DataFrame:
    """
    Filters units that can operate at (suction, discharge) based on provided envelopes.
    strict_envelopes=True: requires all envelope fields present and in-range
    strict_envelopes=False: allows missing envelope fields but flags them
    """
    u = units.copy()

    # Must have max flow and numeric rate to do economics
    u = u[u["Volume Max (MMSCFD)"].notna()]
    u = u[u["Rate_USD_per_month"].notna()]

    # Envelope checks
    env_cols = ["Suction Low psi", "Suction High psi", "Discharge Low psi", "Discharge High psi"]
    for c in env_cols:
        if c not in u.columns:
            u[c] = np.nan

    def env_status(row):
        sl, sh, dl, dh = row["Suction Low psi"], row["Suction High psi"], row["Discharge Low psi"], row["Discharge High psi"]
        if pd.isna(sl) or pd.isna(sh) or pd.isna(dl) or pd.isna(dh):
            return "Unknown"
        ok = (sl <= suction_psig <= sh) and (dl <= discharge_psig <= dh)
        return "OK" if ok else "Out"

    u["Envelope_Status"] = u.apply(env_status, axis=1)

    if strict_envelopes:
        u = u[u["Envelope_Status"] == "OK"]
    else:
        u = u[u["Envelope_Status"].isin(["OK", "Unknown"])]

    return u


def recommend_for_scenarios(units_eligible: pd.DataFrame, scenarios: pd.DataFrame,
                            top_n: int = 3) -> pd.DataFrame:
    """
    For each scenario row, compute units needed, capacity, cost, and rank by total monthly rental.
    """
    results = []

    for _, s in scenarios.iterrows():
        pad = s.get("Pad", "")
        option = s.get("Option", "")
        stage = s.get("Stage", "")
        req_flow = float(s.get("Required Flow (MMSCFD)", np.nan))
        availability = float(s.get("Availability (%)", 100.0)) if not pd.isna(s.get("Availability (%)", np.nan)) else 100.0

        if pd.isna(req_flow) or req_flow <= 0:
            continue

        for _, u in units_eligible.iterrows():
            vol_max = float(u["Volume Max (MMSCFD)"])
            vol_min = float(u["Volume Min (MMSCFD)"]) if not pd.isna(u.get("Volume Min (MMSCFD)", np.nan)) else np.nan
            rate = float(u["Rate_USD_per_month"])

            n_units = int(math.ceil(req_flow / vol_max))
            cap_max = n_units * vol_max
            cap_min = n_units * vol_min if not pd.isna(vol_min) else np.nan

            oversized = (not pd.isna(cap_min)) and (req_flow < cap_min)
            eff_inj = min(req_flow, cap_max) * (availability / 100.0)

            results.append({
                "Pad": pad,
                "Option": option,
                "Stage": stage,
                "Required Flow (MMSCFD)": req_flow,
                "Availability (%)": availability,
                "Effective Injection (MMSCFD)": eff_inj,

                "Unit_ID": u["Unit_ID"],
                "Unit Type": u.get("Unit Type", ""),
                "Vendor": u.get("Vendor", ""),
                "Drive": u.get("Drive", ""),
                "Envelope_Status": u.get("Envelope_Status", ""),

                "Units Needed": n_units,
                "Total Capacity Max (MMSCFD)": cap_max,
                "Total Capacity Min (MMSCFD)": cap_min,

                "Unit Rate ($/mo)": rate,
                "Total Rental ($/mo)": n_units * rate,
                "Oversized Flag": "Yes" if oversized else "",
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["OversizedRank"] = (out["Oversized Flag"] == "Yes").astype(int)
    out = out.sort_values(["Pad", "Stage", "OversizedRank", "Total Rental ($/mo)"])

    out = out.groupby(["Pad", "Option", "Stage"], as_index=False).head(top_n).drop(columns=["OversizedRank"])
    return out


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Compressor Optimization", layout="wide")
st.title("Compressor Optimization Dashboard (Streamlit)")

st.markdown(
    """
Use this app to pick the best compressor rental setup given:
- Suction / discharge requirements
- Required injection rate (MMSCFD)
- Optional availability (%)

It filters the unit library by operating envelope (when available) and ranks options by monthly rental cost.
"""
)

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload compressor unit Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload your compressor unit file to begin.")
    st.stop()

units = load_units_from_excel(uploaded)

st.sidebar.subheader("Pressure Requirements")
suction_psig = st.sidebar.number_input("Suction (psig)", value=60.0, step=1.0)
discharge_psig = st.sidebar.number_input("Discharge (psig)", value=1250.0, step=10.0)

strict_envelopes = st.sidebar.toggle("Strict envelope filtering (require full envelope)", value=True)
top_n = st.sidebar.slider("Top N recommendations per scenario", min_value=1, max_value=10, value=3)

st.subheader("Scenario Inputs (multi-pad / multi-option / multi-stage)")
st.caption("Enter one row per Pad + Option + Stage. Required Flow should be in MMSCFD.")

default_scenarios = pd.DataFrame([
    {"Pad": "Centralized 9 Wells", "Option": "Option A", "Stage": "POP", "Required Flow (MMSCFD)": 10.80, "Availability (%)": 100},
    {"Pad": "West Pad", "Option": "Option B", "Stage": "POP", "Required Flow (MMSCFD)": 5.40, "Availability (%)": 100},
    {"Pad": "East Pad", "Option": "Option B", "Stage": "POP", "Required Flow (MMSCFD)": 5.40, "Availability (%)": 100},
])

scenarios = st.data_editor(default_scenarios, num_rows="dynamic", use_container_width=True)

eligible = filter_eligible_units(units, suction_psig, discharge_psig, strict_envelopes)

c1, c2, c3 = st.columns(3)
c1.metric("Units in Library", len(units))
c2.metric("Eligible Units", len(eligible))
c3.metric("Strict Mode", "ON" if strict_envelopes else "OFF")

st.subheader("Recommendations")
recs = recommend_for_scenarios(eligible, scenarios, top_n=top_n)

if recs.empty:
    st.warning("No recommendations found. Try turning OFF strict filtering or check envelope/price/flow data.")
    st.stop()

st.dataframe(recs, use_container_width=True)

st.subheader("Download Results")
st.download_button(
    "Download recommendations as CSV",
    data=recs.to_csv(index=False).encode("utf-8"),
    file_name="compressor_recommendations.csv",
    mime="text/csv"
)
