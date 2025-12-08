import streamlit as st
import pandas as pd

from solver import PricingStrategyOptimizer
from solver.data import DataManager, PricingConstraints, ConstraintBlock

st.title("Pricing Strategy Optimization")

# ----------------------------------------------------------
# Step 1. Upload file
# ----------------------------------------------------------

st.write("### Step 1. Upload Excel file with data")

uploaded_file = st.file_uploader("Choose file", type=["xlsx"])
dm = None

if uploaded_file is not None:
    st.write("### File is uploaded. Reading data...")

    try:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        try:
            dm = DataManager(df)
            st.success("Data is valid")

        except Exception as e:
            st.error(f"Error creating DataManager: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# If no file uploaded — STOP here; hide Step 2 and 3 entirely
if dm is None:
    st.info("Please upload a valid file to continue with Step 2 and Step 3.")
    st.stop()


# ----------------------------------------------------------
# Step 2: Constraints
# ----------------------------------------------------------

st.write("### Step 2: Set optimization constraints")

# ---- 1. Select Objective ----
objective = st.radio(
    "Optimization objective:",
    ["rpi - Revenue per Intent", "icr - Intent to Completed Rate"],
    index=0,
    horizontal=True
)

# ---- 2. Overall constraints ----
overall_block = None

with st.expander("Overall constraints (optional)"):
    st.write("If you leave fields disabled, they remain None (no constraint).")

    oc_min_conv = st.checkbox("Enable min demand intent conversion rate")
    oc_min_comp = st.checkbox("Enable min order completion rate")
    oc_min_take = st.checkbox("Enable min take rate (commission)")
    oc_max_take = st.checkbox("Enable max take rate (commission)")
    oc_min_icr  = st.checkbox("Enable min icr (used ONLY when optimizing for rpi)")
    oc_min_rpi  = st.checkbox("Enable min rpi (used ONLY when optimizing for icr)")

    overall_kwargs = {}

    if oc_min_conv:
        overall_kwargs["min_conversion_rate"] = st.number_input("min_conversion_rate:", value=0.0, min_value=0.0, max_value=1.0)
    if oc_min_comp:
        overall_kwargs["min_completion_rate"] = st.number_input("min_completion_rate:", value=0.0, min_value=0.0, max_value=1.0)
    if oc_min_take:
        overall_kwargs["min_take_rate"] = st.number_input("min_take_rate:", value=0.0, min_value=0.0, max_value=1.0)
    if oc_max_take:
        overall_kwargs["max_take_rate"] = st.number_input("max_take_rate:", value=1.0, min_value=0.0, max_value=1.0)
    if oc_min_icr:
        overall_kwargs["min_icr"] = st.number_input("min_icr:", value=0.0, min_value=0.0, max_value=1.0)
    if oc_min_rpi:
        # RPI is not a 0..1 metric → leave unrestricted
        overall_kwargs["min_rpi"] = st.number_input("min_rpi:", value=0.0)

    if overall_kwargs:
        overall_block = ConstraintBlock(**overall_kwargs)
    else:
        overall_block = ConstraintBlock()


# ---- 3. Per-bin constraints ----
bin_names = ["00 - 04 km", "04 - 08 km", "08 - 14 km", "14 - 20 km", "20+ km"]

per_bin_constraints = {}
tabs = st.tabs(bin_names)

for tab, bin_name in zip(tabs, bin_names):
    with tab:
        st.write(f"Constraints for **{bin_name}**")
        st.write("Leave fields disabled for None (i.e., no constraint).")

        cb_min_conv = st.checkbox(f"{bin_name}: min_conversion_rate", key=f"{bin_name}_mc")
        cb_min_comp = st.checkbox(f"{bin_name}: min_completion_rate", key=f"{bin_name}_mp")
        cb_min_take = st.checkbox(f"{bin_name}: min_take_rate", key=f"{bin_name}_mt")
        cb_max_take = st.checkbox(f"{bin_name}: max_take_rate", key=f"{bin_name}_xt")

        kwargs_bin = {}

        if cb_min_conv:
            kwargs_bin["min_conversion_rate"] = st.number_input(
                f"{bin_name} min_conversion_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key=f"{bin_name}_minconv_val"
            )
        if cb_min_comp:
            kwargs_bin["min_completion_rate"] = st.number_input(
                f"{bin_name} min_completion_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key=f"{bin_name}_mincomp_val"
            )
        if cb_min_take:
            kwargs_bin["min_take_rate"] = st.number_input(
                f"{bin_name} min_take_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key=f"{bin_name}_mintake_val"
            )
        if cb_max_take:
            kwargs_bin["max_take_rate"] = st.number_input(
                f"{bin_name} max_take_rate",
                value=1.0, min_value=0.0, max_value=1.0,
                key=f"{bin_name}_maxtake_val"
            )

        if kwargs_bin:
            per_bin_constraints[bin_name] = ConstraintBlock(**kwargs_bin)


# ---- 4. Build final constraint object ----

constraints = PricingConstraints(
    objective=objective[:3],
    overall=overall_block,
    per_bin=per_bin_constraints
)

st.write("### Final collected constraints:")
st.json(constraints.model_dump())


# ----------------------------------------------------------
# Step 3: Find Optimal Strategy
# ----------------------------------------------------------

st.write("### Step 3: Run optimization")

if st.button("Find Optimal Strategy"):
    try:
        st.write("Running solver... please wait.")

        solver = PricingStrategyOptimizer(dm, constraints)

        # IMPORTANT: solve() returns True or False
        ok = solver.solve()

        if not ok:
            st.error("No strategy found. Adjust constraints.")
            st.stop()

        df_opt, stats = solver.prepare_solution()

        st.success("Optimization completed!")

        # -----------------------------
        # Show Optimal Strategy Table
        # -----------------------------
        st.write("### Optimal Strategy (per-bin choices)")
        st.dataframe(df_opt)

        opt_csv = df_opt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Optimal Strategy (CSV)",
            opt_csv,
            file_name="optimal_strategy.csv",
            mime="text/csv"
        )

        # -----------------------------
        # Show Stats Comparison Table
        # -----------------------------
        st.write("### Comparison: Optimal vs Default Strategy")
        st.dataframe(stats)

        stats_csv = stats.to_csv().encode("utf-8")
        st.download_button(
            "Download Stats (CSV)",
            stats_csv,
            file_name="strategy_stats.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Solver error: {e}")
