import streamlit as st
import pandas as pd

from solver import PricingStrategyOptimizer
from solver.data import DataManager, PricingConstraints, ConstraintBlock

st.title("Pricing Strategy Optimization")

# Initialize session variables if missing
for key in [
    "opt_df", "opt_stats",
    "manual_selection", "manual_compare_df", "manual_inputs_df"
]:
    if key not in st.session_state:
        st.session_state[key] = None


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

if dm is None:
    st.info("Please upload a valid file to continue.")
    st.stop()


# ----------------------------------------------------------
# Tabs: Optimization | Manual Search
# ----------------------------------------------------------

tab_opt, tab_manual = st.tabs(["Optimization", "Manual Search"])


# ----------------------------------------------------------
# TAB 1: OPTIMIZATION
# ----------------------------------------------------------

with tab_opt:

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
            overall_kwargs["min_conversion_rate"] = st.number_input(
                "min_conversion_rate:",
                value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_comp:
            overall_kwargs["min_completion_rate"] = st.number_input(
                "min_completion_rate:",
                value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_take:
            overall_kwargs["min_take_rate"] = st.number_input(
                "min_take_rate:",
                value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_max_take:
            overall_kwargs["max_take_rate"] = st.number_input(
                "max_take_rate:",
                value=1.0, min_value=0.0, max_value=1.0
            )
        if oc_min_icr:
            overall_kwargs["min_icr"] = st.number_input(
                "min_icr:",
                value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_rpi:
            overall_kwargs["min_rpi"] = st.number_input(
                "min_rpi:",
                value=0.0
            )

        if overall_kwargs:
            overall_block = ConstraintBlock(**overall_kwargs)
        else:
            overall_block = ConstraintBlock()

    # ---- 3. Per-bin constraints ----
    bin_names = dm.distance_vals

    per_bin_constraints = {}
    bin_tabs = st.tabs(bin_names)

    for bin_tab, bin_name in zip(bin_tabs, bin_names):
        with bin_tab:
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

    # ---- 4. Build final constraints ----
    constraints = PricingConstraints(
        objective=objective[:3],
        overall=overall_block,
        per_bin=per_bin_constraints
    )

    st.write("### Final collected constraints:")
    st.json(constraints.model_dump())

    # ----------------------------------------------------------
    # Step 3: Run optimization
    # ----------------------------------------------------------

    st.write("### Step 3: Run optimization")

    if st.button("Find Optimal Strategy"):
        try:
            st.write("Running solver... please wait.")

            solver = PricingStrategyOptimizer(dm, constraints)
            ok = solver.solve()

            if not ok:
                st.error("No strategy found. Adjust constraints.")
                st.stop()

            df_opt, stats = solver.prepare_solution()

            # Save results in session state
            st.session_state["opt_df"] = df_opt
            st.session_state["opt_stats"] = stats

            st.success("Optimization completed!")

        except Exception as e:
            st.error(f"Solver error: {e}")

    # ----------------------------------------------------------
    # Restore previous results if exist
    # ----------------------------------------------------------
    if st.session_state["opt_df"] is not None:
        st.write("### Previous Optimal Strategy")
        st.dataframe(st.session_state["opt_df"])

    if st.session_state["opt_stats"] is not None:
        st.write("### Previous Stats Comparison")
        st.dataframe(st.session_state["opt_stats"])



# ----------------------------------------------------------
# TAB 2: MANUAL SEARCH
# ----------------------------------------------------------

with tab_manual:

    st.write("### Manual Strategy Search")
    st.write("Select client and courier diffs for each distance bin.")

    # Load previous manual table if exists
    if st.session_state["manual_inputs_df"] is not None:
        manual_df = st.session_state["manual_inputs_df"]
    else:
        manual_df = pd.DataFrame({
            "Distance": dm.distance_vals,
            "Client diff": [0.0] * dm.n_bins,
            "Courier diff": [0.0] * dm.n_bins,
        })

    # Editable table
    edited_df = st.data_editor(
        manual_df,
        hide_index=True,
        column_config={
            "Client diff": st.column_config.SelectboxColumn(
                "Client diff",
                options=dm.client_diffs
            ),
            "Courier diff": st.column_config.SelectboxColumn(
                "Courier diff",
                options=[float(i) for i in dm.courier_diffs]
            ),
        }
    )

    # Persist edited table
    st.session_state["manual_inputs_df"] = edited_df

    if st.button("Evaluate Manual Strategy"):

        try:
            manual_selection = {
                row["Distance"]: (row["Client diff"], row["Courier diff"])
                for _, row in edited_df.iterrows()
            }

            default_strategy = {
                dist: (0.0, 0.0) for dist in dm.distance_vals
            }

            manual_stats = dm.calculate_strategy(manual_selection)
            default_stats = dm.calculate_strategy(default_strategy)

            df_compare = pd.DataFrame({
                "Manual Strategy": manual_stats,
                "Default Strategy": default_stats
            })

            # Save results
            st.session_state["manual_selection"] = manual_selection
            st.session_state["manual_compare_df"] = df_compare

            st.success("Manual strategy evaluated.")

        except Exception as e:
            st.error(f"Error evaluating manual strategy: {e}")

    # Restore previous results
    if st.session_state["manual_compare_df"] is not None:
        st.write("### Previous Manual Comparison")
        st.dataframe(st.session_state["manual_compare_df"])
