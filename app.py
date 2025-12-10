import streamlit as st
import pandas as pd

from solver import PricingStrategyOptimizer
from solver.data import DataManager, RawDataProcessor, PricingConstraints, ConstraintBlock

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

    st.write("### File is uploaded. Validating and processing...")

    try:
        # Run RawDataProcessor
        processor = RawDataProcessor(uploaded_file)
        df_merged = processor.process()

        st.success("File structure is valid. Data successfully merged.")

        st.write("### Merged scenario table")
        st.dataframe(df_merged)

        st.session_state["df_raw"] = uploaded_file
        st.session_state["df_merged"] = df_merged

        # Initialize DataManager
        try:
            dm = DataManager(df_merged)
            st.session_state["dm"] = dm
            st.success("DataManager initialized successfully.")
        except Exception as e:
            st.error(f"Error creating DataManager: {e}")


    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # no file uploaded yet
    st.info("Please upload an Excel file with Client data, Courier data, and Weights sheets.")
    st.stop()

# If file was uploaded but DataManager failed to initialize
if uploaded_file is not None and dm is None:
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
                "min_conversion_rate:", value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_comp:
            overall_kwargs["min_completion_rate"] = st.number_input(
                "min_completion_rate:", value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_take:
            overall_kwargs["min_take_rate"] = st.number_input(
                "min_take_rate:", value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_max_take:
            overall_kwargs["max_take_rate"] = st.number_input(
                "max_take_rate:", value=1.0, min_value=0.0, max_value=1.0
            )
        if oc_min_icr:
            overall_kwargs["min_icr"] = st.number_input(
                "min_icr:", value=0.0, min_value=0.0, max_value=1.0
            )
        if oc_min_rpi:
            overall_kwargs["min_rpi"] = st.number_input("min_rpi:", value=0.0)

        if overall_kwargs:
            overall_block = ConstraintBlock(**overall_kwargs)
        else:
            overall_block = ConstraintBlock()

    # ---- 3. Per-bin constraints (same for all distance bins) ----
    per_bin_constraints = {}

    with st.expander("Per-bin constraints (applied to all distance bins equally, optional)"):
        st.write(
            "If you enable a constraint here, it will be applied with the same value "
            "to every distance bin."
        )

        pb_min_conv = st.checkbox("Enable per-bin min_conversion_rate")
        pb_min_comp = st.checkbox("Enable per-bin min_completion_rate")
        pb_min_take = st.checkbox("Enable per-bin min_take_rate (commission)")
        pb_max_take = st.checkbox("Enable per-bin max_take_rate (commission)")

        per_bin_kwargs = {}

        if pb_min_conv:
            per_bin_kwargs["min_conversion_rate"] = st.number_input(
                "per-bin min_conversion_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key="perbin_minconv_val"
            )
        if pb_min_comp:
            per_bin_kwargs["min_completion_rate"] = st.number_input(
                "per-bin min_completion_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key="perbin_mincomp_val"
            )
        if pb_min_take:
            per_bin_kwargs["min_take_rate"] = st.number_input(
                "per-bin min_take_rate",
                value=0.0, min_value=0.0, max_value=1.0,
                key="perbin_mintake_val"
            )
        if pb_max_take:
            per_bin_kwargs["max_take_rate"] = st.number_input(
                "per-bin max_take_rate",
                value=1.0, min_value=0.0, max_value=1.0,
                key="perbin_maxtake_val"
            )

        if per_bin_kwargs:
            # Apply same ConstraintBlock to every distance bin
            for bin_name in dm.distance_vals:
                per_bin_constraints[bin_name] = ConstraintBlock(**per_bin_kwargs)

    # ---- 4. Build constraints ----
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

            st.session_state["opt_df"] = df_opt
            st.session_state["opt_stats"] = stats

            st.success("Optimization completed!")

        except Exception as e:
            st.error(f"Solver error: {e}")

    # ----------------------------------------------------------
    # Restore previous results + DOWNLOAD buttons
    # ----------------------------------------------------------
    if st.session_state["opt_df"] is not None:
        st.write("### Optimal Strategy")
        st.dataframe(st.session_state["opt_df"])

        opt_csv = st.session_state["opt_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Optimal Strategy (CSV)",
            opt_csv,
            file_name="optimal_strategy.csv",
            mime="text/csv"
        )

    if st.session_state["opt_stats"] is not None:
        st.write("### Stats: Optimal vs Default")
        st.dataframe(st.session_state["opt_stats"])

        stats_csv = st.session_state["opt_stats"].to_csv().encode("utf-8")
        st.download_button(
            "Download Stats (CSV)",
            stats_csv,
            file_name="strategy_stats.csv",
            mime="text/csv"
        )



# ----------------------------------------------------------
# TAB 2: MANUAL SEARCH
# ----------------------------------------------------------

with tab_manual:

    st.write("### Manual Strategy Search")

    # Load previous manual table
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
                options=[float(i) for i in dm.client_diffs]
            ),
            "Courier diff": st.column_config.SelectboxColumn(
                "Courier diff",
                options=[float(v) for v in dm.courier_diffs]
            ),
        }
    )

    st.session_state["manual_inputs_df"] = edited_df

    if st.button("Evaluate Manual Strategy"):
        try:
            manual_selection = {
                row["Distance"]: (row["Client diff"], row["Courier diff"])
                for _, row in edited_df.iterrows()
            }

            default_strategy = {dist: (0.0, 0.0) for dist in dm.distance_vals}

            manual_stats = dm.calculate_strategy(manual_selection)
            default_stats = dm.calculate_strategy(default_strategy)

            # --- Create dataframe ---
            df_compare = pd.DataFrame({
                "Manual Strategy": manual_stats,
                "Default Strategy": default_stats
            })

            # --- Percentage Change ---
            pct = (df_compare["Manual Strategy"] - df_compare["Default Strategy"]) \
                  / df_compare["Default Strategy"] * 100

            pct = pct.replace([float("inf"), -float("inf")], None)

            def fmt(x):
                if x is None or pd.isna(x):
                    return None
                return f"{'+' if x > 0 else ''}{x:.2f}%"

            df_compare["% Change"] = pct.apply(fmt)

            # Save to session
            st.session_state["manual_selection"] = manual_selection
            st.session_state["manual_compare_df"] = df_compare

            st.success("Manual strategy evaluated.")

        except Exception as e:
            st.error(f"Error evaluating manual strategy: {e}")

    # Restore previous results + DOWNLOAD button
    if st.session_state["manual_compare_df"] is not None:
        st.write("### Manual vs Default")
        st.dataframe(st.session_state["manual_compare_df"])

        csv = st.session_state["manual_compare_df"].to_csv().encode("utf-8")
        st.download_button(
            "Download Manual Strategy Comparison (CSV)",
            csv,
            file_name="manual_vs_default.csv",
            mime="text/csv"
        )
