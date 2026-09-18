import streamlit as st
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("3. Feature selection")

# Make sure the Clean page has been run first
if "clean_df" not in st.session_state:
    st.warning("Run the Clean page first.")
    st.stop()

# Candidate predictor variables
CANDIDATES = [
    "quarter",
    "is_holiday_peak",
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "rainfall_mm",
    "rainy_days",
    "humidity_pct",
    "typhoon_count",
    "typhoon_max_wind_kt",
    "storm_signal_days",
    "pm25_ugm3",
    "wave_height_m",
]

if st.button("Run Spearman + VIF"):

    # Get the cleaned dataset
    df = st.session_state.clean_df

    # --------------------------------------------------
    # STEP 1: Spearman correlation filter
    # --------------------------------------------------

    results = []

    for col in CANDIDATES:

        rho, p_value = spearmanr(
            df[col],
            df["arrivals"],
            nan_policy="omit"
        )

        results.append({
            "feature": col,
            "rho": rho,
            "p_value": p_value
        })

    # Keep features that satisfy BOTH conditions
    kept = [
        r["feature"]
        for r in results
        if abs(r["rho"]) > 0.10
        and r["p_value"] < 0.05
    ]

    # Display Spearman results
    st.subheader("Spearman correlation results")
    st.dataframe(results)

    st.write("Features passing Spearman filter:")
    st.write(kept)

    # Stop if no features passed
    if not kept:
        st.error(
            "No features passed the Spearman filter. "
            "Check the correlation results above."
        )
        st.stop()

    # --------------------------------------------------
    # STEP 2: Iterative VIF
    # --------------------------------------------------

    X = df[kept].dropna()

    vif_log = []

    while X.shape[1] > 1:

        vifs = [
            variance_inflation_factor(
                X.values,
                i
            )
            for i in range(X.shape[1])
        ]

        max_vif = max(vifs)

        # Stop when all VIF values are below 5
        if max_vif < 5:
            break

        # Find feature with highest VIF
        drop_col = X.columns[vifs.index(max_vif)]

        vif_log.append({
            "dropped": drop_col,
            "vif": max_vif
        })

        X = X.drop(columns=[drop_col])

    # Final selected features
    selected_features = list(X.columns)

    # Store results for later pages
    st.session_state.selected_features = selected_features

    st.session_state.feature_report = {
        "results": results,
        "vif_log": vif_log
    }

    # --------------------------------------------------
    # DISPLAY FINAL RESULTS
    # --------------------------------------------------

    st.subheader("VIF removals")

    if vif_log:
        st.dataframe(vif_log)
    else:
        st.write("No features were removed by the VIF procedure.")

    st.success(
        f"Selected features: {selected_features}"
    )


# Display previous results when returning to the page
if "feature_report" in st.session_state:

    report = st.session_state.feature_report

    st.subheader("Saved feature selection results")

    st.dataframe(report["results"])

    st.write("VIF removals:")

    if report["vif_log"]:
        st.dataframe(report["vif_log"])
    else:
        st.write("No features were removed by VIF.")

    st.success(
        f"Selected features: "
        f"{st.session_state.selected_features}"
    )

else:

    st.info(
        'Click "Run Spearman + VIF" to select features '
        'from the cleaned dataset.'
    )