import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


st.title("7. Explain with SHAP")


# --------------------------------------------------
# Check required pipeline stages
# --------------------------------------------------

if "best_model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

if "X_train_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

if "X_test_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

if "selected_features" not in st.session_state:
    st.warning("Run the Features page first.")
    st.stop()


# --------------------------------------------------
# Get model and data
# --------------------------------------------------

model = st.session_state.best_model

X_train = st.session_state.X_train_seq
X_test = st.session_state.X_test_seq

feature_names = st.session_state.selected_features

n_features = len(feature_names)
lookback = X_train.shape[1]


st.write(
    f"LSTM input shape: {X_train.shape}"
)

st.write(
    f"Features being explained: {feature_names}"
)


# --------------------------------------------------
# Flatten LSTM input
#
# Original:
# (samples, 12 months, 8 features)
#
# SHAP representation:
# (samples, 12 × 8)
# --------------------------------------------------

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


# --------------------------------------------------
# Prediction function for SHAP
# --------------------------------------------------

def predict_from_flat(X_flat):

    X_3d = X_flat.reshape(
        X_flat.shape[0],
        lookback,
        n_features
    )

    predictions = model.predict(
        X_3d,
        verbose=0
    )

    return predictions.reshape(-1)


# --------------------------------------------------
# Run SHAP
# --------------------------------------------------

if st.button("Run SHAP explanation"):

    with st.spinner(
        "Calculating SHAP values. This may take a little while..."
    ):

        # Use a small background set to keep SHAP manageable
        background_size = min(
            20,
            len(X_train_flat)
        )

        background = X_train_flat[
            :background_size
        ]

        # Explain a small number of test windows
        explain_size = min(
            10,
            len(X_test_flat)
        )

        explain_data = X_test_flat[
            :explain_size
        ]

        # Kernel SHAP works with our flattened representation
        explainer = shap.KernelExplainer(
            predict_from_flat,
            background
        )

        shap_values = explainer.shap_values(
            explain_data,
            nsamples=100
        )

        shap_values = np.asarray(
            shap_values
        )

        # Some SHAP versions return an extra dimension
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        # Base value
        expected_value = explainer.expected_value

        if np.ndim(expected_value) > 0:
            expected_value = expected_value[0]

        # --------------------------------------------------
        # Reshape SHAP values back to:
        #
        # samples × months × features
        # --------------------------------------------------

        shap_3d = shap_values.reshape(
            explain_size,
            lookback,
            n_features
        )

        # --------------------------------------------------
        # Aggregate SHAP values across the 12-month window
        # --------------------------------------------------

        aggregated_shap = shap_3d.sum(axis=1)

        # Aggregate absolute SHAP values for global importance
        global_importance = np.abs(
            shap_3d
        ).mean(axis=1).mean(axis=0)

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        st.session_state.shap_values = shap_3d

        st.session_state.aggregated_shap = aggregated_shap

        st.session_state.global_importance = global_importance

        st.session_state.shap_expected_value = expected_value

        st.session_state.shap_explain_data = explain_data

        st.session_state.shap_explain_size = explain_size


# --------------------------------------------------
# Display results
# --------------------------------------------------

if "shap_values" in st.session_state:

    shap_3d = st.session_state.shap_values

    aggregated_shap = st.session_state.aggregated_shap

    global_importance = (
        st.session_state.global_importance
    )

    expected_value = (
        st.session_state.shap_expected_value
    )


    # ==================================================
    # 1. WATERFALL PLOT
    # ==================================================

    st.subheader("1. SHAP waterfall — one forecast")

    sample_index = st.number_input(
        "Test forecast index",
        min_value=0,
        max_value=(
            st.session_state.shap_explain_size - 1
        ),
        value=0,
        step=1
    )

    sample_shap = aggregated_shap[
        sample_index
    ]

    sample_values = X_test[
        sample_index
    ].mean(axis=0)


    # Create SHAP Explanation
    explanation = shap.Explanation(
        values=sample_shap,
        base_values=expected_value,
        data=sample_values,
        feature_names=feature_names
    )


    fig1 = plt.figure(
        figsize=(10, 6)
    )

    shap.plots.waterfall(
        explanation,
        max_display=n_features,
        show=False
    )

    st.pyplot(
        fig1,
        clear_figure=True
    )


    # ==================================================
    # 2. GLOBAL FEATURE IMPORTANCE
    # ==================================================

    st.subheader(
        "2. Global feature importance"
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_importance
    })

    importance_df = importance_df.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


    st.dataframe(
        importance_df
    )


    fig2 = plt.figure(
        figsize=(9, 5)
    )

    plt.barh(
        importance_df["feature"],
        importance_df["mean_abs_shap"]
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    st.pyplot(
        fig2,
        clear_figure=True
    )


    # ==================================================
    # 3. DEPENDENCE PLOT
    # ==================================================

    st.subheader(
        "3. SHAP dependence plot"
    )

    top_feature = importance_df.iloc[0]["feature"]

    top_index = feature_names.index(
        top_feature
    )


    # Mean feature value across the 12-month window
    feature_values = X_test[
        :st.session_state.shap_explain_size,
        :,
        top_index
    ].mean(axis=1)


    # SHAP contribution for that feature
    feature_shap_values = aggregated_shap[
        :,
        top_index
    ]


    fig3 = plt.figure(
        figsize=(9, 5)
    )

    plt.scatter(
        feature_values,
        feature_shap_values
    )

    plt.xlabel(
        f"{top_feature} (scaled mean)"
    )

    plt.ylabel(
        "SHAP value"
    )

    plt.title(
        f"SHAP Dependence Plot — {top_feature}"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.tight_layout()

    st.pyplot(
        fig3,
        clear_figure=True
    )


    st.info(
        f"Top predictor by mean absolute SHAP value: "
        f"{top_feature}"
    )


else:

    st.info(
        'Click "Run SHAP explanation" to generate '
        'the waterfall, global importance, and dependence plots.'
    )