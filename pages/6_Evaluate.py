import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.title("6. Evaluate the LSTM")


# --------------------------------------------------
# Check required pipeline stages
# --------------------------------------------------

if "best_model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

if "X_test_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

if "scaler_y" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()


# --------------------------------------------------
# Helper function for MAPE
# --------------------------------------------------

def calculate_mape(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid division by zero
    mask = y_true != 0

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

if st.button("Run evaluation"):

    model = st.session_state.best_model

    scaler_y = st.session_state.scaler_y

    X_test = st.session_state.X_test_seq
    y_test_scaled = st.session_state.y_test_seq


    # --------------------------------------------------
    # LSTM predictions
    # --------------------------------------------------

    y_pred_scaled = model.predict(
        X_test,
        verbose=0
    )


    # Convert predictions back to original arrivals scale
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_true = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).ravel()


    # --------------------------------------------------
    # LSTM metrics
    # --------------------------------------------------

    lstm_mae = mean_absolute_error(
        y_true,
        y_pred
    )

    lstm_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    lstm_mape = calculate_mape(
        y_true,
        y_pred
    )

    lstm_r2 = r2_score(
        y_true,
        y_pred
    )


    # --------------------------------------------------
    # Naive baseline
    #
    # Predict the previous month's arrivals.
    # --------------------------------------------------

    test_df = st.session_state.test_df.copy()

    test_arrivals = test_df["arrivals"].values

    naive_true = []
    naive_pred = []

    for i in range(1, len(test_arrivals)):

        if (
            not np.isnan(test_arrivals[i])
            and not np.isnan(test_arrivals[i - 1])
        ):

            naive_true.append(
                test_arrivals[i]
            )

            naive_pred.append(
                test_arrivals[i - 1]
            )


    naive_true = np.array(naive_true)
    naive_pred = np.array(naive_pred)


    naive_mae = mean_absolute_error(
        naive_true,
        naive_pred
    )

    naive_rmse = np.sqrt(
        mean_squared_error(
            naive_true,
            naive_pred
        )
    )

    naive_mape = calculate_mape(
        naive_true,
        naive_pred
    )

    naive_r2 = r2_score(
        naive_true,
        naive_pred
    )


    # --------------------------------------------------
    # Seasonal-naive baseline
    #
    # Predict using the same month from
    # the previous year (12 months earlier).
    # --------------------------------------------------

    full_df = st.session_state.clean_df.copy()

    seasonal_true = []
    seasonal_pred = []

    for i in range(12, len(full_df)):

        current = full_df.iloc[i]["arrivals"]
        previous_year = full_df.iloc[i - 12]["arrivals"]

        # Only evaluate rows belonging to the test period
        if (
            full_df.iloc[i]["date"]
            >= test_df["date"].min()
            and not pd.isna(current)
            and not pd.isna(previous_year)
        ):

            seasonal_true.append(current)
            seasonal_pred.append(previous_year)


    seasonal_true = np.array(seasonal_true)
    seasonal_pred = np.array(seasonal_pred)


    seasonal_mae = mean_absolute_error(
        seasonal_true,
        seasonal_pred
    )

    seasonal_rmse = np.sqrt(
        mean_squared_error(
            seasonal_true,
            seasonal_pred
        )
    )

    seasonal_mape = calculate_mape(
        seasonal_true,
        seasonal_pred
    )

    seasonal_r2 = r2_score(
        seasonal_true,
        seasonal_pred
    )


    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results = pd.DataFrame({

        "Model": [
            "LSTM",
            "Naive",
            "Seasonal Naive"
        ],

        "MAE": [
            lstm_mae,
            naive_mae,
            seasonal_mae
        ],

        "RMSE": [
            lstm_rmse,
            naive_rmse,
            seasonal_rmse
        ],

        "MAPE (%)": [
            lstm_mape,
            naive_mape,
            seasonal_mape
        ],

        "R²": [
            lstm_r2,
            naive_r2,
            seasonal_r2
        ]
    })


    st.session_state.evaluation_results = results

    st.session_state.y_true = y_true

    st.session_state.y_pred = y_pred


# --------------------------------------------------
# Display saved results
# --------------------------------------------------

if "evaluation_results" in st.session_state:

    st.subheader("Test-set evaluation")

    st.dataframe(
        st.session_state.evaluation_results
    )

    st.info(
        "All metrics above were calculated on the "
        "held-out test period."
    )

else:

    st.info(
        'Click "Run evaluation" to evaluate the LSTM '
        'against the baseline models.'
    )