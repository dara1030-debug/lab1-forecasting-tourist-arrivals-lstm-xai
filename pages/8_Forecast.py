import streamlit as st
import pandas as pd
import numpy as np


st.title("8. Forecast Tourist Arrivals")


# --------------------------------------------------
# Check that the model and scaler are available
# --------------------------------------------------

if "best_model" not in st.session_state:
    st.warning("Run the Train page first.")
    st.stop()

if "scaler_X" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

if "scaler_y" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

if "selected_features" not in st.session_state:
    st.warning("Run the Features page first.")
    st.stop()


# --------------------------------------------------
# Get the trained model and selected features
# --------------------------------------------------

model = st.session_state.best_model
scaler_X = st.session_state.scaler_X
scaler_y = st.session_state.scaler_y

features = st.session_state.selected_features

lookback = 12


st.write(
    "Enter the feature readings for the next month. "
    "The model uses the previous 12 months as context."
)

st.write(
    "Selected features:"
)

st.write(features)


# --------------------------------------------------
# Load the cleaned dataset
# --------------------------------------------------

df = st.session_state.clean_df.copy()

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Get the latest 12 months
# --------------------------------------------------

latest_12 = df.tail(lookback).copy()


st.subheader("Latest 12-month input window")

st.dataframe(
    latest_12[
        ["date"] + features
    ]
)


# --------------------------------------------------
# Input fields for the forecast month
# --------------------------------------------------

st.subheader("Next-month feature values")

st.write(
    "These values are initialized from the latest "
    "available month. You may edit them before forecasting."
)


input_values = {}


for feature in features:

    default_value = latest_12[feature].iloc[-1]

    # If the latest value is missing, use the
    # training median as a fallback.
    if pd.isna(default_value):

        default_value = (
            df[feature]
            .dropna()
            .median()
        )

    input_values[feature] = st.number_input(
        feature,
        value=float(default_value),
        format="%.4f"
    )


# --------------------------------------------------
# Forecast button
# --------------------------------------------------

if st.button("Predict tourist arrivals"):

    # ----------------------------------------------
    # Copy the latest 12-month window
    # ----------------------------------------------

    forecast_window = latest_12[
        features
    ].copy()


    # ----------------------------------------------
    # Replace the final month's feature values
    # with the user's new readings
    # ----------------------------------------------

    for feature in features:

        forecast_window.loc[
            forecast_window.index[-1],
            feature
        ] = input_values[feature]


    # ----------------------------------------------
    # Handle any remaining missing values
    # using dataset medians
    # ----------------------------------------------

    for feature in features:

        if forecast_window[feature].isna().any():

            median_value = (
                df[feature]
                .dropna()
                .median()
            )

            forecast_window[feature] = (
                forecast_window[feature]
                .fillna(median_value)
            )


    # ----------------------------------------------
    # Scale using the SAME scaler fitted during
    # the Prepare stage.
    # ----------------------------------------------

    X_scaled = scaler_X.transform(
        forecast_window[features]
    )


    # ----------------------------------------------
    # Reshape into LSTM input:
    #
    # (1 sample, 12 months, number of features)
    # ----------------------------------------------

    X_input = X_scaled.reshape(
        1,
        lookback,
        len(features)
    )


    # ----------------------------------------------
    # Generate prediction
    # ----------------------------------------------

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )


    # ----------------------------------------------
    # Convert prediction back to original
    # tourist-arrival scale
    # ----------------------------------------------

    prediction = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )[0, 0]


    # Prevent a negative displayed forecast.
    prediction = max(
        0,
        prediction
    )


    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    st.subheader(
        "Predicted tourist arrivals"
    )

    st.success(
        f"{prediction:,.0f} tourist arrivals"
    )


    st.session_state.forecast_prediction = prediction


    # ----------------------------------------------
    # Show the values actually used
    # ----------------------------------------------

    st.subheader(
        "Input used for prediction"
    )

    input_df = pd.DataFrame(
        [input_values]
    )

    st.dataframe(
        input_df
    )


    st.info(
        "Prediction generated using the trained LSTM, "
        "the 12-month lookback window, and the same "
        "training-fitted feature and target scalers."
    )


# --------------------------------------------------
# Display previously generated prediction
# --------------------------------------------------

elif "forecast_prediction" in st.session_state:

    st.subheader(
        "Previous forecast"
    )

    st.success(
        f"{st.session_state.forecast_prediction:,.0f} "
        "tourist arrivals"
    )