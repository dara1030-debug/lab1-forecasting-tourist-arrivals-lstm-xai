import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.title("5. Train & tune the LSTM")

if "X_train_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()


# --------------------------------------------------
# Get prepared training data
# --------------------------------------------------

X = st.session_state.X_train_seq
y = st.session_state.y_train_seq

st.write(
    f"Training data shape: {X.shape}"
)

st.write(
    f"Target shape: {y.shape}"
)


# --------------------------------------------------
# Chronological train/validation split
# --------------------------------------------------

validation_ratio = 0.20

split_idx = int(
    len(X) * (1 - validation_ratio)
)

X_train = X[:split_idx]
y_train = y[:split_idx]

X_val = X[split_idx:]
y_val = y[split_idx:]


st.write(
    f"Training windows: {len(X_train)}"
)

st.write(
    f"Validation windows: {len(X_val)}"
)


# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------

def build_model(
    units=32,
    dropout=0.0,
    learning_rate=0.001
):

    model = keras.Sequential([
        keras.layers.Input(
            shape=(X.shape[1], X.shape[2])
        ),

        keras.layers.LSTM(
            units
        ),

        keras.layers.Dropout(
            dropout
        ),

        keras.layers.Dense(
            1
        )
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model


# --------------------------------------------------
# Hyperparameter options
# --------------------------------------------------

units_options = [16, 32, 64]
dropout_options = [0.0, 0.2]
learning_rate_options = [0.001]


if st.button("Run LSTM tuning"):

    results = []

    best_model = None
    best_val_loss = float("inf")
    best_config = None

    progress = st.progress(0)

    total_configs = (
        len(units_options)
        * len(dropout_options)
        * len(learning_rate_options)
    )

    completed = 0


    # --------------------------------------------------
    # Try each configuration
    # --------------------------------------------------

    for units in units_options:

        for dropout in dropout_options:

            for learning_rate in learning_rate_options:

                tf.keras.backend.clear_session()

                model = build_model(
                    units=units,
                    dropout=dropout,
                    learning_rate=learning_rate
                )

                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )

                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=16,
                    shuffle=False,
                    callbacks=[early_stopping],
                    verbose=0
                )

                best_epoch_loss = min(
                    history.history["val_loss"]
                )

                results.append({
                    "units": units,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "best_val_loss": best_epoch_loss,
                    "epochs_run": len(
                        history.history["loss"]
                    )
                })


                # Keep the best model
                if best_epoch_loss < best_val_loss:

                    best_val_loss = best_epoch_loss

                    best_model = model

                    best_config = {
                        "units": units,
                        "dropout": dropout,
                        "learning_rate": learning_rate
                    }


                completed += 1

                progress.progress(
                    completed / total_configs
                )


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    st.session_state.best_model = best_model

    st.session_state.train_results = results

    st.session_state.best_config = best_config

    st.session_state.best_val_loss = best_val_loss


# --------------------------------------------------
# Display saved results
# --------------------------------------------------

if "train_results" in st.session_state:

    st.subheader("Hyperparameter results")

    results_df = st.session_state.train_results

    st.dataframe(results_df)


    st.subheader("Best configuration")

    st.success(
        f"Units: "
        f"{st.session_state.best_config['units']} | "
        f"Dropout: "
        f"{st.session_state.best_config['dropout']} | "
        f"Learning rate: "
        f"{st.session_state.best_config['learning_rate']}"
    )

    st.write(
        f"Best validation MSE: "
        f"{st.session_state.best_val_loss:.6f}"
    )

    st.info(
        "The test set was not used during hyperparameter tuning."
    )

else:

    st.info(
        'Click "Run LSTM tuning" to train and compare '
        'the candidate LSTM configurations.'
    )