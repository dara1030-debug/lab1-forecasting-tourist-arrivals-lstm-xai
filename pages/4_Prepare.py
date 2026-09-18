import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("4. Prevent leakage and prepare sequences")

if "selected_features" not in st.session_state:
    st.warning("Run the Features page first.")
    st.stop()

LOOKBACK = 12

train_ratio = st.slider(
    "Train split ratio",
    min_value=0.60,
    max_value=0.95,
    value=0.80,
    step=0.05
)

st.caption(
    f"{train_ratio:.0%} train / {1 - train_ratio:.0%} test, "
    "split chronologically — no shuffling. "
    "Earliest rows train the model; the most recent rows test it."
)


def make_sequences(X, y, lookback):
    Xs = []
    ys = []

    for i in range(len(X) - lookback):
        target = y[i + lookback]

        # Do not create a training/evaluation sequence
        # when the target value is missing.
        if np.isnan(target).any():
            continue

        Xs.append(X[i:i + lookback])
        ys.append(target)

    return np.array(Xs), np.array(ys)


if st.button("Run"):

    df = st.session_state.clean_df.copy()

    # -----------------------------------------
    # 1. Chronological train/test split
    # -----------------------------------------

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # -----------------------------------------
    # 2. Selected features
    # -----------------------------------------

    cols = st.session_state.selected_features

    # -----------------------------------------
    # 3. Handle missing predictor values
    #
    # IMPORTANT:
    # Median is calculated from TRAINING data
    # only, preventing test-data leakage.
    # -----------------------------------------

    train_medians = train_df[cols].median()

    train_df[cols] = train_df[cols].fillna(train_medians)
    test_df[cols] = test_df[cols].fillna(train_medians)

    # -----------------------------------------
    # 4. Fit scalers ONLY on training data
    # -----------------------------------------

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train_df[cols])
    scaler_y.fit(train_df[["arrivals"]].dropna())

    st.session_state.scaler_X = scaler_X
    st.session_state.scaler_y = scaler_y

    # -----------------------------------------
    # 5. Transform data
    # -----------------------------------------

    train_X = scaler_X.transform(train_df[cols])
    test_X = scaler_X.transform(test_df[cols])

    train_y = scaler_y.transform(
        train_df[["arrivals"]]
    )

    test_y = scaler_y.transform(
        test_df[["arrivals"]]
    )

    # -----------------------------------------
    # 6. Create LSTM sequences
    # -----------------------------------------

    X_train_seq, y_train_seq = make_sequences(
        train_X,
        train_y,
        LOOKBACK
    )

    X_test_seq, y_test_seq = make_sequences(
        test_X,
        test_y,
        LOOKBACK
    )

    st.session_state.X_train_seq = X_train_seq
    st.session_state.y_train_seq = y_train_seq

    st.session_state.X_test_seq = X_test_seq
    st.session_state.y_test_seq = y_test_seq

    st.session_state.test_df = test_df

    # -----------------------------------------
    # 7. Save preparation report
    # -----------------------------------------

    st.session_state.prepare_report = {

        "train_ratio": train_ratio,

        "train_rows": len(train_df),

        "test_rows": len(test_df),

        "train_range": (
            train_df["date"].min().date(),
            train_df["date"].max().date()
        ),

        "test_range": (
            test_df["date"].min().date(),
            test_df["date"].max().date()
        ),

        "train_windows": len(X_train_seq),

        "test_windows": len(X_test_seq),

        "lookback": LOOKBACK,

        "missing_train_targets": int(
            train_df["arrivals"].isna().sum()
        ),

        "missing_test_targets": int(
            test_df["arrivals"].isna().sum()
        )
    }


# -----------------------------------------
# Display saved results
# -----------------------------------------

if "prepare_report" in st.session_state:

    r = st.session_state.prepare_report

    st.write(
        f"Split used: "
        f"{r['train_ratio']:.0%} train / "
        f"{1 - r['train_ratio']:.0%} test"
    )

    st.write(
        f"Train: {r['train_rows']} rows "
        f"({r['train_range'][0]} – {r['train_range'][1]})"
    )

    st.write(
        f"Test: {r['test_rows']} rows "
        f"({r['test_range'][0]} – {r['test_range'][1]})"
    )

    st.write(
        f"Missing training targets excluded from sequences: "
        f"{r['missing_train_targets']}"
    )

    st.write(
        f"Missing test targets excluded from sequences: "
        f"{r['missing_test_targets']}"
    )

    st.success(
        f"{r['train_windows']} training windows, "
        f"{r['test_windows']} test windows, "
        f"lookback = {r['lookback']}"
    )

else:

    st.info(
        'Choose a split ratio above, then click "Run" '
        'to prepare the sequences.'
    )