import streamlit as st

st.title("2. Clean the data")

if "raw_df" not in st.session_state:
    st.warning("Run the Dataset page first.")
    st.stop()

if st.button("Run cleaning"):
    df = st.session_state.raw_df.copy()

    # 1. Check and remove duplicate dates
    duplicates_removed = int(df.duplicated(subset="date").sum())
    df = df.drop_duplicates(subset="date", keep="first")

    # 2. Check missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]

    # 3. IQR outlier check on arrivals
    q1, q3 = df["arrivals"].quantile([0.25, 0.75])
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flagged = df[
        (df["arrivals"] < lower) |
        (df["arrivals"] > upper)
    ]

    # Store cleaned data for later pages
    st.session_state.clean_df = df

    st.session_state.clean_report = {
        "duplicates_removed": duplicates_removed,
        "missing": missing,
        "flagged": flagged[["date", "arrivals"]],
    }


# Display results
if "clean_report" in st.session_state:

    report = st.session_state.clean_report

    st.write(
        f"Duplicates removed: {report['duplicates_removed']}"
    )

    st.write("Missing values by column:")
    st.dataframe(report["missing"])

    st.write("Flagged outliers:")
    st.dataframe(report["flagged"])

else:
    st.info(
        'Click "Run cleaning" to process the dataset loaded on the Dataset page.'
    )

st.subheader("Rows with missing values")

if "clean_df" in st.session_state:
    missing_rows = st.session_state.clean_df[
        st.session_state.clean_df.isna().any(axis=1)
    ]

    st.dataframe(missing_rows)