# Philippine Tourist Arrivals Forecasting App

A multipage Streamlit application for exploring, cleaning, modeling, explaining, and forecasting Philippine tourist arrivals.

The app uses an LSTM neural network for time-series forecasting and SHAP for Explainable AI (XAI). It walks through the full machine learning workflow: dataset inspection, cleaning, feature selection, data preparation, model training, evaluation, explanation, and forecasting.

## Project Purpose

This project was developed for **Laboratory Exercise 1: Forecasting Philippine Tourist Arrivals with LSTM and XAI**.

It uses historical Philippine tourist arrival data together with selected tourism, weather, and environmental variables to predict future tourist arrivals. The project also compares the LSTM model against simple forecasting baselines to show whether the machine learning model performs better than simpler approaches.

## Main Features

- Explore the tourist arrivals dataset
- Check duplicates, missing values, and possible outliers
- Select useful input features using correlation, statistical testing, and VIF
- Prepare monthly time-series windows for LSTM training
- Train and evaluate an LSTM model
- Compare the LSTM model with Naive and Seasonal Naive baselines
- Explain predictions using SHAP
- Generate a tourist-arrival forecast from the latest available data

## Technologies Used

- Python 3.11
- Streamlit
- Pandas
- NumPy
- SciPy
- Statsmodels
- Scikit-learn
- TensorFlow / Keras
- SHAP
- Matplotlib

## Project Structure

```text
tourist-arrivals-app/
|-- data/
|   `-- tourist_arrivals.csv
|-- pages/
|   |-- 1_Dataset.py
|   |-- 2_Clean.py
|   |-- 3_Features.py
|   |-- 4_Prepare.py
|   |-- 5_Train.py
|   |-- 6_Evaluate.py
|   |-- 7_Explain.py
|   `-- 8_Forecast.py
|-- Home.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/dara1030-debug/lab1-forecasting-tourist-arrivals-lstm-xai.git
cd lab1-forecasting-tourist-arrivals-lstm-xai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run Home.py
```

Streamlit will open the application in your web browser.

## App Workflow

The app is organized into eight pages. Use the sidebar to move through each step.

### 1. Dataset

Displays the raw dataset and basic information, including:

- Dataset dimensions
- Column names and data types
- Summary statistics
- Sample records
- Date range

### 2. Clean

Checks the dataset for data quality issues:

- Duplicate rows
- Missing values
- Possible outliers using the IQR method

For this dataset, the app identifies:

- 2 duplicate rows
- Missing values in `temp_min_c`, `temp_max_c`, and `arrivals`
- 3 possible outliers

Outliers are flagged for inspection instead of being automatically removed.

### 3. Features

Selects useful input features using:

- Spearman correlation
- Statistical significance testing
- Variance Inflation Factor (VIF)

The selected features are:

- `quarter`
- `is_holiday_peak`
- `rainy_days`
- `humidity_pct`
- `typhoon_max_wind_kt`
- `storm_signal_days`
- `pm25_ugm3`
- `wave_height_m`

The variables `rainfall_mm` and `typhoon_count` were removed because of high multicollinearity.

### 4. Prepare

Prepares the data for LSTM forecasting:

- Chronological train-test split
- 80% training data and 20% testing data
- 12-month lookback window
- Scaling fitted on training data only

Resulting sequence setup:

| Item | Value |
| --- | ---: |
| Training period | 2000-01-01 to 2020-09-01 |
| Testing period | 2020-10-01 to 2025-12-01 |
| Training windows | 234 |
| Testing windows | 51 |
| Lookback | 12 months |
| Features | 8 |

Three training windows with missing target values were excluded.

### 5. Train

Trains and compares six LSTM configurations using different:

- LSTM unit counts
- Dropout rates
- Learning rates

Selected model configuration:

| Hyperparameter | Value |
| --- | ---: |
| LSTM units | 32 |
| Dropout | 0.2 |
| Learning rate | 0.001 |
| Best validation MSE | 0.000112 |

The test set was not used during hyperparameter tuning.

### 6. Evaluate

Compares the LSTM model with Naive and Seasonal Naive baselines on the held-out test period.

| Model | MAE | RMSE | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 312,154.76 | 441,374.63 | 208.11% | -4.4939 |
| Naive | 53,244.24 | 76,122.48 | 16.11% | 0.8877 |
| Seasonal Naive | 174,981.24 | 288,362.86 | 377.57% | -0.5863 |

The Naive baseline performed better than the trained LSTM model on the held-out test period. This comparison is included to give a more honest view of the model's performance.

### 7. Explain

Uses SHAP to explain how the selected input features affect the LSTM model's predictions.

The LSTM input shape is:

```text
(234, 12, 8)
```

Global mean absolute SHAP values:

| Feature | Mean Absolute SHAP |
| --- | ---: |
| `humidity_pct` | 0.0007 |
| `is_holiday_peak` | 0.0004 |
| `rainy_days` | 0.0002 |
| `typhoon_max_wind_kt` | 0.0002 |
| `wave_height_m` | 0.0002 |
| `quarter` | 0.0001 |
| `storm_signal_days` | 0.0001 |
| `pm25_ugm3` | 0.00005 |

The SHAP results show that `humidity_pct` had the largest average contribution to the model's predictions. These explanations describe model behavior only and should not be treated as proof of causation.

### 8. Forecast

Generates a tourist-arrival forecast using:

- The trained LSTM model
- Training-fitted feature and target scalers
- The latest 12 months of available data
- User-adjustable latest feature values

The app scales the input, reshapes it into the LSTM format, generates a prediction, and converts the result back to the original tourist-arrival scale.

Example forecast:

```text
561,580 tourist arrivals
```

## Data Leakage Prevention

The project reduces data leakage risk by:

- Splitting the dataset chronologically
- Keeping the test period separate from training
- Fitting scalers on training data only
- Avoiding test data during hyperparameter tuning
- Reusing the training-fitted scalers during forecasting

## Usage

After running:

```bash
streamlit run Home.py
```

Navigate through the pages in this order:

```text
Home
Dataset
Clean
Features
Prepare
Train
Evaluate
Explain
Forecast
```

Each page represents one stage of the machine learning workflow.

## Limitations

- The LSTM model did not outperform the Naive baseline on the test period.
- The LSTM model has a high MAPE, which means it has substantial relative prediction error.
- SHAP analysis uses a limited number of test windows.
- The available data and selected features may not capture all factors that affect tourist arrivals.
- Forecast quality depends on the quality and representativeness of the input feature values.

## Conclusion

This project demonstrates an end-to-end workflow for Philippine tourist arrival forecasting with LSTM and Explainable AI.

The Streamlit app combines dataset exploration, cleaning, feature selection, time-series preparation, model training, evaluation, SHAP explanation, and forecasting in one interactive system. The results also show why machine learning models should be compared with simple baseline methods before drawing conclusions.

## License

This project was developed for academic and educational purposes.
