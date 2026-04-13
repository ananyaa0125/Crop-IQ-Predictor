# Crop IQ: Weather-Based Crop Price Prediction with IoT Integration

## Overview
Crop IQ is an end-to-end machine learning web application designed to predict daily price fluctuations for cash crops across major agricultural hubs in Punjab (Jalandhar, Ludhiana, Hoshiarpur, and Kapurthala). 

Instead of relying solely on historical price averages, this project utilizes a highly optimized **XGBoost Regressor** to model the complex, non-linear relationships between localized weather patterns (temperature, rainfall) and market supply dynamics (arrival shocks). 

## Live Demo
https://crop-iq-predictor.streamlit.app/

## Machine Learning Methodology
To prevent historical inflation overfitting, this model predicts **Day-over-Day Price Change** rather than absolute prices. The data pipeline incorporates advanced feature engineering, including:
* **Arrival Shocks:** Measuring sudden market surplus/deficits against a 7-day rolling average.
* **Volatility Index:** Tracking 14-day market price standard deviation to gauge economic panic.
* **Weather Lags:** Factoring in precipitation and temperature from 3-7 days prior, simulating the delay between field conditions and market impact.

The model was rigorously validated using **Time-Series Split (Forward Chaining)** to ensure zero data leakage from future dates. 

## Tech Stack
* **Frontend/Deployment:** Streamlit, Streamlit Community Cloud
* **Machine Learning:** XGBoost (`XGBRegressor`), Scikit-Learn
* **Data Engineering:** Python, Pandas, NumPy

## Data Architecture & Sources
This model relies on a fused dataset combining market economics with meteorological data:
* **Market Economics:** Local Mandi (market) records for daily crop arrivals, supply shocks, and pricing (Sourced via Agmarknet/Kaggle).
* **Meteorological Data:** Historical and lagged weather data including temperature ranges and precipitation (NASA POWER).
* **Live Ingestion:** Simulated IoT sensor feeds for real-time environmental parameters via the Streamlit UI.

## How to Run Locally
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit server: `streamlit run app.py`
