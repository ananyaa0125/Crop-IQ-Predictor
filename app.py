import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# --- Page Config ---
st.set_page_config(page_title="Crop IQ Dashboard", page_icon="🥔", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    try:
        model.load_model("CropIQ_XGBoost_Optimized.json")
        return model
    except Exception as e:
        st.error("Model file not found! Make sure 'CropIQ_XGBoost_Optimized.json' is in the folder.")
        return None

model = load_model()

# --- UI Header ---
st.title("🌱 Crop IQ: Potato Price Predictor")
st.markdown("Simulate live IoT weather data and market arrivals to predict tomorrow's price changes in Punjab.")
st.divider()

if model:
    # --- Sidebar Inputs (Simulating IoT Sensors) ---
    st.sidebar.header("📡 Live Sensor Data")
    
    todays_price = st.sidebar.number_input("Today's Price (₹/Quintal)", value=1000)
    price_lag1 = st.sidebar.number_input("Yesterday's Price", value=980)
    price_lag3 = st.sidebar.number_input("Price 3 Days Ago", value=950)
    
    st.sidebar.subheader("☁️ Weather Conditions")
    max_temp = st.sidebar.slider("Today's Max Temp (°C)", 10, 50, 32)
    min_temp = st.sidebar.slider("Today's Min Temp (°C)", 0, 35, 18)
    rain_lag3 = st.sidebar.slider("Rainfall 3 Days Ago (mm)", 0.0, 50.0, 0.0)
    temp_max_lag3 = st.sidebar.slider("Max Temp 3 Days Ago (°C)", 10, 50, 30)
    
    st.sidebar.subheader("🚜 Market Supply")
    todays_arrivals = st.sidebar.number_input("Today's Arrivals (Tonnes)", value=150)
    avg_arrivals_7d = st.sidebar.number_input("7-Day Average Arrivals", value=130)
    volatility_14d = st.sidebar.number_input("14-Day Price Volatility (Risk Index)", value=45.0)

    # --- Feature Engineering on the Fly ---
    temp_range = max_temp - min_temp
    arrival_shock = todays_arrivals - avg_arrivals_7d
    is_rainy = 1 if rain_lag3 > 0 else 0

    # THE FIX: Added the 4 missing columns and ordered them exactly as the model expects
    expected_columns = [
        'Arrivals (Tonnes)', 'Modal Price (Rs./Quintal)', 'weather_year', 'weather_mo', 
        'weather_da', 'weather_temp', 'weather_dewp', 'weather_visib', 'weather_wdsp', 
        'weather_mxpsd', 'weather_min', 'weather_fog', 'weather_rain_drizzle', 
        'weather_thunder', 'year', 'month', 'day_of_year', 'week_of_year', 
        'day_of_year_sin', 'day_of_year_cos', 'temp_range', 'is_rainy', 
        'Arrival_Shock', 'Price_Volatility_14D', 'weather_max_lag3', 'Price_Lag_1', 'Price_Lag_3'
    ]
    
    input_dict = {
        'Arrivals (Tonnes)': todays_arrivals,
        'Modal Price (Rs./Quintal)': todays_price,
        'weather_year': 2024,
        'weather_mo': 5,
        'weather_da': 25,
        'weather_temp': (max_temp + min_temp) / 2,
        'weather_dewp': min_temp - 2, 
        'weather_visib': 2.0, 
        'weather_wdsp': 1.0,  
        'weather_mxpsd': 2.0, 
        'weather_min': min_temp,
        'weather_fog': 0,
        'weather_rain_drizzle': is_rainy,
        'weather_thunder': 0,
        'year': 2024,
        'month': 5,
        'day_of_year': 150,
        'week_of_year': 22,
        'day_of_year_sin': 0.5,
        'day_of_year_cos': 0.8,
        'temp_range': temp_range,
        'is_rainy': is_rainy,
        'Arrival_Shock': arrival_shock,
        'Price_Volatility_14D': volatility_14d,
        'weather_max_lag3': temp_max_lag3,
        'Price_Lag_1': price_lag1,
        'Price_Lag_3': price_lag3
    }
    
    # Create dataframe with exact column order
    input_data = pd.DataFrame([input_dict])[expected_columns]

    # --- Prediction Execution ---
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing weather and market patterns..."):
            predicted_change = model.predict(input_data)[0]
            predicted_tomorrow = todays_price + predicted_change
            
            # --- Results Display ---
            st.subheader("📊 Market Forecast")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Price Tomorrow", f"₹{predicted_tomorrow:.2f}", f"₹{predicted_change:.2f} change")
            
            risk_level = "High" if volatility_14d > 100 else "Normal"
            col2.metric("Market Volatility", risk_level)
            
            action = "SELL NOW" if predicted_change < 0 else "HOLD"
            action_color = "red" if action == "SELL NOW" else "green"
            col3.markdown(f"**Recommendation:**\n<h3 style='color:{action_color};'>{action}</h3>", unsafe_allow_html=True)
            
            # Context
            st.info(f"**Why?** The model factored in an arrival shock of {arrival_shock} tonnes and a temperature range of {temp_range}°C to make this prediction.")
