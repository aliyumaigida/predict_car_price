import streamlit as st
import pandas as pd
import joblib
import numpy as np
try:
    import joblib
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib


# Try loading the model
try:
    model = joblib.load("stacked_model_pipeline.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Used Car Price Prediction App 🚗💰")

# Input fields
make_year = st.number_input("Make Year", min_value=2001, max_value=2025, step=1)
mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, step=0.1)
engine_cc = st.number_input("Engine (cc)", min_value=500, max_value=5000)
owner_count = st.number_input("Number of Previous Owners", min_value=0, max_value=10)
accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=10)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "CNG", "Hybrid"])
brand = st.selectbox("Brand", [
    "Chevrolet", "Honda", "BMW", "Hyundai", "Nissan", "Tesla",
    "Toyota", "Kia", "Volkswagen", "Ford", "others"
])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
color = st.selectbox("Color", ["White", "Black", "Silver", "Red", "Blue", "Other"])
service_history = st.selectbox("Service History", ["Full", "Partial", "Unknown"])
insurance_valid = st.selectbox("Insurance Valid", ["Yes", "No"])

# Predict button
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([{
            'make_year': make_year,
            'mileage_kmpl': mileage_kmpl,
            'engine_cc': engine_cc,
            'fuel_type': fuel_type,
            'owner_count': owner_count,
            'brand': brand,
            'transmission': transmission,
            'color': color,
            'service_history': service_history,
            'accidents_reported': accidents_reported,
            'insurance_valid': insurance_valid
        }])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

