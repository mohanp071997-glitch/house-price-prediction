
import streamlit as st
import pandas as pd
import joblib

# Load the saved ML model
model = joblib.load("house_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details to estimate its price.")

# User inputs
area = st.number_input(
    "Area (sqft)", min_value=100, max_value=10000, value=1500
)

bedrooms = st.number_input(
    "Bedrooms", min_value=1, max_value=10, value=3
)

bathrooms = st.number_input(
    "Bathrooms", min_value=1, max_value=10, value=2
)

age = st.number_input(
    "House Age (years)", min_value=0, max_value=100, value=5
)

location = st.selectbox(
    "Location",
    ["Coimbatore", "Chennai", "Madurai", "Salem", "Trichy"]
)

parking = st.selectbox("Parking", ["Yes", "No"])

furnishing = st.selectbox(
    "Furnishing",
    ["Fully-Furnished", "Semi-Furnished", "Unfurnished"]
)

# Prediction button
if st.button("Predict Price"):
    new_house = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_years": age,
        "Location": location,
        "Parking": parking,
        "Furnishing": furnishing
    }])

    predicted_price = model.predict(new_house)[0]

    st.success(f"Estimated House Price: ₹{predicted_price:.2f} Lakhs")
