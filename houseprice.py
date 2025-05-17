import streamlit as st
import pandas as pd
import joblib

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# App title
st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the price of a house:")

# User inputs
area = st.number_input("Area (in square meters):", min_value=10, max_value=1000, value=100)
floor = st.number_input("Floor number:", min_value=0, max_value=50, value=2)
rooms = st.number_input("Number of rooms:", min_value=1, max_value=10, value=3)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, floor, rooms]], columns=["Area", "Floor", "Rooms"])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Price: ${prediction:,.2f}")
