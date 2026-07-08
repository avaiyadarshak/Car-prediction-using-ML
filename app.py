import streamlit as st
import pandas as pd
import joblib

# Load Dataset
df = pd.read_csv("cars.csv")

# Load Model
model = joblib.load("car_price_model.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("Car Price Prediction")

# ---------------- Inputs ---------------- #

make = st.selectbox(
    "Car Brand",
    sorted(df["Make"].dropna().unique())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel_Type"].dropna().unique())
)

body = st.selectbox(
    "Body Type",
    sorted(df["Body_Type"].dropna().unique())
)

engine = st.number_input(
    "Engine Displacement (CC)",
    value=1200.0
)

power = st.number_input(
    "Power",
    value=80.0
)

seat = st.number_input(
    "Seating Capacity",
    value=5
)

# ---------------------------------------- #

if st.button("Predict Price"):

    # Template Row
    input_df = df.drop("Ex-Showroom_Price", axis=1).iloc[[0]].copy()

    # Update Important Columns
    input_df["Make"] = make
    input_df["Fuel_Type"] = fuel
    input_df["Body_Type"] = body
    input_df["Displacement"] = engine
    input_df["Power"] = power
    input_df["Seating_Capacity"] = seat

    prediction = model.predict(input_df)

    st.success(f"Predicted Price : ₹ {prediction[0]:,.2f}")