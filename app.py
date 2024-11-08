# app.py

# Import necessary libraries
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import streamlit as st
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# ----- FASTAPI SECTION -----

# FastAPI application
app = FastAPI()

# Load the model and scaler with error handling
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

# Define the data structure for incoming prediction requests
class ModelData(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float
    # Add other fields if your model requires more features

@app.post("/predict/")
async def predict(data: ModelData) -> Dict[str, str]:
    """
    Endpoint for making predictions.
    Accepts JSON input with transaction details and returns a prediction score.
    """
    if model is None or scaler is None:
        logging.error("Model or scaler is not loaded.")
        return {"error": "Model or scaler not loaded."}

    logging.info(f"Received data: {data}")

    # Prepare data for the model
    try:
        input_data = np.array([[data.transaction_amount, data.customer_age, data.customer_balance]])
        logging.info(f"Input data before scaling: {input_data}")
        input_data = scaler.transform(input_data)
        logging.info(f"Input data after scaling: {input_data}")
    except Exception as e:
        logging.error(f"Error during data scaling: {e}")
        return {"error": f"Data scaling error: {e}"}

    # Get fraud probability
    try:
        probabilities = model.predict_proba(input_data)
        fraud_score = probabilities[0][1]  # Probability of fraud (class 1)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": f"Prediction error: {e}"}

    # Determine risk level and message
    if fraud_score < 0.4:
        risk = "Low"
        message = "This Transaction is not a fraud"
    elif 0.4 <= fraud_score <= 0.7:
        risk = "Medium"
        message = "This Transaction is not a fraud"
    else:
        risk = "High"
        message = "This Transaction is a fraud"

    # Return risk score, risk level, and message with fraud_score as a string
    result = {
        "fraud_score": f"{fraud_score:.4f}",  # Convert to string with 4 decimal places for clarity
        "risk": risk,
        "message": message
    }
    logging.info(f"Returning result: {result}")
    return result


# ----- STREAMLIT SECTION -----

# Define FastAPI URL for Streamlit to connect with FastAPI
api_url = "http://127.0.0.1:8000/predict/"

# Streamlit Application
st.title("Real-Time Loan Fraud Prediction")

# Get user inputs
transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

if st.button("Predict"):
    # Prepare data as JSON for FastAPI
    payload = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }
    # Call the FastAPI and get the prediction
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.write("Error:", result["error"])
            else:
                # Display the result message, risk level, and fraud score
                st.write(result["message"])
                st.write(f"Risk: {result['risk']}")
                st.write(f"Risk score: {result['fraud_score']}")
        else:
            st.write("Error: Unable to get a valid response from the prediction API.")
            st.write(f"Status code: {response.status_code}")
            st.write(f"Response text: {response.text}")
    except requests.exceptions.RequestException as e:
        st.write("Error: Unable to connect to the API.")
        st.write(f"Exception: {e}")