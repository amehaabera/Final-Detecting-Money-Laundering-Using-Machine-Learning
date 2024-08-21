import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier  # Ensure the version matches your requirement

# Load the model
model_path = 'xgb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define function to preprocess user input
def preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest):
    # Create dataframe with user input
    input_df = pd.DataFrame({
        'type': [payment],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrg': [newbalanceOrg],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    # Return preprocessed input
    return input_df

# Define Streamlit app
def app():
    # Set app title
    st.title('Money Laundering Detector')

    # Add sidebar to select transaction type
    transaction_types = {
        0: 'CASH_IN',
        1: 'CASH_OUT',
        2: 'DEBIT',
        3: 'PAYMENT',
        4: 'TRANSFER'
    }
    selected_type = st.sidebar.selectbox(
        "Select Type of Transfer:",
        options=list(transaction_types.keys()),
        format_func=lambda x: transaction_types[x]
    )

    # Define input fields
    payment = transaction_types[selected_type]
    amount = st.number_input('Amount', min_value=0.0)
    oldbalanceOrg = st.number_input('Old Balance (Origin)', min_value=0.0)
    newbalanceOrg = st.number_input('New Balance (Origin)', min_value=0.0)
    oldbalanceDest = st.number_input('Old Balance (Destination)', min_value=0.0)
    newbalanceDest = st.number_input('New Balance (Destination)', min_value=0.0)

    # Preprocess user input
    input_data = preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest)

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 0:
            st.write('The Person is Fraud')
        else:
            st.write('The Person is Not Fraud')

        # Display input data
        st.write('Input Data:')
        st.write(input_data)

if __name__ == '__main__':
    app()

