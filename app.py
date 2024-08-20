import streamlit as st
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('xgb_model.pkl', 'rb'))

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
    
    # Convert 'type' to categorical if needed
    # If the model expects encoded types, do the encoding here
    # For instance, if 'type' needs to be encoded as integers:
    input_df['type'] = input_df['type'].map({'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4})
    
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
    
    st.sidebar.subheader("Enter Type of Transfer Made:")
    type_id = st.sidebar.selectbox("", list(transaction_types.keys()))
    payment = transaction_types[type_id]

    # Define input fields
    amount = st.number_input('Amount', min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input('Old Balance (Origin)', min_value=0.0, format="%.2f")
    newbalanceOrg = st.number_input('New Balance (Origin)', min_value=0.0, format="%.2f")
    oldbalanceDest = st.number_input('Old Balance (Destination)', min_value=0.0, format="%.2f")
    newbalanceDest = st.number_input('New Balance (Destination)', min_value=0.0, format="%.2f")

    # Preprocess user input
    input_data = preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.write('The Person is Fraud')
    else:
        st.write('The Person is Not Fraud')

    # Display input data
    st.write('Input Data:')
    st.write(input_data)

if __name__ == '__main__':
    app()
