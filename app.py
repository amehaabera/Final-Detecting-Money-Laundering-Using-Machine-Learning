import streamlit as st
import pickle
import pandas as pd
pip install --upgrade pip setuptools wheel
pip install --upgrade pip
pip install streamlit
pip install xgboost
streamlit run your_script_name.py
pip uninstall -r requirements.txt -y

# Load the model
try:
    model = pickle.load(open('xgb_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Make sure 'xgb_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
    # Adjust this part based on your model’s requirements
    type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    if payment in type_mapping:
        input_df['type'] = type_mapping[payment]
    else:
        st.error(f"Invalid transaction type: {payment}")
        st.stop()

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
    try:
        input_data = preprocess_input(payment, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest)
    except Exception as e:
        st.error(f"Error preprocessing input: {e}")
        st.stop()
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()
    
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
