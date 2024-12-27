import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the loaded model
st.write(f"Loaded model type: {type(model)}")  # This will help you debug if the model is loaded correctly

# Define feature names (make sure this matches the order used during training)
feature_names = ['Transaction Day', 'Transaction Hour', 'Sex', 'Age', 'Sub City', 'Occupation', 'Branch Name', 'Account Type', 'Transaction Type', 'Conducting Manner', 'Amount in Birr', 'Old Balance Org', 'New Balance Orig', 'Old Balance Held Dest', 'Balance Held Dest']

# Preprocess input data to match the feature transformations applied during training
def preprocess_data(data):
    input_df = data.copy()
    preprocessed_data = pd.DataFrame(columns=feature_names)
    
    # Label encode 'Conducting Manner' column
    le = LabelEncoder()
    le.fit(['Account to Account', 'Cash', 'DEBIT', 'Internet Banking', 'Mobile Banking'])  # Make sure this matches the training set
    
    preprocessed_data['Transaction Day'] = input_df['Transaction Day']
    preprocessed_data['Transaction Hour'] = input_df['Transaction Hour']
    preprocessed_data['Sex'] = input_df['Sex']
    preprocessed_data['Age'] = input_df['Age']
    preprocessed_data['Sub City'] = input_df['Sub City']
    preprocessed_data['Occupation'] = input_df['Occupation']
    preprocessed_data['Branch Name'] = input_df['Branch Name']
    preprocessed_data['Account Type'] = input_df['Account Type']
    preprocessed_data['Conducting Manner'] = input_df['Conducting Manner']
    preprocessed_data['Transaction Type'] = le.transform(input_df['Transaction Type'])  # Transform categorical 'Conducting Manner'
    preprocessed_data['Amount in Birr'] = np.log1p(input_df['Amount in Birr'])  # Log-transform 'Amount in Birr'
    preprocessed_data['Old Balance Org'] = input_df['Old Balance Org']
    preprocessed_data['New Balance Orig'] = input_df['New Balance Orig']
    preprocessed_data['Old Balance Held Dest'] = input_df['Old Balance Held Dest']
    preprocessed_data['Balance Held Dest'] = input_df['Balance Held Dest']
        
    return preprocessed_data

# Make predictions using the loaded model
def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    return predictions

# Create a multi-page layout using the sidebar
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "App", "About"])

    if selection == "Home":
        st.title("Anti-money laundering (AML) transaction monitoring software System")
        st.write("""
        This application uses machine learning models to predict the likelihood of a transaction being fraudulent.
        You can enter the transaction details and the system will predict if the transaction is **Normal** or **Fraudulent**.
        """)
        st.write("""
        ### Features:
        - Enter details of transactions such as payment type, amount, and balances.
        - Get real-time predictions about the likelihood of fraudulent transactions.
        """)

    elif selection == "App":
        st.title("**Money Laundering Transaction Detection System**")

        # Input form
        st.header('Enter transaction details:')
        Transaction_Day = st.number_input('Transaction Day', min_value=0, max_value=6)  # Add min and max values
        Transaction_Hour = st.number_input('Transaction Hour', min_value=0, max_value=23)  # Add min and max values  
        Sex = st.number_input('Sex', min_value=0, max_value=1)
        Age = st.number_input('Age', min_value=1)
        Sub_City = st.number_input('Sub City', min_value=0)
        Occupation = st.number_input('Occupation', min_value=0)
        Branch_Name = st.number_input('Branch Name', min_value=0)
        Account_Type = st.number_input('Account Type', min_value=0)
        Transaction_Type =  st.selectbox(label='Select payment type', options=['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        Conducting_Manner = st.number_input('Conducting Manner', min_value=0)
        Amount_in_Birr = st.number_input('Amount in Birr', min_value=0.0)
        Old_Balance_Org = st.number_input('Old Balance Org')
        New_Balance_Orig = st.number_input('New Balance Origin', value=Old_Balance_Org)
        Old_Balance_Held_Dest = st.number_input('Old Balance Destination')
        Balance_Held_Dest = st.number_input('Balance Held Dest', value=Old_Balance_Held_Dest)

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'Transaction Day': [Transaction_Day],
            'Transaction Hour': [Transaction_Hour],
            'Sex': [Sex],
            'Age': [Age],
            'Sub City': [Sub_City],
            'Occupation': [Occupation],
            'Branch Name': [Branch_Name],
            'Account Type': [Account_Type],
            'Transaction Type': [Transaction_Type],
            'Conducting Manner': [Conducting_Manner],
            'Amount in Birr': [Amount_in_Birr],
            'Old Balance Org': [Old_Balance_Org],
            'New Balance Orig': [New_Balance_Orig],
            'Old Balance Held Dest': [Old_Balance_Held_Dest],
            'Balance Held Dest': [Balance_Held_Dest]
        }, columns=feature_names)

        # Validate input data
        if input_data.isnull().values.any():
            st.error("Please enter valid numeric values for all fields.")
        else:
            # Predict button
            if st.button('Predict'):
                try:
                    predictions = predict(input_data)
                    is_fraud = 'Fraud' if predictions[0] == 1 else 'Normal'
                    
                    # Display the result in green if 'Normal', red if 'Fraud'
                    if is_fraud == 'Normal':
                        st.markdown(f"<h3 style='color:green; font-weight:bold;'>Predicted Class: {is_fraud}</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color:red; font-weight:bold;'>Predicted Class: {is_fraud}</h3>", unsafe_allow_html=True)
                
                except ValueError:
                    st.error("Please enter valid numeric values for all fields.")
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

    elif selection == "About":
        st.title("About the Application")
        st.write("""
        This application was developed to help detect potential money laundering transactions using machine learning.
        The model is trained on a dataset of historical transactions and uses various transaction features to make predictions.
        """)
        st.write("""
        ### Developer Contact:
        - **Developer**: Ameha Abera Kidane
        - **Email**: amehaabera@gmail.com
        - **Linkedin**: https://www.linkedin.com/in/ameha-abera-kidane/
        - **GitHub**: (https://github.com/amehaabera)
        """)

if __name__ == '__main__':
    main()

