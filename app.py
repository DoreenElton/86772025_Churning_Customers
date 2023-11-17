# app.py
import streamlit as st
import pickle
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load the preprocessor and pre-trained model
loaded_model = joblib.load('tuned_mlp_model.pkl')

with open('label.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Create a Streamlit web app
st.title('Churn Prediction App')

# Add input fields for user to enter new data
st.sidebar.header('Enter Customer Data')
feature1 = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
feature2 = st.sidebar.slider('tenure', min_value=0, max_value=100, value=50)
feature3 = st.sidebar.selectbox('OnlineSecurity', ['No', 'Yes', 'No internet service'])
feature4 = st.sidebar.selectbox('TechSupport', ['No', 'Yes', 'No internet service'])
feature5 = st.sidebar.slider('TotalCharges', min_value=0.0, max_value=1000000.0, value=0.0)
feature6 = st.sidebar.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
feature7 = st.sidebar.slider('MonthlyCharges', min_value=0.0, max_value=1000.0, value=0.0)
feature8 = st.sidebar.selectbox('PaperlessBilling', ['Yes', 'No'])

# Create a DataFrame with the user input
if st.sidebar.button('Predict'):
    new_data = pd.DataFrame({
        'Contract': [feature1],
        'tenure': [feature2],
        'OnlineSecurity': [feature3],
        'TechSupport': [feature4],
        'TotalCharges': [feature5],
        'OnlineBackup': [feature6],
        'MonthlyCharges': [feature7],
        'PaperlessBilling': [feature8]
    })

    # Encode categorical features
    encoded_data = loaded_label_encoder.transform(new_data)

    # Scale the data
    scaled_data = loaded_scaler.transform(encoded_data)

    # Create a DataFrame with the scaled data and feature names
    column_features = ['Contract', 'tenure', 'OnlineSecurity', 'TechSupport', 'TotalCharges', 'OnlineBackup', 'MonthlyCharges', 'PaperlessBilling']
    df = pd.DataFrame(scaled_data, columns=column_features)

    # Make predictions using predict_proba
    prediction_proba = loaded_model.predict_proba(df)[:, 1]

    # Display the prediction result
    st.subheader('Prediction Result:')
    st.write('Likely to Churn' if prediction_proba[0] > 0.5 else 'Not Likely to Churn')
