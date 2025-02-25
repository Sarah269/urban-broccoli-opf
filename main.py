
# Online Payment Fraud Detection Machine Learning App

# Load Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier

# Write Title for Streamlit App
st.write("""
### Online Payment Fraud Detection Machine Learning App 

This app predicts the probability that a transaction is fraudulent. 

* **Data Source**: Kaggle
* **Dataset**: [Online Payment Fraud Detection](https://www.kaggle.com/code/ananthu19/online-payments-fraud-detection/input)
* **Dataset Details**: 10 features. 6,363,620 rows. Imbalanced
* **Classification Model**:  DecisionTreeClassifier
* **Model Preparation**:  Data Resampling, Hyperparameter Tuning
* **Limitation**: The dataset did not have any fraud transactions for Cash In, Debit, and Payment. 

          
""")

# Metrics
st.write(""" 
         
**Online Payment Fraud Dataset Metrics** 
         
         """)

col1,col2 = st.columns((6,4))

with col1:
   
   tile = st.container(height=120)
   tile.metric(label=" :moneybag: Fraud Transactions ($)", value = "$12,056,415,427") 

with col2:
   tile = st.container(height=120)
   tile.metric(label=" :moneybag: % Total Transactions", value = "1.05%")

col3, col4 = st.columns((6,4))

with col3:
   tile = st.container(height=120)
   tile.metric(label=" :moneybag: Avg. Fraud Transaction ($)", value = "$1,467,967")

with col4:
   tile = st.container(height=120)
   tile.metric(label=" :moneybag: Number of Fraud Transactions", value = "8,213")

# Data Visualization

st.write("Transaction Amounts By Transaction Type")

st.image("TransDollars.png")

st.write("Number of Transactions By Type")

st.image("NumberTrans.png")


# Write Streamlit Sidebar Title
st.sidebar.header('User Input Parameters')

# Define User Input Feature Function
def user_input_features():

    amount = st.sidebar.number_input("Transaction Amount ($)", min_value = 0, value=0)

    oldbalanceOrg = st.sidebar.number_input("Old Balance ($)", value = 0)

    newbalanceOrig = st.sidebar.number_input("New Balance ($)", value = 0)

    option = st.sidebar.radio("Transaction Type", options=["Cash In", "Cash Out", "Debit", "Payment", "Transfer"])

    # Set flags for transaction types
    match option:
        case 'Cash Out':
            type_CASH_OUT = 1
            type_CASH_IN  = 0
            type_DEBIT    = 0
            type_PAYMENT  = 0
            type_TRANSFER = 0

        case 'Cash In':
            type_CASH_OUT = 0
            type_CASH_IN  = 1
            type_DEBIT    = 0
            type_PAYMENT  = 0
            type_TRANSFER = 0
        
        case 'Debit':
            type_CASH_OUT = 0
            type_CASH_IN  = 0
            type_DEBIT    = 1
            type_PAYMENT  = 0
            type_TRANSFER = 0

        case 'Payment':
            type_CASH_OUT = 0
            type_CASH_IN  = 0
            type_DEBIT    = 0
            type_PAYMENT  = 1
            type_TRANSFER = 0

        case 'Transfer':
            type_CASH_OUT = 0
            type_CASH_IN  = 0
            type_DEBIT    = 0
            type_PAYMENT  = 0
            type_TRANSFER = 1

    data = {'amount': amount,
        'oldbalanceOrg':oldbalanceOrg,
        'newbalanceOrig':newbalanceOrig,
        'type_CASH_IN':type_CASH_IN,
        'type_CASH_OUT':type_CASH_OUT,
        'type_DEBIT':type_DEBIT,
        'type_PAYMENT':type_PAYMENT,
        'type_TRANSFER':type_TRANSFER
        }

    features = pd.DataFrame(data, index=[0])
    return features

# Capture user selected features from sidebar
df_input = user_input_features()

st.write("___")  
st.write(""" 
         
         **User Parameters for Fraud Detection**
         
          """)


st.dataframe(df_input)
st.write("___")  

# Prediction
if st.sidebar.button("Predict"):
    # Load Model
    # Cache resource so that is loads once
    @st.cache_resource
    def load_model():
        model = joblib.load('onlinepymt_model_jl.sav.bz2')
        return model

    load_joblib_model = load_model()

    # Predict with model
    prediction = load_joblib_model.predict(df_input)
    predict_proba = load_joblib_model.predict_proba(df_input)

    # List Prediction outcome
    st.subheader('Prediction ')
    # st.write(prediction)
    if prediction == 0:
        st.write(" Transaction is not Fraud")
    elif prediction == 1:
        st.write("Transaction is Fraud")
    else:
        st.write("Oops...unexpected result...Try again")

    # List Prediction Probability

    st.subheader('Prediction Probability')
  
    df_prediction_proba = pd.DataFrame(predict_proba)
    df_prediction_proba.columns = ['Not Fraud', 'Fraud']
    df_prediction_proba.rename(columns={0: 'Not Fraud',
                                    1: 'Fraud'})
                                
    # Progress Bar for Transaction Probability
    st.dataframe(df_prediction_proba,
                column_config={
                'Not Fraud': st.column_config.ProgressColumn(
                    'Not Fraud',
                    format='%.2f',
                    width='medium',
                    min_value=0,
                    max_value=1
                ),
                'Fraud': st.column_config.ProgressColumn(
                    'Fraud',
                    format='%.2f',
                    width='medium',
                    min_value=0,
                    max_value=1
                ),
                
                }, hide_index=True)