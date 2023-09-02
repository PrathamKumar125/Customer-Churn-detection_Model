
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

df_= pd.read_csv("Model/Customer Churn.csv")
scaler.fit_transform(df_)
# scaler.transform(df_)

# loading the saved model
loaded_model = pickle.load(open('Model/trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    
  input_data_as_numpy_array = np.asarray(input_data)

  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  
  scaler.fit_transform(input_data_reshaped)
  X= scaler.transform(input_data_reshaped)

  prediction =loaded_model.predict(X)
  print(prediction)

  if (prediction[0] == 0):
   return 'Customer is not churn'
  else:
    return 'Customer is churn'
    
  
def main():
    
    
    # giving a title
    st.title('Customer Churn Prediction Web App')
    
    
    # getting the input data from the user
    
    CallFailure = st.text_input('Number of Call Failure')
    Complains = st.text_input('Number of Complains')
    Subscription_Length = st.text_input('Subscription Length')
    Charge_Amount= st.text_input('Charge Amount')
    Seconds_of_Use= st.text_input('Seconds of Use')
    Frequency_of_use= st.text_input('Frequency of use')
    Frequency_of_SMS= st.text_input('Frequency of SMS')
    Distinct_Called_Numbers= st.text_input('Distinct_Called_Numbers')
    Age_Group= st.text_input('Age Group(1-5)')
    Tariff_Plan= st.text_input('Tariff Plan')
    Status= st.text_input('Status')
    Age= st.text_input('Age')
    Customer_Value= st.text_input('Customer Value')

    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([CallFailure,Complains,Subscription_Length,Charge_Amount,Seconds_of_Use,Frequency_of_use,Frequency_of_SMS,Distinct_Called_Numbers,Age_Group,Tariff_Plan,Status,Age,Customer_Value])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  
