# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:21:43 2024

@author: kdeepak_new
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open(r'C:/Users/kdeepak_new/Documents/Naresh IT/ML_projects/trained_model_1.sav','rb'))

# creating a function

def diabetes_prediction(input_data):

    # Now we change this input data to numpy array bcz the above input data is a form of list
    input_data_array  = np.array(input_data)

    # reshape the array as we are predicting for one instances
    # here we are predicting only one instances generally we have 786 examles and one columns we did n't reshape it then input columns is except 786 instances
    input_data_reshaped = input_data_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The person is not diabetics"
    else:
        return "The person was diabetics"
    
def main():
    # Giving a title
    st.title("Diabetes Predictions Web App")
    
    # Getting the input data from the user
    Pregnancies = st.text_input("Number of pregnasis")
    Glucose	 = st.text_input("Number of Gloucose level")
    BloodPressure = st.text_input("Number of Bloodpressure level")
    SkinThickness = st.text_input("skinthickness value")
    Insulin = st.text_input("insulin value")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("diabetes pedigree function value")
    Age = st.text_input("Age of the persion")
    
    # code for prediction 
    diagnosis = ''
    
    # creating a button for prediction
    if st.button("Diabetes Test result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    