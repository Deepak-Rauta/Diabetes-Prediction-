# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:33:47 2024

@author: kdeepak_new
"""

import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\kdeepak_new\Downloads\Loan_data_model.pkl", 'rb'))

# Function for prediction
def loan_prediction(input_data):
    input_data_array = np.array(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return "The person is NOT eligible for a loan."
    else:
        return "The person is eligible for a loan."

# Main UI function
def main():
    # Set up the page title
    st.title("🏦 Loan Prediction App")
    st.write("Provide the required information below to check loan eligibility.")

    # Input fields
    Gender = st.text_input("Gender (0 for Female, 1 for Male):")
    Married = st.text_input("Married (0 for No, 1 for Yes):")
    Dependents = st.text_input("Number of Dependents (0, 1, 2, or 3):")
    Education = st.text_input("Education (0 for Not Graduate, 1 for Graduate):")
    Self_Employed = st.text_input("Self-Employed (0 for No, 1 for Yes):")
    ApplicantIncome = st.text_input("Applicant Income:")
    CoapplicantIncome = st.text_input("Co-applicant Income:")
    LoanAmount = st.text_input("Loan Amount:")
    Loan_Amount_Term = st.text_input("Loan Term (in months):")
    Credit_History = st.text_input("Credit History (0 for Bad, 1 for Good):")
    Property_Area = st.text_input("Property Area (0 for Rural, 1 for Semiurban, 2 for Urban):")

    # Button for prediction
    if st.button("Predict Loan Eligibility"):
        try:
            # Convert inputs to appropriate types
            input_data = [
                int(Gender), int(Married), int(Dependents), int(Education),
                int(Self_Employed), float(ApplicantIncome), float(CoapplicantIncome),
                float(LoanAmount), float(Loan_Amount_Term), int(Credit_History), int(Property_Area)
            ]
            
            # Get prediction
            result = loan_prediction(input_data)
            st.success(result)
        except ValueError:
            st.error("Please ensure all inputs are filled correctly.")

if __name__ == '__main__':
    main()
