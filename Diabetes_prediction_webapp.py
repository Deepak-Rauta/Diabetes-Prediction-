# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:21:43 2024

@author: kdeepak_new
"""






import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open(r'C:/Users/kdeepak_new/Documents/Naresh IT/ML_projects/trained_model_1.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_array = np.array(input_data, dtype=float)
    input_data_reshaped = input_data_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"

# Main app function
def main():
    # App title and description
    st.title("🔍 Diabetes Prediction Web App")
    st.markdown("""
        ### 🩺 Predict whether a person has diabetes based on their health metrics.
        Fill in the details below and click the **Test Result** button to get the prediction.
    """)

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    Pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=0)
    Glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, step=1, value=120)
    BloodPressure = st.sidebar.number_input("Blood Pressure Level (mm Hg)", min_value=0, max_value=200, step=1, value=80)
    SkinThickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1, value=20)
    Insulin = st.sidebar.number_input("Insulin Level (μU/mL)", min_value=0, max_value=1000, step=1, value=80)
    BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
    DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, value=0.5)
    Age = st.sidebar.number_input("Age (Years)", min_value=0, max_value=120, step=1, value=30)

    # Button for prediction
    if st.button("Predict Diabetes"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        # Display results
        st.success(diagnosis)
        if "not diabetic" in diagnosis:
            st.balloons()
        else:
            st.error("Consult a healthcare provider for further advice.")

    # Footer
    st.markdown("""
    ---
    Developed with ❤️ using [Streamlit](https://streamlit.io/)
    """)

if __name__ == '__main__':
    main()










