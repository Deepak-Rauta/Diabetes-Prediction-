# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:31:47 2024

@author: kdeepak_new
"""



import numpy as np
import pickle
loaded_model = pickle.load(open(r"C:\Users\kdeepak_new\Downloads\Loan_data_model.pkl",'rb'))
#making a predictive system

input_data = (1, 1, 1, 1, 0, 4583, 1508.0, 128.0, 360.0, 1.0, 0)

input_data_array = np.array(input_data)
input_data_reshaped = input_data_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person was not eligible for loan")
else:
    print("The person was eligible for loan")