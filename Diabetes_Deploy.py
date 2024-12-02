# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
loaded_model = pickle.load(open(r'C:/Users/kdeepak_new/Documents/Naresh IT/ML_projects/trained_model_1.sav','rb'))

input_data = (3,126,88,41,235,39.3,0.704,27)

# Now we change this input data to numpy array bcz the above input data is a form of list
input_data_array  = np.array(input_data)

# reshape the array as we are predicting for one instances
# here we are predicting only one instances generally we have 786 examles and one columns we did n't reshape it then input columns is except 786 instances
input_data_reshaped = input_data_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetics")
else:
    print("The person was diabetics")
