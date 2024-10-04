import streamlit as st
import numpy as np
import tensorflow 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load te trained model
model=tf.keras.models.load_model('model.h5')

#Load the encoder and scaler
with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)        

##Streamlit app

st.title('Customer Churn Prediction')   


Geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
Gender=st.selectbox('Gender',label_encoder_gender.classes_)
Age=st.slider('Age',18,92)
Balance=st.number_input('Balance')
CreditScore=st.number_input('Credit Score')
EstimatedSalary=st.number_input('Estimated Salary')
Tenure=st.slider('Tenure',0,10)
NumOfProducts=st.slider('Number Of Products',1,4)
HasCrCard=st.selectbox('Has Credit Card',[0,1])
IsActiveMember=st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[CreditScore],
    'Gender':[label_encoder_gender.transform([Gender])[0]],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
})

#One-Hot encoded Geography
geo_encoded=one_hot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

#Concatenation one hot encoded
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

##Scaling the input data
input_scaled=scaler.transform(input_data)

#Prediction
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')   
