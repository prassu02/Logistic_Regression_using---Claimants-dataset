#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model= pickle.load(open('log31.pkl','rb'))
model


# In[3]:


st.title('Model Deployment Using Logistic Regression')


# In[4]:


def user_input_parameter():
    CLMSEX= st.sidebar.selectbox('Enter Your Gender, Female--0, Male--1',[0,1])
    CLMINSUR= st.sidebar.selectbox('Insurance Detail, Yes--1, No--0',[0,1])
    SEATBELT= st.sidebar.selectbox('Seatbelt Detail, Yes--1, No--0',[0,1])
    CLMAGE= st.sidebar.slider('Enter the Age', 0,100)
    LOSS= st.sidebar.number_input('Enter the Loss')
    dict1= {'CLMSEX':CLMSEX,'CLMINSUR':CLMINSUR,'SEATBELT':SEATBELT,'CLMAGE':CLMAGE,'LOSS':LOSS}
    features= pd.DataFrame(dict1,index=[0])
    return features
df= user_input_parameter()
predicted= model.predict(df)
pred_proba= model.predict_proba(df)
st.subheader('Predicted')
st.write('Approved' if pred_proba[0][1]>=0.5 else 'Not Approved')
## the below the code 'dont need this code is optional'
st.subheader('Predicted_Proba')
st.write(pred_proba)


# In[ ]:




