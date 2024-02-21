import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import streamlit as st
import joblib
data = pd.read_csv('expresso_processed.csv')

df = data.copy()

encoder = LabelEncoder()
scaler = StandardScaler()

df.drop(['Unnamed: 0', 'MRG'], axis = 1, inplace = True)

for i in df.drop('CHURN', axis = 1).columns:
    if df[i].dtypes == 'O':
        df[i] = encoder.fit_transform(df[i])
    else:
        df[i] = scaler.fit_transform(df[[i]])


st.markdown("<h1 style = 'text_align: center; font-family: helvetica; color: #1F4172; '>Expressed CHURN PROJECT</h1", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive'>BUILT BY GOMY TANGO </h4>", unsafe_allow_html = True)

st.image('pngwing.com (9).png', width = 350, use_column_width = True)

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<p>Churn attrition prediction is a vital aspect of customer relationship management, particularly in industries such as telecommunications, subscription services, and banking. It involves the use of data analysis and predictive modeling techniques to forecast the likelihood of customers discontinuing their relationship with a business or service provider. By identifying patterns and trends in customer behavior, such as usage patterns, transaction history, and engagement metrics, businesses can proactively take measures to retain at-risk customers and minimize churn. This proactive approach not only helps in preserving revenue streams but also enables businesses to enhance customer satisfaction and loyalty by addressing potential issues before they escalate.</p", unsafe_allow_html = True)
st.markdown('<br>', unsafe_allow_html = True)
st.dataframe(data, use_container_width = True)

st.sidebar.image('pngwing.com (1).png', caption = 'welcome user')

tenure = st.sidebar.selectbox('Tenure', data['TENURE'].unique())
montant = st.sidebar.number_input('Montant', data['MONTANT'].min(), data['MONTANT'].max())
freq_rech = st.sidebar.number_input('FREQUENCE_RECH', data.FREQUENCE_RECH.min(), data.FREQUENCE_RECH.max())
revenue = st.sidebar.number_input('REVENUE', data.REVENUE.min(), data.REVENUE.max())
arpu_segment = st.sidebar.number_input('ARPU_SEGMENT', data.ARPU_SEGMENT.min(), data.ARPU_SEGMENT.max())
frequence = st.sidebar.number_input('FREQUENCE', data.FREQUENCE.min(), data.FREQUENCE.max())
data_volume = st.sidebar.number_input('DATA_VOLUME', data.DATA_VOLUME.min(), data.DATA_VOLUME.max())
no_net = st.sidebar.number_input('ON_NET', data.ON_NET.min(), data.ON_NET.max())
regularity = st.sidebar.number_input('REGULARITY', data.REGULARITY.min(), data.REGULARITY.max())

new_tenure = encoder.transform([tenure])

input_var = pd.DataFrame({'TENURE': [new_tenure], 'MONTANT':[montant], 'FREQUENCE_RECH':[freq_rech], 
                          'REVENUE':[revenue], 'ARPU_SEGMENT':[arpu_segment], 'FREQUENCE':[frequence], 
                          'DATA_VOLUME' :[data_volume], 'ON_NET': [no_net], 'REGULARITY': [regularity]})

st.dataframe(input_var)

model = joblib.load('ChurnModel.pkl')
predictor = st.button("Tap to predict")

if predictor:
    predicted = model.predict(input_var)
    output = None
    if predicted[0] == 0:
        output = 'Not-churn'
    else:
        output = 'Churn'
    st.success(f'Your predicted attrition is {output}')
    st.balloons()


