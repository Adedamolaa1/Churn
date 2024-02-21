import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression

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

x = df.drop('CHURN', axis = 1)
y = df.CHURN

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.20, stratify= y)

model = LogisticRegression()
model.fit(xtrain, ytrain)

st.title('EXPRESSO CHURN')
st.dataframe(data)

tenure = st.selectbox('TENURE', data.TENURE.unique())
montant = st.number_input('MONTANT', data.MONTANT.min(), data.MONTANT.max())
freq_rech = st.number_input('FREQUENCE_RECH', data.FREQUENCE_RECH.min(), data.FREQUENCE_RECH.max())
revenue = st.number_input('REVENUE', data.REVENUE.min(), data.REVENUE.max())
arpu_segment = st.number_input('ARPU_SEGMENT', data.ARPU_SEGMENT.min(), data.ARPU_SEGMENT.max())
frequence = st.number_input('FREQUENCE', data.FREQUENCE.min(), data.FREQUENCE.max())
data_volume = st.number_input('DATA_VOLUME', data.DATA_VOLUME.min(), data.DATA_VOLUME.max())
no_net = st.number_input('ON_NET', data.ON_NET.min(), data.ON_NET.max())
regularity = st.number_input('REGULARITY', data.REGULARITY.min(), data.REGULARITY.max())

new_tenure = encoder.transform([tenure])
st.write(new_tenure)


# age = st.slider('Age', data['age'].min(), data['age'].max())
# ba = st.select_slider('BA', data['ba'].unique())

# print(data.isnull().sum())