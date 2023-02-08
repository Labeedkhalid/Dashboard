import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
st.title('Heart Disease Prediciton WebApp')



st.sidebar.header('User Input Features')


def user_input_features():  
    age = st.sidebar.slider('Enter Your Age:')
    sex = st.sidebar.selectbox('sex',(0,1))
    cp=st.sidebar.selectbox('Chest pain type ',(0,1,2,3))
    tres = st.sidebar.slider('Resting blood pressure:')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl:')
    fbs = st.sidebar.selectbox('Fasting blood sugar ',(0,1))
    res= st.sidebar.number_input('Resting electrocardiographic results:')
    tha = st.sidebar.number_input('Maximum geart rate achieved:')
    exa = st.sidebar.selectbox('Exercise induced angina:',(0,1))
    old=st.sidebar.number_input('Oldpeak')
    slope = st.sidebar.number_input('slope of the peak exercise ST segmen:')
    ca =st.sidebar.selectbox('Number of major vessels', (0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2))
    
    
    df=    {'age':age,
            'sex':sex,
            'cp':cp, 
            'trestbps':tres,
            'chol':chol,
            'fbs':fbs,
            'restecg':res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal}
    
    features = pd.DataFrame(df , index=[0])
    return features
input_df = user_input_features()


#combining user input features with entire dataset

heart_dataset= pd.read_csv('heart disease.csv')
heart_dataset= heart_dataset.drop(columns=['target'])

df= pd.concat([input_df,heart_dataset],axis=0)



# df= pd.get_dummies(df,columns=['sex' , 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope', 'ca', 'thal'])

df = df[:1]  #select only first row

st.write(input_df)

load_clf = pickle.load(open('Random_forest_model.pk2','rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)