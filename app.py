import streamlit as st 
import joblib 
import pandas as pd 

st.title("Iris Flower Species Prediction")

data = {
    'sepal length (cm)' : st.number_input("Sepal Length (cm)"),
    'sepal width (cm)' : st.number_input("Sepal Width (cm)"),
    'petal length (cm)' : st.number_input("Petal Length (cm)"),
    'petal width (cm)' : st.number_input("Petal Width (cm)"),
}

df = pd.DataFrame([data])

st.table(df)

model = joblib.load('iris.pkl')
button = st.button("Predict")
if button : 
    prediction = model.predict(df)
    if prediction[0] == 0 :
        prediction = 'Setosa'
    elif prediction[0] == 1 :
        prediction = 'Versicolor'
    elif prediction[0] == 2 :
        prediction = 'Virginica'
    st.success(f'The predicted species is {prediction}')



