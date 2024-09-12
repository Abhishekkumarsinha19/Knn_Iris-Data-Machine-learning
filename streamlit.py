import streamlit as st
import joblib
import numpy as np
st.header("Iris Flower Classification Using Knn Model")
sepal_length=st.number_input("Enter Sepal Length")
sepal_width=st.number_input("Enter Sepal Width")
petal_length=st.number_input("Enter Petal Length")
petal_width=st.number_input("Enter Petal Width")

button=st.button("SUBMIT")
loaded_model=joblib.load(r"C:\Users\hp\ML file\knn_iris.pkl")
x=np.array([[sepal_length, sepal_width, petal_length, petal_width]])
predicted_value=loaded_model.predict(x)
decode_dict={0:"Iris-setosa",1:"Iris-virginica",2:"Iris-versicolor"}
predicted_name=decode_dict[predicted_value[0]]

if button:
    st.info(predicted_value)
    st.info(predicted_name)