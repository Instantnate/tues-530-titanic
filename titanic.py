import streamlit as st
import pandas as pd
from fastai.vision.all import *

st.title("Titanic Survivorship")

titanic_model = load_learner("titanic_model.pkl")

name = st.text_input("What is your name?")
age = st.number_input("How old are you?")
sex = st.radio("What is your gender", "(male/female)")
fare = st.number_input("Enter your ticket price.")
sib_sp = st.number_input("Enter the number of people in your party.")
parch = st.number_input("Enter Parent of Children: ")
pclass = st.number_input("Passenger class: ")

if st.button("Submit"):
    user_input = {
        "Age": [age],
        "Sex": [sex],
        "Fare": [fare],
        "SibSp": [sib_sp],
        "Parch": [parch],
        "Pclass": [pclass]
    }

    new_df = pd.DataFrame(user_input)
    prediction = titanic_model.predict(new_df.iloc[0])
    st.title(f"{name}, the chance you will surived the Titanic disaster would be: {prediction[2][1] * 100}%")