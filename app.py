from transformers import pipeline 
import streamlit as st 

# title 
st.title("Toxic Tweets") 

# subtitle 
st.markdown("Link to the app- ")

# define model 
classifier = pipeline("sentiment-analysis")

# text input 
user_input = st.text_input("Enter your text here.")

if user_input == "":
    user_input = "I am amazing!"

if st.button('Submit'):
    st.write("Input: ", user_input)
    st.write("Label: ", classifier(user_input)[0]["label"])
    st.write("Accuracy: ", classifier(user_input)[0]["score"])