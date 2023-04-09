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

if st.button('Submit'):
    st.write(classifier(user_input))