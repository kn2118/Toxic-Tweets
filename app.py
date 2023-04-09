from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st 

# title 
st.title("Toxic Tweets") 

# subtitle 
st.markdown("Link to the app- ")

# text input 
user_input = st.text_input("I am amazing!")
if user_input == "":
    user_input = "I am amazing!"

# f: input -> classification
def classify(model_name: str, user_input: str):
    
    classifier = pipeline("sentiment-analysis", model=model_name)
    # print classifier 
    st.write("\nInput: ", user_input)
    st.write("Label: ", classifier(user_input)[0]["label"])
    st.write("Accuracy: ", classifier(user_input)[0]["score"])

if st.button('distilbert-base-uncased-finetuned-sst-2-english'):
    classify("distilbert-base-uncased-finetuned-sst-2-english", user_input)
elif st.button("cardiffnlp/twitter-roberta-base-sentiment"):
    classify("cardiffnlp/twitter-roberta-base-sentiment", user_input)




