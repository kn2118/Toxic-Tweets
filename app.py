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
    # prep model 
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model, tokenizer)

    # print classifier 
    st.write("Input: ", user_input)
    st.write("Label: ", classifier(user_input)[0]["label"])
    st.write("Accuracy: ", classifier(user_input)[0]["score"])

if st.button('distilbert-base-uncased-finetuned-sst-2-english'):
    classify("distilbert-base-uncased-finetuned-sst-2-english", user_input)
elif st.button('finiteautomata/bertweet-base-sentiment-analysis'):
    classify("finiteautomata/bertweet-base-sentiment-analysis", user_input)
elif st.button('model3'):
    classify("model3", user_input)




