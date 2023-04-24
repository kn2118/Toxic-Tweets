from transformers import AutoTokenizer, AutoModel
import streamlit as st 
import torch 

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
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Kev07/Toxic-Tweets")
    model = AutoModel.from_pretrained("Kev07/Toxic-Tweets")

    # tokenize input
    batch = tokenizer(user_input, truncation=True, padding='max_length', return_tensors="pt")

    # run model on tokenized input 
    with torch.no_grad():
        outputs = model(**(batch.cuda()) )
        predictions = F.softmax(outputs.logits, dim=1)
        labels = torch.argmax(predictions, dim= 1)
        labels = [model.config.id2label[label_id] for label_id in labels.tolist()]

        # print classifier 
        st.write("\nInput: ", user_input)
        st.write("Label: ", labels)
        st.write("Accuracy: ", max(predictions))

if st.button('classify'):
    classify("", user_input)




