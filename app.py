
# Author: Kevin Ng
# Date: 4/29/2023 

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st 
import torch 
import pandas as pd 

# title 
st.title("Toxic Tweets") 

# subtitle 
st.markdown("Link to the app- ")

# text input 
user_input = st.text_input("I am amazing!")
if user_input == "":
    user_input = "I am amazing!"

# set up model & tokenizer for trained toxic model 
tokenizer = AutoTokenizer.from_pretrained("Kev07/Toxic-Tweets")
model = AutoModelForSequenceClassification.from_pretrained("Kev07/Toxic-Tweets")
res = []
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# f: input -> classification
def classify(model_name: str, user_input: str):
    # toxic model 
    if model_name == "finetuned distilbert":
        # set up 10 samples 
        test_input = ["Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.", 
                      ":If you have a look back at the source, the information I updated was the correct form. I can only guess the source hadn't updated. I shall update the information once again but thank you for your message.", 
                      "Thank you for understanding. I think very highly of you and would not revert without discussion.",
                      "Please stop. If you continue to vandalize Wikipedia, as you did to Homosexuality, you will be blocked from editing.",
                      "MEL GIBSON IS A NAZI BITCH WHO MAKES SHITTY MOVIES. HE HAS SO MUCH BUTTSEX THAT HIS ASSHOLE IS NOW BIG ENOUGH TO BE CONSIDERED A COUNTRY.",
                      ":Disagree. Soviet railways need their own article. They were administered completely differently, organized differently, and named differently. That the current operation uses the former rolling stock has no bearing and makes the current article a mess. -",
                      "I HATE YOU",
                      "I HATE MILEY CYRUS DDDDDDDDDDDDDD SEE YA SUCKERSSSSSSSSSSSSSSSSSSS DDDDD BTW U WILLLL NEVER KNOW WHO I AM OHH AND MY NAME IS TOM SMITH", 
                      "Dear sir: YOU  are a fucking cunt. Rahm Emanuel is SATANNNNNNNNNNNNNNNNNNN! He enjoys poking naked men, and the world must know that he likes the penis. So stop being a faggot, you cunt.", 
                      "The Yatt got me, this is insane"
                      ]
        # add user input to samples 
        test_input.append(user_input)
        # perform inference on inputs
        for text in test_input:
            batch = tokenizer(text, truncation=True, padding='max_length', return_tensors="pt")
            with torch.no_grad():
                outputs = model(**batch) # train 
                predictions = torch.sigmoid(outputs.logits)*100 # sigmoid activation
                probs = predictions[0].tolist() 
                
            # find maximum between toxic and very toxic
            # find maximum of other categories
            first_max = max(probs)
            index_first = probs.index(first_max)
            second_max = max(probs[2:])
            index_second = probs.index(second_max)

            # update results
            res.append((first_max, index_first, second_max, index_second))
        data = {
                "tweet": test_input,
                "highest class": [ labels[ res[i][1]] for i in range(len( res)) ], 
                "highest percentage": [  res[i][0] for i in range(len( res))],
                "second highest class": [ labels[ res[i][3]] for i in range(len( res)) ], 
                "second highest percentage": [ res[i][2] for i in range(len( res))]
                }
        df = pd.DataFrame(data=data)
        st.table(df) 
    # binary sentiment analysis (positive/negative)
    else:
        classifier = None 
        classifier = pipeline("sentiment-analysis", model=model_name)
        # print classifier 
        st.write("\nInput: ", user_input)
        st.write("Label: ", classifier(user_input)[0]["label"])
        st.write("Accuracy: ", classifier(user_input)[0]["score"])

# buttons to run model
if st.button('Run Distilbert Model'):
    classify("distilbert-base-uncased-finetuned-sst-2-english", user_input)
elif st.button("Run Roberta Model"):
    classify("cardiffnlp/twitter-roberta-base-sentiment", user_input)
elif st.button("Run Toxic Model"):
    classify("finetuned distilbert", user_input)





