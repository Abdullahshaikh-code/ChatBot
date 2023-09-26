import json
import torch
from  sklearn.feature_extraction.text  import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

Tfid_Vectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words="english",max_df=0.90,min_df=0.10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

data = torch.load("TrainedData.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    Input_sentence = input("")
    if Input_sentence == "quit":
        break
    sentence = tokenize(Input_sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                responses=intent["responses"]
                responses.append(Input_sentence)
                tfidf=Tfid_Vectorizer.fit_transform(responses)                
                vals = cosine_similarity(tfidf[-1], tfidf)
                idx=vals.argsort()[0][-2]
                flat=vals.flatten()
                flat.sort()
                req_tfidf=flat[-2]
                if(req_tfidf<0.1 ):
                     print("I am sorry. Unable to understand you!")

                else:
                   print(f"{bot_name}: {responses[idx]}")
                
    else:
        print(f"{bot_name}: I am sorry. Unable to understand you!")