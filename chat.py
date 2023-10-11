import random
import json
from flask import Flask, render_template, request, jsonify
import torch

from model import NeuralNet
from nltk_utils import bow, tokenize

app = Flask(__name__)

    
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def process():
    user_message = request.args.get('msg')  # Get the user's input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('trig.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['class']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    response=" "
    count=0
    print("Let's chat! (type 'quit' to exit)")
    while True:
    # sentence = "do you use credit cards?"
        sentence = user_message
        if (sentence == "quit"):
            break

        sentence = tokenize(sentence)
    
        X = bow(sentence[0], all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        response=""
        tag = tags[predicted.item()]
        count=0
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents:
                count=count+1
                if tag == intent["class"]:
                    
                    result = all(word in sentence for word in tokenize(intent['question']))
                    
                    if result:
                        
                        response=response+intent['explanation']
                        #print(response)
                        break
                    else:
                        continue
        else:
            response="sorry :( I dont understand.."

    
    res=response
    return jsonify({"response": res})
    

if __name__ == "__main__":
    app.run(debug=True)
