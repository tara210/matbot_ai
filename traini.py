import json 
from nltk_utils import tokenize,stemm,bow
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

#loading json file  
with open('trig.json','r') as file:
    d=json.load(file)
ignore_words = ['?', '.', '!']
all_words=[]
clas=[]
xy=[]
for i in d:
    if i['class'] not in clas:
        clas.append(i['class'])
    for w in tokenize(i['question']):
        all_words.append(w)
        xy.append((w,i['class']))
all_words = [word for sublist in all_words for word in sublist]
all_words=list(set(all_words))
#print(all_words)
clas=list(set(clas))
#print("&&&&&",clas)
#print(xy)
#print(len(xy), "questions",xy)
#print(len(clas), "tags:", clas)
#print(len(all_words), "unique stemmed words:", all_words)
X_train = []
y_train = []
for (pattern_sentence, cla) in xy:
    
    # X: bag of words for each pattern_sentence
    bag = bow(pattern_sentence, all_words)
 
    X_train.append(bag)
    
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = clas.index(cla)
    y_train.append(label)
X_train=np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 64
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 64
output_size = len(clas)
#print(input_size,len(all_words), output_size)
#print(X_train)
#print(y_train)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:    
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"class": clas
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
