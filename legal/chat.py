import random
import json
import pickle
import torch
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())

words=[]
classes=[]
document=[]
ignoreLetters=["?", "!", ".", ","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        document.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
           

# Remove duplicates and sort
ignore_words = ["?", "!", ".", ","]
all_words = sorted(set(all_words) - set(ignore_words))
tags = sorted(set(tags))
# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# Define model
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save the model
model_path = 'nlp_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved as {model_path}')
