import random
import json
import pickle
import torch
import tensorflow as tf
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer=WordNetLemmatizer()
#intents=json.loads(open('intents.json').read())
with open(r'C:/Users/HP/OneDrive/Desktop/prayatna/Prayatna-Legal-Warriors/legal/intents.json', 'r') as file:
    intents = json.load(file)


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
           
words=[lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words=sorted(set(classes))

classes=sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
outputEmpty=[0]*len(classes)

for document in document:
    bag=[]
    wordPatterns=document[0]
    wordPatterns=[lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow=list(outputEmpty)
    outputRow[classes.index(document[1])]=1
    training.append(bag+outputRow)
random.shuffle(training)
training=np.array(training)


trainX=training[:, :len(words)]
trainY=training[:, :len(words):] 

model=tf.keras.Sequential() 

model.add(tf.keras.layers.Dense(128,input_shape=(len(trainX[0]),),activation='relu'))

#to reduce overfitting
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]),activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Remove duplicates and sort

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(trainX),np.array(trainY),epochs=100,batch_size=5,verbose=1)
model.save('chatbot_legalwarriors.h5',hist)
print("executed")