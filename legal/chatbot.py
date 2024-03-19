import random
import json
import pickle
import torch
import numpy as np

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer=WordNetLemmatizer()
with open(r'C:/Users/HP/OneDrive/Desktop/prayatna/Prayatna-Legal-Warriors/legal/intents.json', 'r') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl','rb'))
classesm= pickle.load(open('classes.pkl','rb'))

model=load_model("chatbot_legalwarriors.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results=[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        #return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
        return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list 

'''def get_response(intents_list, intents_json):
    #list_of_intents=intents_json['intents']
    tag = intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result= random.choice(i['responses'])     
            break     
    return result

def get_response(intents_list, intents_json):
    if intents_list:  # Check if the list is not empty
        tag = intents_list[0]['intent']  # Access the first element's 'intent' key
        # Further processing based on the 'tag' value
        return response  # Return the generated response
    else:
        return "No intents found"'''
def get_response(intents_list, intents_json):
    if intents_list:  # Check if the list is not empty
        tag = intents_list[0]['intent']  # Access the first element's 'intent' key
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    return "No intents found"

print('Great: Bot is running!')

while True:
    message=input("")
    ints = predict_class(message)

    #res=get_response(ints,intents)
    res = get_response(ints, intents)
    print(res)


'''import json
def load_intents(filename):
    try:
        with open(filename, 'r') as file:
            intents_data = json.load(file)
        return intents_data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None



def get_response(user_input, intents_data):
    for intent in intents_data['intents']:
        # Your logic to match user input with intents goes here
        # For example, you can check if user_input matches any patterns in intents
        if user_input in intent['patterns']:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

def main():
    intents_filename = 'intents.json'
    intents_data = load_intents(intents_filename)

    if intents_data is not None:
        # Example user input (replace this with your input logic)
        user_input = input("Enter your message: ")

        response = get_response(user_input, intents_data)
        print(response)

if __name__ == "__main__":
    main()'''