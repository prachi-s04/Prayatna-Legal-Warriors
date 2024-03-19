import json
import random

import os

# Check if the file exists
if os.path.exists('intents.json'):
    with open('intents.json', 'r') as file:
        # Your code to read the file goes here
        pass  # Placeholder for actual code
else:
    print("File 'intents.json' not found.")

# Load the dataset
'''with open('intents.json', 'r') as file:'''
data = json.load("intents")

# Function to process user input and generate response
def get_response(user_input):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            if user_input.lower() == pattern.lower():
                return random.choice(intent['responses'])
    return "I'm sorry, I don't understand that."

# Main function to interact with the chatbot
def main():
    print("Welcome to the Constitutional Rights Chatbot!")
    print("You can ask questions related to fundamental rights, equality, discrimination, and more.")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the chatbot. Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
