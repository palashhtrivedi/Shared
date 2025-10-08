import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle
import google.generativeai as genai

# Configure the Gemini API key
# Important: Replace "YOUR_API_KEY" with your actual key
genai.configure(api_key="AIzaSyB41eIVE-dJ4EnLYpH7NRgdWZOpTsRuy2U")
gemini_model = genai.GenerativeModel('gemini-pro')

with open('intents.json') as file:
    data = json.load(file)

def chat():
    # load trained model
    model = keras.models.load_model('chat-model.keras')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    # Set a threshold for when to use the Gemini model
    confidence_threshold = 0.7 

    print(Fore.YELLOW + 'Start talking with MindMate, your Personal Therapeutic AI Assistant. (Type quit to stop talking)' + Style.RESET_ALL)
    
    while True:
        print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end = "")
        inp = input()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'MindMate:' + Style.RESET_ALL, "Take care. See you soon.")
            break
    
        # Predict the intent using the Keras model
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
        
        # Check the confidence of the prediction
        confidence = np.max(result)

        if confidence > confidence_threshold:
            # If confidence is high, use the pre-defined response from intents.json
            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.GREEN + 'MindMate' + Style.RESET_ALL, random.choice(i['responses']))
        else:
            # If confidence is low, fall back to Gemini
            try:
                response = gemini_model.generate_content(inp)
                print(Fore.GREEN + 'MindMate' + Style.RESET_ALL, response.text)
            except Exception as e:
                print(Fore.RED + 'MindMate' + Style.RESET_ALL, f"I'm sorry, I couldn't generate a response. ({e})")

# Call the chat function to start the conversation
chat()