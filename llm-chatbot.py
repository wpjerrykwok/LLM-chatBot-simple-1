# %% [markdown]
# # LLM with a ChatBot
# 
# * created by Wai Ping Jerry Kwok
# * on 10 December 2023
# 
# # Problem statement
# 
# To develop a simple chatbot that can answer basic questions about a specific topic.

# %% [markdown]
# # Setup environment

# %%
# import libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import json
import random
import pickle
import requests

import tensorflow as tf

# %% [markdown]
# # Load and Preprocess Data

# %% [markdown]
# **Discussion**
# * The intents file contains the curated information about a smaller subset of Winnipeg Libraries in a Question and Answer format
# * 'patterns' contains the usual patterns that users would input to query the related information
# * 'responses' contains the expected results from the organization to answer the query
# reference: https://wpl.winnipeg.ca/library/

# %%
# load data
# define the URL of the JSON file
url = "https://raw.githubusercontent.com/wpjerrykwok/LLM-chatBot/main/intents_wpglibrary.json"

# make a GET request to the URL
response = requests.get(url)

# check if the request was successful
if response.status_code == 200:
    # parse the response as JSON
    raw_data = response.json()
    # print the raw data
    print(raw_data)
else:
    # handle the error
    print(f"Request failed with status code {response.status_code}")

# %%
stemmer = LancasterStemmer()

# %%
# reminder to delete the pickle file if you change the intents file
try:
    with open('data.pickle', 'rb') as data_file:
        words, labels, training, output = pickle.load(data_file)
except:
# get the words and labels
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in raw_data['intents']:
        for pattern in intent['patterns']:
            tokenized_words = nltk.word_tokenize(pattern)
            words.extend(tokenized_words)
            docs_x.append(tokenized_words)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # stem the words
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # create training and output data
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # one hot encoding
    for x, doc in enumerate(docs_x):
        bag = []

        stemmed_words = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in stemmed_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # convert to numpy arrays
    training = np.array(training)
    output = np.array(output)

    # save data
    with open('data.pickle', 'wb') as data_file:
        pickle.dump((words, labels, training, output), data_file)

# %% [markdown]
# # Train the model

# %%
# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=[len(training[0])]))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# reminder to delete the model file if you change the model
try:
    model.load_weights('model.keras')
except:
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save('model.keras')

# %%
# create a bag of words function to be used in the chat
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    tokenized_words = nltk.word_tokenize(s)
    stemmed_words = [stemmer.stem(w.lower()) for w in tokenized_words]

    for w in stemmed_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# %%
# create a chat function
def chat():
    print('Start talking with the bot! (type quit to stop)')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        results = model.predict(np.array([bag_of_words(inp, words)]), verbose=0)
        results_index = np.argmax(results)
        tag = labels[results_index]

        for intent in raw_data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']

        # print('You: ' + inp)
        print('Bot: ' + random.choice(responses))

# %%
chat()


