# LLM-chatBot
a project to develop a simple chatBot

## Motivation
Large language models (LLMs), also known as artificial intelligence (AI) chatbots, are a hyped and controversial topic. The applications of LLMs can be limitless depending on the organization, use case, and data structures. In this project, we used public information to build a chatbot that provides information in a conversation environment. Our project aims to implement a model that can respond to questions within a confined context.

## What the model does
The model uses a curated list of questions and answers about public library services in Winnipeg as a confined context and responds to user inputs accordingly. 

## How we built it
We used JSON to hold the data, tokenized the data and built a fully connected Neural Network as the machine learning model.

## Challenges we faced
It was difficult to obtain a suitable dataset as the corpus because the content should be informational yet need a pattern on how the user would usually ask for it.

## Accomplishments
We successfully built a simple ChatBot that can respond meaningfully to a limited number of questions. 

## What we learned
We learned more about the fundamentals of an LLM, how to call a JSON file, modularize our codes with neat format and concise comments, and learned to use a lot of different ML libraries.

## What's next for LLM ChatBot
We want to build the ChatBot with a friendly user interface, process accordingly even the user input with typos, and build a more automatic pipeline to ingest data as a corpus. 

# Built With
JSON tensorflow nltk stem pickle requests
