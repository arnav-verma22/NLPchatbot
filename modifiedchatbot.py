import tensorflow as tf
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import random


with open("intents.json") as file:
    data=json.load(file)

a = data['intents']
dict_q = {}
dict_a = {}
tags = []
questions = []
answers = []
for intent in a:
    tags.append(intent['tag'])
    questions.append(intent['patterns'])
    answers.append(intent['responses'])


for i in range(6):
    for j in range(len(questions[i])):
        dict_q[questions[i][j]] = tags[i]

    for j in range(len(answers[i])):
        dict_a[answers[i][j]] = tags[i]

dfq = pd.DataFrame.from_dict(dict_q, orient="index")
dfq.reset_index(drop=False, inplace=True)


