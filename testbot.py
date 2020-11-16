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

nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = []
stop = stopwords.words('english')
stop.remove('not')
ps = PorterStemmer()

for i in dfq['index']:
    review = re.sub('[^a-zA-Z]', ' ', i)
    review = review.lower()
    review = review.split()
    review = [ps.stem(i) for i in review ]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dfq.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=25, activation='relu'))
ann.add(tf.keras.layers.Dense(units=15, activation='relu'))
ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='relu'))
ann.compile(optimizer = 'softmax', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(xtrain, ytrain, batch_size = 32, epochs = 100)


nn_prediction = np.around(ann.predict(xtest))