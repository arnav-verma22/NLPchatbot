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


def preprocess_review(statement):
    review = re.sub('[^a-zA-Z]', ' ', statement)
    review = review.lower()
    review = review.split()
    review = [ps.stem(i) for i in review]
    review = ' '.join(review)
    corpus.append(review)

for i in dfq['index']:
    preprocess_review(i)

def bag_of_words(corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1400)
    x = cv.fit_transform(corpus).toarray()
    return x

ann = tf.keras.models.Sequential()
def building_nn(xtrain, ytrain):
    ann.add(tf.keras.layers.Dense(units=1400, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=800, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=300, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

def train_nn(xtrain, ytrain):
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(xtrain, ytrain, batch_size=32, epochs=100)

def classify(xtest):
    z = np.around(ann.predict(xtest))
    return z

if __name__ == "__main__":
    z = -1
    while z != 2:
        chat = str(input("You: "))
        preprocess_review(chat)
        corpus.append(chat)
        x = bag_of_words(corpus)
        y = dfq.iloc[:, -1].values
        testing = x[-1]
        testing = testing.reshape((1, 1400))
        x = np.delete(x, -1, axis=0)
        print(np.shape(x))

