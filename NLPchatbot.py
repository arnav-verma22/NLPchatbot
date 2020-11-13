import tensorflow as tf
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = []
stop = stopwords.words('english')
stop.remove('not')
ps = PorterStemmer()

for i in df['Review']:
    review = re.sub('[^a-zA-Z]', ' ', i)
    review = review.lower()
    review = review.split()
    review = [ps.stem(i) for i in review if i not in set(stop)]
    review = ' '.join(review)
    corpus.append(review)

comment = str(input('Please give some feedback: '))
corpus.append(comment)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1400)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

testing = x[-1]
testing = testing.reshape((1, 1400))


x = np.delete(x, -1, axis=0)
print(np.shape(x))

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(xtrain, ytrain)

#print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))
y_pred = classifier.predict(xtest)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=1400, activation='relu'))
ann.add(tf.keras.layers.Dense(units=800, activation='relu'))
ann.add(tf.keras.layers.Dense(units=300, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(xtrain, ytrain, batch_size = 32, epochs = 100)


nn_prediction = np.around(ann.predict(xtest))
#y_pred = (round(y_pred))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, y_pred)
print(cm)
print(accuracy_score(ytest, y_pred))

cmn = confusion_matrix(ytest, nn_prediction)
print('NN accuracy', cmn)
print(accuracy_score(ytest, nn_prediction))

z = ann.predict([testing])
if z == 1:
    print("Thank you sir..")
else:
    print("sorry for the discomfort we will try to improve on that...")

