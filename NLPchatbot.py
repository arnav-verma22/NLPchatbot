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
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

testing = x[-1]

x = np.delete(x, -1, axis=0)
print(np.shape(x))

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, y_pred)
print(cm)
print(accuracy_score(ytest, y_pred))

z = classifier.predict([testing])
if z == 1:
    print("Thank you sir..")
else:
    print("sorry for the discomfort we will try to improve on that...")

