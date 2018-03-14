import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

ps = PorterStemmer()

corpus = []
for i in range(1000):
    review = (re.sub( "[^A-Za-z]" , " ", (dataset["Review"][i])).lower()).split()
    review = [ps.stem(words) for words in review if words not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



    
    
    
    

