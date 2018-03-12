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
    




    
    
    
    

