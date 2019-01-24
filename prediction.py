# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:18:47 2019

@author: KHYATI GARG
"""

dataset = pd.read_csv('test.tsv',delimiter='\t',quoting = 3,encoding = "ISO-8859-1")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
unstemmed = []
    
for i in range(0,2000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Phrase'][i])
    review = review.lower()
    review = review.split()
  #  print(review)
    lemma = nltk.stem.WordNetLemmatizer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review )

from sklearn.externals import joblib
    
cv= joblib.load('corpusmodel.pkl')

test=cv.transform(corpus)

classifier= joblib.load('sentimodel.pkl')

y_pred = classifier.predict(test)

