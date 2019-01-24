import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.tsv',delimiter='\t',quoting = 3,encoding = "ISO-8859-1")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
unstemmed = []
    
for i in range(0,15606):
    review = re.sub('[^a-zA-Z]',' ',dataset['Phrase'][i])
    review = review.lower()
    review = review.split()
  #  print(review)
    lemma = nltk.stem.WordNetLemmatizer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review )

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
p=cv
p.fit(corpus)
X = cv.fit_transform(corpus).toarray() 
z = dataset['Sentiment']
y=z[:15606]


from sklearn.externals import joblib
joblib.dump(cv, 'corpusmodel.pkl')
print("corpus Model dumped!")


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

#X_test.to_csv("data.txt")

"""file = open('testfile.txt','w')
print(dataset['importantComment'].values)
for i in range(0,815):  
    if(dataset['importantComment'][i]==1):
        file.write(dataset['commentText'].values.tostring())"""

        
"""from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,y): 
    print("Train:", train_index, "Validation:", test_index) 
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]"""
    
# Fitting classifier to the Training set
# Create your classifier here
    
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X)
xtrain_tfidf =  tfidf_vect.transform(X_train)
xvalid_tfidf =  tfidf_vect.transform(X_test)
"""

from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy',random_state = 0)
classifier.fit(X_train,y_train)

from sklearn.externals import joblib
joblib.dump(classifier, 'sentimodel.pkl')
print("corpus dumped!")



# Load the model that you just saved
lr = joblib.load('corpusmodel.pkl')

# Saving the data columns from training

# Predicting the Test set results

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

pred=pd.DataFrame(y_pred)


"""

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(y_test,predictions)

import xgboost

accur = train_model(xgboost.XGBClassifier(), xtrain_tfidf, y_train, xvalid_tfidf)

"""









