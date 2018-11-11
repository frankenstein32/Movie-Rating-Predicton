
"""Import Libraries"""
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

""" Instantiate Objects"""
text = "the movie was great but i did not liked it"
token = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))
ps = PorterStemmer()

""" Cleaning the Text
 	1. Tokenization
 	2. Stemming
 	3. Removing the Stop Words
 	4. Converting cleaned list back to String """
def getStemmedReview(review):
    review = review.lower()
    review = review.replace("<br /><br />", " ")
    
    tokens = token.tokenize(review)
    new_tokens = [toke for toke in tokens if toke not in stop]
    stemmed_tokens = [ps.stem(tokn) for tokn in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review

# Getting the Cleaned Reviews
he = getStemmedReview(text)

# Read Dataset
X = pd.read_csv('imdb_trainX.txt', header = None)
x = X.values
x = x[ : 25000]
"""print(x.shape) Uncomment to check Shape of the read data"""

# Read the Labels
Y = pd.read_csv('imdb_trainY.txt', header = None)
y = Y.values
y = y[ : 25000]
"""print(y.shape) Uncomment to check shape of the read labels"""

""" Uncomment to Check the Different Labels provided
print(y)
print(np.unique(y, return_counts = True)) """

""" Preparing the Training Data"""
new_train = []
for ix in range(x.shape[0]):
    lis = x[ix][0].split()
    lis = np.array(lis)
    new_train.append(lis)
new_train = np.array(new_train)
""" print(type(new_train), type(new_train[0]), new_train[0].shape) Uncomment to check the Shape of new Training data"""

""" Naive Bayes Algorithm """

# prior probability
def priorprobability(y_train, class_label):
    ans = np.sum(y[ : ,0] == class_label)
    return ans / y_train.shape[0]

## global Dictionary of word vocab
classes = {1: {},
           2 : {},
           3 : {},
           4 : {},
           7 : {},
           8 : {},
           9 : {},
           10 : {}    
}
for ix in range(new_train.shape[0]):
    for w in np.unique(new_train[ix]):
        if w in classes[y[ix][0]].keys():
            classes[y[ix][0]][w] = classes[y[ix][0]][w] + 1
        else:
            classes[y[ix][0]][w] = 1

""" print(classes) Uncomment to check the Dictionary formed """

# liklihood function
def liklihood(text, class_label):
    lis = text.split()
    ans = 1
    for word in lis:
        if word in classes[class_label].keys():
            ans *= ((classes[class_label][word] + 1)/ (len(classes[class_label]) + 25000))
        else:
            ans *= ((1) / (len(classes[class_label]) + 25000))
    return (ans * 10000)

# Posterior Probability
def posteriorprob(text, class_label):
    return(liklihood(text, class_label) * priorprobability(y,class_label))

""" Prediction """
def prediction(text):
    class_labels = np.unique(y)
    posterior_probabilties = []
    for c in class_labels:
        prob = posteriorprob(text, c)
        posterior_probabilties.append(prob)
    index = np.argmax(posterior_probabilties)
    return class_labels[index]

""" Accuracy """
def accurracy(x, y):
    sum = 0
    for ix in range(y.shape[0]):
          if y[ix][0] == prediction(x[ix][0]):
                sum += 1
    print((sum / y.shape[0]) * 100)
    
accurracy(x,y)

""" Finally Preparing the testing data """
x_test = pd.read_csv('imdb_testX.txt', header = None)
x_ = x_test.values
print(x_.shape)

y_test = pd.read_csv('imdb_testY.txt', header = None)
y_ = y_test.values
print(y_.shape)

""" Accuracy Over the Testing Data """
accurracy(x_, y_)