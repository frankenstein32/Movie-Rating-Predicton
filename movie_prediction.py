"""Import Libraries"""
import pandas as pd
import numpy as np

# Read Cleaned Dataset
f = pd.read_csv('x_train.txt', header=None)
lines_reviews = f.values
lines_reviews = lines_reviews[:25000]
"""print(lines_reviews.shape) Uncomment to check Shape of the read data"""

# Read the Labels
Y = pd.read_csv("imdb_trainY.txt", header = None)

train_labels = Y.values
train_labels = train_labels[:25000]
"""print(train_labels.shape) Uncomment to check shape of the read labels"""

""" Uncomment to Check the Different Labels provided
print(train_labels)
print(np.unique(train_labels, return_counts = True)) """

""" Preparing the Training Data"""
word_reviews = []

for ix in range(lines_reviews.shape[0]):
    
    words = lines_reviews[ix][0].split()
    words = np.array(words)
    word_reviews.append(words)
word_reviews = np.array(word_reviews)

""" print(word_reviews.shape) to print the shape of the word_reviews """

## global Dictionary of word vocab
classes = { 1 : {},
            2 : {},
            3 : {},
            4 : {},
            5 : {},
            6 : {},
            7 : {},
            8 : {},
            9 : {},
            10 : {}
}

for ix in range(word_reviews.shape[0]):
    
    for word in np.unique(word_reviews[ix]):
        if word in classes[train_labels[ix][0]].keys():
            classes[train_labels[ix][0]][word] += 1
        else:
             classes[train_labels[ix][0]][word] = 1
            
""" print(classes) Uncomment to check the Dictionary formed """

""" Naive Bayes Algorithm """

# prior probability
def prior(labels,label):
    ans = np.sum(labels[:,0] == label)
    return ans/labels.shape[0]

# liklihood function
def likelihood(test_review,class_label):
    
    test_review = test_review.split(" ")
    prob = 1
    for ix in range(len(test_review)):
        
        word = test_review[ix]
        
        if word in classes[class_label].keys():
            prob *= ((classes[class_label][word] + 1)/(len(classes[class_label]) + 25000))
        else:
            prob *= ((1)/(len(classes[class_label]) + 25000))
        
    return (prob*10000)
# Posterior Probability
def posterior(test_review,class_label):
    return likelihood(test_review,class_label) * prior(train_labels,class_label)

""" Prediction """
def prediction(review):   
    class_labels = np.unique(train_labels)
    probs = []
    
    for ix in range(class_labels.shape[0]):
        
        label = class_labels[ix]
        prob = posterior(review,label)
        probs.append(prob)
    index = np.argmax(probs)
    return class_labels[index]
    

""" Accuracy """
def accuracy(reviews,labels):    
    count = 0
    for ix in range(labels.shape[0]):
        
        if labels[ix] == prediction(reviews[ix][0]):
            count += 1
    return ((count/labels.shape[0])*100)

""" Accuracy on the training data """
print("Accuracy on Training Set is %f"%accuracy(lines_reviews,train_labels))

""" Finally Preparing the testing data """
x_test = pd.read_csv('x_test.txt', header = None)
x_ = x_test.values
"""print(x_.shape) To print the shape of the x_ """

y_test = pd.read_csv('imdb_testY.txt', header = None)
y_ = y_test.values
"""print(y_.shape) To print the shape of the y_"""

""" Accuracy Over the Testing Data """
print("Accuracy on Testing Set is %f"%accuracy(x_, y_))