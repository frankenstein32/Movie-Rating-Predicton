"""Import Libraries"""
import pandas as pd
import numpy as np
import pickle

""" Naive Bayes Algorithm """

""" prior probability """
def prior(labels,label,train_labels):
    ans = np.sum(labels[:,0] == label)
    return ans/labels.shape[0]

""" liklihood function """
def likelihood(classes, test_review, class_label):
    
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
def posterior(test_review,class_label, train_labels, classes):
    return likelihood(classes, test_review,class_label) * prior(train_labels,class_label,train_labels)

""" Prediction """
def prediction(classes, review, train_labels):   
    class_labels = np.unique(train_labels)
    probs = []
    
    for ix in range(class_labels.shape[0]):
        
        label = class_labels[ix]
        prob = posterior(review,label, train_labels, classes)
        probs.append(prob)
    index = np.argmax(probs)
    print(class_labels[index])
    return class_labels[index]

""" Accuracy """
def accuracy(reviews,labels, train_labels, classes):    
    count = 0
    for ix in range(labels.shape[0]):
        
        if labels[ix] == prediction(classes, reviews[ix][0],train_labels):
            count += 1
    return ((count/labels.shape[0])*100)

