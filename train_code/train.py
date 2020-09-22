import pandas as pd
import numpy as np
import pickle

""" global Dictionary of word vocab """
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

def train():
    """ Read Cleaned Dataset """
    f = pd.read_csv('./trainData/x_train.txt', header=None)
    lines_reviews = f.values
    lines_reviews = lines_reviews[:25000]
    """print(lines_reviews.shape) Uncomment to check Shape of the read data"""

    # Read the Labels
    Y = pd.read_csv("./trainData/imdb_trainY.txt", header = None)

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


    for ix in range(word_reviews.shape[0]):
        
        for word in np.unique(word_reviews[ix]):
            if word in classes[train_labels[ix][0]].keys():
                classes[train_labels[ix][0]][word] += 1
            else:
                classes[train_labels[ix][0]][word] = 1
                
    """ print(classes) Uncomment to check the Dictionary formed """
    # print(classes)

    d_file = open('../saved_model.pkl','wb')
    pickle.dump(classes, d_file)
    d_file.close()

train()