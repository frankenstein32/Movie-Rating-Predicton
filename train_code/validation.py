import pandas as pd
import Model_Train as model

# Read Cleaned Dataset
f = pd.read_csv('./trainData/x_train.txt', header=None)
lines_reviews = f.values
lines_reviews = lines_reviews[:25000]
"""print(lines_reviews.shape) Uncomment to check Shape of the read data"""

# Read the Labels
Y = pd.read_csv("trainData/imdb_trainY.txt", header = None)

train_labels = Y.values
train_labels = train_labels[:25000]
"""print(train_labels.shape) Uncomment to check shape of the read labels"""
""" Accuracy on the training data """
print("Accuracy on Training Set is %f"% (model.accuracy(lines_reviews,train_labels)))

""" Finally Preparing the testing data """
x_test = pd.read_csv('./testData/x_test.txt', header = None)
x_ = x_test.values
"""print(x_.shape) To print the shape of the x_ """

y_test = pd.read_csv('./testData/imdb_testY.txt', header = None)
y_ = y_test.values
"""print(y_.shape) To print the shape of the y_"""

""" Accuracy Over the Testing Data """
print("Accuracy on Testing Set is %f"% (model.accuracy(x_, y_)))