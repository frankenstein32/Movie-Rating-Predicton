from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import clean_review as cl
import Model_Train as model
import pickle 

app = Flask(__name__, template_folder='templates')

# Read the Labels
Y = pd.read_csv("./train_code/trainData/imdb_trainY.txt", header = None)

train_labels = Y.values
train_labels = train_labels[:25000]
"""print(train_labels.shape) Uncomment to check shape of the read labels"""

d_file = open('saved_model.pkl', 'rb')
classes = pickle.load(d_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = [x for x in request.form.values()]
    print(review[0], type(review[0]))
    cleaned_review = cl.parseLine(review[0])
    print(cleaned_review)
    pred = model.prediction(classes, cleaned_review, train_labels)

    return render_template('index.html', result='Prediction: {}'.format(pred))

if __name__ == '__main__':
    app.run(debug=True)