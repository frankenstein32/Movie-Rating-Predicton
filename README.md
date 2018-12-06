# Movie-Rating-Predicton
Predicting Movie rating corresponding to the Review Using Naive Bayes Algorithm.
- Data is Given in the form of the text, So NLP is used to clean the data set and then processing the data using the Bag of Words model.

# Flow of the program
-  Clean the data given by created a pipeline.
   * Tokenizing the review.
   * Removing the Stop words.
   * Stemming the words created.
   * Store the cleaned review in a new file.

- Training.<br>
 Training of the data is done by the algorithm Naive - Bayes and usinf the bag of words Model.

- Prediction<br>
Prediction is made my comparing the prediction of Validation data and the true labels of the data.

# How to run
- Run the clean_review.py file and give input file name and output file name along with the command in the terminal.
  for e.g<br>
  
        rashid@rashid-ideapad: python3 clean_review.py imdb_trainX.txt x_train.txt<br>
        
 With this command the clean_review file will clean the text present in the imdb_trainX.txt and write into the x_train.txt
 
 - Do the same cleaning for the imdb_testX.txt file too.
 - Now simply run the movie_prediction.py file
 - It will print the training accuracy and Testing accuracy
 - Keep the files and code in the same directory otherwise you need to change the path for reading and writing the files.
