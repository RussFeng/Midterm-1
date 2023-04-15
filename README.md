# Midterm-1
## Binary Classification
Please refer to "Binary code.R" for this part.

In step 1, I made some preparation for the following data analysis and model training.

In step 2, I used 5-fold cross-validation to compare the performance of logistic_model, svm_model, nnet_model, and tree_model. The table of comparison can be reproduced in my code. I finally chose the logistic model as the best one.

In step 3, I applied logistic model to classify the test data. Though I forgot to train the model using the whole training data at the first time, I achieved a 100% accuracy on the leaderboard.

## Multi-Classification
Please refer to "Multiclass code.R" for this part.

In step 1, I made preparations of the training data, including the creation of a multi_type column for further use.

In step 2, I used cross-validation to compare XGboost, gbm, Random forest, and svm.

In step 3, I drew a table and a plot to show the result of comparison. 

In step 4, I used svm model for my final model to classify the test data.

In step 5, I drew a trend plot to show my leaderboard performance. The best point appears in the second time, where I used the svm model without feature selection. Then, I also tried several tree models, but the performance were worse than the svm model. To further improve the classification performance in the test data, I planned to apply feature selection to avoid overfitting and enhance the generalization ability of the model.

In step 6, I showed my code trying to do the feature selection by rfe(). However, it was time-consuming, and I failed to test the performance of the svm model with feature selection. I think this could be a potential way for further improvement.
