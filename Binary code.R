#-------------------------------------------------------------------------------
# Step 1: Preparation for binary classification
# Read the data
training_data <- read.table("training_data.txt", header = TRUE)
test_data <- read.table("test_data.txt", header = TRUE)
# create a binary column for dynamic(1) vs static(0)
training_data$activity_type <- ifelse(training_data[, 2] %in% c(1, 2, 3), 1, 0)
# plug this new column as the 3rd column in the new training_data
training_data <- cbind(training_data[, 1:2], 
                       type = training_data$activity_type, 
                       training_data[, 3:ncol(training_data)])
training_data <- training_data[, -which(names(training_data) == "activity_type")]

#-------------------------------------------------------------------------------
# Step 2: 5-fold and comparision of 4 algorithms for binary classification
X <- training_data[, -c(1, 2, 3)] # features matrix
Y <- training_data$type # Response

library(caret)

# Define 5-fold cross-validation
set.seed(123)
folds <- createFolds(Y, k = 5, list = TRUE, returnTrain = TRUE)

# Define models
logistic_model <- train(X, Y, method = "glm", family = "binomial",
                        trControl = trainControl(method = "cv", index = folds))
svm_model <- train(X, Y, method = "svmRadial", 
                   trControl = trainControl(method = "cv", index = folds))
nnet_model <- train(X, Y, method = "nnet", 
                    trControl = trainControl(method = "cv", index = folds))
tree_model <- train(X, Y, method = "rpart", 
                    trControl = trainControl(method = "cv", index = folds))

# Compare the performance of the above 4 algorithms
compare_models <- resamples(list(logistic = logistic_model, 
                                 svm = svm_model, 
                                 nnet = nnet_model, 
                                 tree = tree_model))
summary(compare_models)

#-------------------------------------------------------------------------------
# Step 3: Apply the best algorithm to test_data for binary classification
# binary prediction
prediction <- predict(logistic_model, newdata = test_data[, -1])
# Convert probabilities to binary classification results
threshold <- 0.5
binary_predictions <- ifelse(prediction >= threshold, 1, 0)
# write the txt file to save the predict outcomes
write.table(binary_predictions, 
            file = "binary_0519.txt",
            col.names = FALSE, 
            row.names = FALSE)
binary_pre <- read.table("binary_0519.txt")
nrow(binary_pre)
table(binary_pre)