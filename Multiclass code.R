#-------------------------------------------------------------------------------
# Step 1: Preparation for multi-classification

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
table(training_data$type)
# create a multi_type column
training_data$multi_type <- ifelse(training_data[, 2] %in% 7:12, 
                                   7, training_data$activity)
training_data <- cbind(training_data[, 1:3], 
                       multitype = training_data$multi_type, 
                       training_data[, 4:ncol(training_data)])
training_data <- training_data[, -which(names(training_data) == "multi_type")]
head(training_data)
table(training_data$activity,training_data$multitype)

#-------------------------------------------------------------------------------
# Step 2: Train the data and compare baseline models
# Compare: XGboost, gbm, svm, 
library(caret)
library(gbm)
library(xgboost)
library(randomForest)
library(e1071) # svm

# cross-validation
set.seed(123) 
trainIndex <- createDataPartition(training_data$multitype, p = 0.8, list = FALSE)
train <- training_data[trainIndex,]
test <- training_data[-trainIndex,]
train <- train[, -c(1, 2, 3)]
test <- test[, -c(1, 2, 3)]

# gbm
gbm_model <- gbm(multitype ~ ., 
                 data = train, 
                 distribution = "multinomial", 
                 n.trees = 100, 
                 interaction.depth = 3)
gbm_pred <- predict(gbm_model, 
                    newdata = test[, -1],
                    n.trees = 100, 
                    type = "response")
gbm_pred <- apply(gbm_pred, 1, which.max)
gbm_acc <- mean(gbm_pred == test$multitype)
print(paste("GBM Accuracy:", round(gbm_acc, 4)))

# xgboost
train$multitype <- train$multitype - 1
test$multitype <- test$multitype - 1
xgb_model <- xgboost(data = as.matrix(train[, -1]), 
                     label = train$multitype,
                     max.depth = 3, 
                     eta = 0.1, 
                     nround = 100, 
                     objective = "multi:softmax", 
                     num_class = 7)

dtest <- xgb.DMatrix(data = as.matrix(test[, -1]))
xgb_pred <- predict(xgb_model, newdata = dtest)
xgb_pred <- xgb_pred + 1
xgb_acc <- mean(xgb_pred == test$multitype)
print(paste("XGBoost Accuracy:", round(xgb_acc, 4)))

# Random forest
rf_model <- randomForest(as.factor(multitype) ~ ., 
                         data = train, 
                         ntree = 500, 
                         importance = TRUE)
rf_pred <- predict(rf_model, test[, -1])
rf_acc <- mean(rf_pred == test$multitype)
print(paste("Random Forest Accuracy:", rf_acc))

# SVM
svm_model <- svm(as.factor(multitype) ~ ., 
                 data = train,
                 kernel = "linear")
svm_pred <- predict(svm_model, test[, -1])
svm_acc <- mean(svm_pred == test$multitype)
print(paste("SVM Accuracy:", svm_acc))

#------------------------------------------------------------------------------

# Step 3: Comparison
# install.packages("knitr")
# install.packages("kableExtra")
library(knitr)
library(kableExtra)

# Create a dataframe
results_df <- data.frame(
  Method = c("GBM", "XGBoost", "Random Forest", "SVM"),
  Accuracy = c(gbm_acc, xgb_acc, rf_acc, svm_acc)
)

# Unify the form of result
results_df$Accuracy <- sprintf("%.4f", results_df$Accuracy)

# Make the table
kable(results_df, col.names = c("Method", "Accuracy")) %>%
  kable_styling(full_width = FALSE, position = "center", 
                font_size = 18, bootstrap_options = "striped")

# Make the plot
library(ggplot2)

results_df <- data.frame(
  Method = c("GBM", "XGBoost", "Random Forest", "SVM"),
  Accuracy = c(gbm_acc, xgb_acc, rf_acc, svm_acc)
)

ggplot(results_df, aes(x = Method, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ggtitle("Model Comparison") +
  xlab("Method") +
  ylab("Accuracy") +
  theme_minimal()

#-------------------------------------------------------------------------------
# Step 4: Apply SVM
train <- training_data[, -c(1, 2, 3)]
svm_model <- svm(as.factor(multitype) ~ ., 
                 data = train,
                 kernel = "linear")
svm_pred <- predict(svm_model, test_data[,-1])
svm_pred <- gsub('"', '', svm_pred)
write.table(svm_pred, 
            file = "multiclass_0519.txt",
            quote = FALSE, col.names = FALSE, row.names = FALSE)
data <- read.table("multiclass_0519.txt")
table(data)
table(training_data$multitype)
nrow(data)

#-------------------------------------------------------------------------------
# Step 5: My leaderboard performance

trend_data <- c(0.922, 0.959, 0.928, 0.930, 0.934)
plot(trend_data,
     ylim = c(0.9, 0.97), 
     xlim = c(0.5, 5.5),
     type = "o", 
     col = "blue", 
     pch = 20, 
     lwd = 2, 
     xlab = "Data",
     ylab = "Value", 
     main = "Trend Chart with Data Points")
text(x = 1:5, 
     y = trend_data, 
     labels = trend_data, 
     pos = 3)

#-------------------------------------------------------------------------------
# Step 6: Potential improvment
# feature selection
svmProfile <- rfe(x = train[, -1], y = as.factor(train[, 1]),
                  sizes = c(350, 400, 450, 500, 561),
                  rfeControl = rfeControl(functions = caretFuncs,
                                          number = 200),
                  ## pass options to train()
                  method = "svmLinear") # or "svmRadial"
svmProfile

# Obtain the best subset
best_features <- svm_rfe$optVariables
train_subset <- train[, c(1, best_features + 1)]
