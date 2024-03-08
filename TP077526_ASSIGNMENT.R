
# Read the data set
df <- read.csv('heart.csv', header = TRUE)

str(df)
summary(df)

# Libraries
#-------------
library(dplyr) # A Grammar of Data Manipulation
library(DataExplorer) # Automate Data Exploration and Treatment
library(VIM) # VIM introduces tools for visualization of missing and imputed values
library(mice) # Multivariate Imputation by Chained Equations
library(missForest) # Nonparametric Missing Value Imputation using Random Forest
library(caTools) # Tools: Moving Window Statistics, GIF, Base64, ROC AUC, etc
library(caret) # Classification and Regression Training
library(ROCR) # Visualizing the Performance of Scoring Classifiers
library(ggcorrplot) # Visualization of a Correlation Matrix using 'ggplot2'
library(pROC) # Display and Analyze ROC Curves
library(PRROC) # Precision-Recall and ROC Curves for Weighted aznd Unweighted Data
library(ggplot2) # Create Elegant Data Visualisations Using the Grammar of Graphics
library(glmnet) # Lasso and Elastic-Net Regularized Generalized Linear Models
library(rpart) # Recursive Partitioning and Regression Trees
library(Metrics) # Evaluation Metrics for Machine Learning
library(class) # Functions for Classification
library(randomForest) # Breiman and Cutler's Random Forests for Classification and Regression
#-------------

# Exploratory Data Analysis
#-------------

# Age Plot
age_plot <- ggplot(df, aes(x = Age)) + 
  geom_histogram(color = "black", fill = "lightgreen") +
  geom_vline(aes(xintercept = mean(Age)), color = "red", linetype = "dashed", 
             linewidth = 1) +
  labs(title = "Age distribution")
print(age_plot)

# Sex Plot
df_sex <- df %>%
  group_by(Sex) %>%
  summarise(count = n()) %>%
  mutate(Freq = count / sum(count))
sex_bar <- ggplot(df_sex, aes(x = Sex, y = Freq, fill = Sex)) +
  geom_bar(stat = "identity") +
  labs(title = "Gender distribution", y = "Frequency")
print(sex_bar)

# Chest Pain Type Distribution
chest_pain_bar <- ggplot(df, aes(x = ChestPainType, fill = ChestPainType)) +
  geom_bar() +
  facet_grid(~Sex) +
  labs(title = "Chest Pain Type Distribution")
print(chest_pain_bar)

# Heart Disease percentage among both genders
df_hd <- data.frame(sex=c("M", "F"), 
                    hd=c(460, 50),
                    total=c(725, 193))
df_hd <- df_hd %>%
  mutate(freq=(hd/total)*100)
df_hd_bar <- ggplot(df_hd, aes(x="", y=freq, fill=hd))+
  geom_bar(width = 1, stat = "identity") +
  labs(title="Percentage of heart diseases among both sexes") +
  facet_grid(~ sex)
print(df_hd_bar)

# Correlation between numeric values
df2 <- df[c(1, 4, 5, 6, 8, 10, 12)]
corr <- cor(round(df2, 2))
corr_chart <- ggcorrplot(corr, hc.order = TRUE,
                         lab = FALSE)
print(corr_chart)

# Age and Heart Disease
hd_vs_age <- ggplot(df, aes(x = Age, y = HeartDisease, color = Sex)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  labs(title = "Age and Heart Disease")
print(hd_vs_age)

# Heart Disease and Maximum Heart Rate
hd_vs_maxhr <- ggplot(df, aes(x = MaxHR, y = HeartDisease, color = Sex)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  labs(title = "Heart Disease and Maximum Heart Rate")
print(hd_vs_maxhr)

# Heart Disease and Cholesterol
hd_vs_cholesterol <- ggplot(df, aes(x = Cholesterol, y = HeartDisease, 
                                    color = Sex)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  labs(title = "Heart Disease and Cholesterol")
print(hd_vs_cholesterol)

# Heart Disease and Sugar Blood Level
hd_vs_sugar <- ggplot(df, aes(x = FastingBS, y = HeartDisease, color = Sex)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  labs(title = "Heart Disease and Sugar Blood Level")
print(hd_vs_sugar)
#-------------

# Data Pre-processing
#-------------

# Factors
df$Sex <- as.factor (df$Sex)
df$ChestPainType <- as.factor (df$ChestPainType)
df$RestingECG <- as.factor (df$RestingECG)
df$ExerciseAngina <- as.factor (df$ExerciseAngina)
df$ST_Slope <- as.factor (df$ST_Slope)
df$HeartDisease <- as.factor(df$HeartDisease)

# Numeric
df$Age <- as.numeric (df$Age)
df$RestingBP <- as.numeric (df$RestingBP)
df$Cholesterol <- as.numeric (df$Cholesterol)
df$FastingBS <- as.numeric (df$FastingBS)
df$MaxHR <- as.numeric (df$MaxHR)
df$Oldpeak <- as.numeric (df$Oldpeak)

# Missing values
plot_missing(df)

# Patients with/without Heart Disease
heartdisease_count_plot <- ggplot(df, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar() +
  labs(title = "Patients with/without heart disease",
       x = "Heart Disease",
       y = "Count")
print(heartdisease_count_plot)
#-------------

# Model Implementation
#-------------

# Training and Test Dataset using Startified Sampling Method
set.seed(123)
split = sample.split(df, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Baseline Implementation for Logistic Regression
logistic_model <- glm(HeartDisease ~ ., data = training_set, family = "binomial")
summary(logistic_model)

# Baseline Perfomance for Logistic Regression

# Make predictions on the training set
train_predictions <- predict(logistic_model, newdata = training_set, type = "response")
train_binary_predictions <- ifelse(train_predictions > 0.5, 1, 0)

# Evaluate training performance
train_confusion_matrix <- table(training_set$HeartDisease, train_binary_predictions)
train_accuracy <- sum(diag(train_confusion_matrix)) / sum(train_confusion_matrix)
train_precision <- train_confusion_matrix[2, 2] / sum(train_confusion_matrix[, 2])
train_recall <- train_confusion_matrix[2, 2] / sum(train_confusion_matrix[2, ])
train_f1 <- 2 * (train_precision * train_recall) / (train_precision + train_recall)
train_roc_curve <- roc(training_set$HeartDisease, train_predictions)
train_auc <- auc(train_roc_curve)

# Display the values
cat("\nTraining Confusion Matrix:")
print(train_confusion_matrix)
cat("\nTraining Accuracy:", train_accuracy)
cat("\nTraining Precision:", train_precision)
cat("\nTraining Recall:", train_recall)
cat("\nTraining F1 Score:", train_f1, "\n\n")
cat("\nTraining AUC:", train_auc)
confusionMatrix(train_confusion_matrix)



# Make predictions on the test set
test_predictions <- predict(logistic_model, newdata = test_set, type = "response")
test_binary_predictions <- ifelse(test_predictions > 0.5, 1, 0)

# Evaluate test performance
test_confusion_matrix <- table(test_set$HeartDisease, test_binary_predictions)
test_accuracy <- sum(diag(test_confusion_matrix)) / sum(test_confusion_matrix)
test_precision <- test_confusion_matrix[2, 2] / sum(test_confusion_matrix[, 2])
test_recall <- test_confusion_matrix[2, 2] / sum(test_confusion_matrix[2, ])
test_f1 <- 2 * (test_precision * test_recall) / (test_precision + test_recall)
test_roc_curve <- roc(test_set$HeartDisease, test_predictions)

test_auc <- auc(test_roc_curve)

# Display the values
cat("\nTest Confusion Matrix:")
print(test_confusion_matrix)
cat("\nTest Accuracy:", test_accuracy)
cat("\nTest Precision:", test_precision)
cat("\nTest Recall:", test_recall)
cat("\nTest F1 Score:", test_f1, "\n\n")
cat("\nTest AUC:", test_auc)
confusionMatrix(test_confusion_matrix)


# Tuning Implementation for Logistic Regression
x_train <- model.matrix(HeartDisease ~ ., data = training_set)[,-1]
y_train <- training_set$HeartDisease
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, type.measure = "class")
cat("Optimal Lambda:", cv_model$lambda.min, "\n")
coef(cv_model, s = "lambda.min")
tuned_logistic_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = cv_model$lambda.min)


# Tuning Perfomance for Logistic Regression
tuned_train_predictions <- predict(tuned_logistic_model, newx = x_train, type = "response")
tuned_train_binary_predictions <- ifelse(tuned_train_predictions > 0.5, 1, 0)

tuned_train_confusion_matrix <- table(y_train, tuned_train_binary_predictions)
tuned_train_accuracy <- sum(diag(tuned_train_confusion_matrix)) / sum(tuned_train_confusion_matrix)
tuned_train_precision <- tuned_train_confusion_matrix[2, 2] / sum(tuned_train_confusion_matrix[, 2])
tuned_train_recall <- tuned_train_confusion_matrix[2, 2] / sum(tuned_train_confusion_matrix[2, ])
tuned_train_f1 <- 2 * (tuned_train_precision * tuned_train_recall) / (tuned_train_precision + tuned_train_recall)
tuned_train_roc_curve <- roc(y_train, tuned_train_predictions)
tuned_train_auc <- auc(tuned_train_roc_curve)

print("Tuned Training Confusion Matrix:")
print(tuned_train_confusion_matrix)
cat("Tuned Training Accuracy:", tuned_train_accuracy)
cat("\nTuned Training Precision:", tuned_train_precision)
cat("\nTuned Training Recall:", tuned_train_recall)
cat("\nTuned Training F1 Score:", tuned_train_f1, "\n")
cat("\nTuned Training AUC:", tuned_train_auc)
confusionMatrix(tuned_train_confusion_matrix)


# Test set
x_test <- model.matrix(HeartDisease ~ ., data = test_set)[,-1]
tuned_test_predictions <- predict(tuned_logistic_model, newx = x_test, type = "response")
tuned_test_binary_predictions <- ifelse(tuned_test_predictions > 0.5, 1, 0)

tuned_test_confusion_matrix <- table(test_set$HeartDisease, tuned_test_binary_predictions)
tuned_test_accuracy <- sum(diag(tuned_test_confusion_matrix)) / sum(tuned_test_confusion_matrix)
tuned_test_precision <- tuned_test_confusion_matrix[2, 2] / sum(tuned_test_confusion_matrix[, 2])
tuned_test_recall <- tuned_test_confusion_matrix[2, 2] / sum(tuned_test_confusion_matrix[2, ])
tuned_test_f1 <- 2 * (tuned_test_precision * tuned_test_recall) / (tuned_test_precision + tuned_test_recall)
tuned_test_roc_curve <- roc(test_set$HeartDisease, tuned_test_predictions)
tuned_test_auc <- auc(tuned_test_roc_curve)

print("\nTuned Test Confusion Matrix:")
print(tuned_test_confusion_matrix)
cat("Tuned Test Accuracy:", tuned_test_accuracy)
cat("\nTuned Test Precision:", tuned_test_precision)
cat("\nTuned Test Recall:", tuned_test_recall)
cat("\nTuned Test F1 Score:", tuned_test_f1, "\n")
cat("\nTuned Test AUC:", tuned_test_auc)
confusionMatrix(tuned_test_confusion_matrix)


# Baseline Implementation for Decision Tree
decision_tree_model <- rpart(HeartDisease ~ ., data = training_set, method = "class")
summary(decision_tree_model)

# Baseline Performance for Decision Tree

# Training Set
train_tree_predictions <- predict(decision_tree_model, newdata = training_set, type = "class")

train_tree_confusion_matrix <- table(training_set$HeartDisease, train_tree_predictions)
train_tree_accuracy <- sum(diag(train_tree_confusion_matrix)) / sum(train_tree_confusion_matrix)
train_tree_precision <- confusionMatrix(train_tree_confusion_matrix)$byClass["Pos Pred Value"]
train_tree_recall <- confusionMatrix(train_tree_confusion_matrix)$byClass["Sensitivity"]
train_tree_f1 <- confusionMatrix(train_tree_confusion_matrix)$byClass["F1"]
train_tree_roc_curve <- roc(as.factor(training_set$HeartDisease), as.numeric(train_tree_predictions))
train_tree_auc <- auc(train_tree_roc_curve)

cat("Training Confusion Matrix:\n")
print(train_tree_confusion_matrix)
cat("\nTraining Accuracy:", train_tree_accuracy)
cat("\nTraining Precision:", train_tree_precision)
cat("\nTraining Recall:", train_tree_recall)
cat("\nTraining F1 Score:", train_tree_f1)
cat("\nTraining AUC:", train_tree_auc)
confusionMatrix(train_tree_confusion_matrix)


# Test Set
test_tree_predictions <- predict(decision_tree_model, newdata = test_set, type = "class")

test_tree_confusion_matrix <- table(test_set$HeartDisease, test_tree_predictions)
test_tree_accuracy <- sum(diag(test_tree_confusion_matrix)) / sum(test_tree_confusion_matrix)
test_tree_precision <- confusionMatrix(test_tree_confusion_matrix)$byClass["Pos Pred Value"]
test_tree_recall <- confusionMatrix(test_tree_confusion_matrix)$byClass["Sensitivity"]
test_tree_f1 <- confusionMatrix(test_tree_confusion_matrix)$byClass["F1"]
test_tree_roc_curve <- roc(as.factor(test_set$HeartDisease), as.numeric(test_tree_predictions))
test_tree_auc <- auc(test_tree_roc_curve)

cat("Test Confusion Matrix:\n")
print(test_tree_confusion_matrix)
cat("\nTest Accuracy:", test_tree_accuracy)
cat("\nTest Precision:", test_tree_precision)
cat("\nTest Recall:", test_tree_recall)
cat("\nTest F1 Score:", test_tree_f1)
cat("\nTest AUC:", test_tree_auc)
confusionMatrix(test_tree_confusion_matrix)


# Tuning Implementation
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

grid <- expand.grid(
  cp = seq(0.01, 0.1, by = 0.01)
)

tuned_tree_model <- train(
  HeartDisease ~ .,
  data = training_set,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "ROC"
)

print(tuned_tree_model)
plot(tuned_tree_model)

str(df)

levels(training_set$HeartDisease) <- c("NoDisease", "HeartDisease")
levels(test_set$HeartDisease) <- c("NoDisease", "HeartDisease")
levels(df$HeartDisease) <- c("NoDisease", "HeartDisease")


# Tuning Performance

# Training set
tuned_train_tree_predictions <- predict(tuned_tree_model, newdata = training_set, type = "raw")

tuned_train_tree_confusion_matrix <- confusionMatrix(as.factor(tuned_train_tree_predictions), as.factor(training_set$HeartDisease))
tuned_train_tree_accuracy <- tuned_train_tree_confusion_matrix$overall["Accuracy"]
tuned_train_tree_precision <- tuned_train_tree_confusion_matrix$byClass["Pos Pred Value"]
tuned_train_tree_recall <- tuned_train_tree_confusion_matrix$byClass["Sensitivity"]
tuned_train_tree_f1 <- tuned_train_tree_confusion_matrix$byClass["F1"]
tuned_train_tree_roc_curve <- roc(training_set$HeartDisease, as.numeric(tuned_train_tree_predictions))
tuned_train_tree_auc <- auc(tuned_train_tree_roc_curve)

cat("Tuned Training Confusion Matrix:\n")
print(tuned_train_tree_confusion_matrix)
cat("\nTuned Training Accuracy:", tuned_train_tree_accuracy)
cat("\nTuned Training Precision:", tuned_train_tree_precision)
cat("\nTuned Training Recall:", tuned_train_tree_recall)
cat("\nTuned Training F1 Score:", tuned_train_tree_f1)
cat("\nTuned Training AUC:", tuned_train_tree_auc)


# Test set
tuned_test_tree_predictions <- predict(tuned_tree_model, newdata = test_set, type = "raw")

tuned_test_tree_confusion_matrix <- confusionMatrix(as.factor(tuned_test_tree_predictions), as.factor(test_set$HeartDisease))
tuned_test_tree_accuracy <- tuned_test_tree_confusion_matrix$overall["Accuracy"]
tuned_test_tree_precision <- tuned_test_tree_confusion_matrix$byClass["Pos Pred Value"]
tuned_test_tree_recall <- tuned_test_tree_confusion_matrix$byClass["Sensitivity"]
tuned_test_tree_f1 <- tuned_test_tree_confusion_matrix$byClass["F1"]
tuned_test_tree_roc_curve <- roc(test_set$HeartDisease, as.numeric(tuned_test_tree_predictions))
tuned_test_tree_auc <- auc(tuned_test_tree_roc_curve)

cat("Tuned Test Confusion Matrix:\n")
print(tuned_test_tree_confusion_matrix)
cat("\nTuned Test Accuracy:", tuned_test_tree_accuracy)
cat("\nTuned Test Precision:", tuned_test_tree_precision)
cat("\nTuned Test Recall:", tuned_test_tree_recall)
cat("\nTuned Test F1 Score:", tuned_test_tree_f1)
cat("\nTuned Test AUC:", tuned_test_tree_auc)


# Random Forest

# Baseline Implementation
rf_model <- randomForest(HeartDisease ~ ., data = training_set)
summary(rf_model)

# Baseline Performance

# Training Set
rf_train_predictions <- predict(rf_model, newdata = training_set)
rf_train_confusion_matrix <- confusionMatrix(rf_train_predictions, training_set$HeartDisease)

rf_train_accuracy <- rf_train_confusion_matrix$overall["Accuracy"]
rf_train_precision <- rf_train_confusion_matrix$byClass["Pos Pred Value"]
rf_train_recall <- rf_train_confusion_matrix$byClass["Sensitivity"]
rf_train_f1 <- rf_train_confusion_matrix$byClass["F1"]
rf_train_auc <- roc(training_set$HeartDisease, as.numeric(rf_train_predictions))

cat("Random Forest Training Confusion Matrix:\n")
print(rf_train_confusion_matrix)
cat("\nTraining Accuracy:", rf_train_accuracy)
cat("\nTraining Precision:", rf_train_precision)
cat("\nTraining Recall:", rf_train_recall)
cat("\nTraining F1 Score:", rf_train_f1)
cat("\nTraining AUC:", auc(rf_train_auc))

# Test Set
rf_test_predictions <- predict(rf_model, newdata = test_set)
rf_test_confusion_matrix <- confusionMatrix(rf_test_predictions, test_set$HeartDisease)

rf_test_accuracy <- rf_test_confusion_matrix$overall["Accuracy"]
rf_test_precision <- rf_test_confusion_matrix$byClass["Pos Pred Value"]
rf_test_recall <- rf_test_confusion_matrix$byClass["Sensitivity"]
rf_test_f1 <- rf_test_confusion_matrix$byClass["F1"]
rf_test_auc <- roc(test_set$HeartDisease, as.numeric(rf_test_predictions))

cat("\n\nRandom Forest Test Confusion Matrix:\n")
print(rf_test_confusion_matrix)
cat("\nTest Accuracy:", rf_test_accuracy)
cat("\nTest Precision:", rf_test_precision)
cat("\nTest Recall:", rf_test_recall)
cat("\nTest F1 Score:", rf_test_f1)
cat("\nTest AUC:", auc(rf_test_auc))


# Tuning Implementation
control_random <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "random")
tunegrid <- expand.grid(.mtry = seq(1, 15))
rf_random <- train(HeartDisease ~ ., data = training_set, 
                   method = "rf", 
                   metric = "Accuracy", 
                   tuneLength = 15, 
                   trControl = control_random)


# Tuning Performance

# Training Set
rf_random_train_predictions <- predict(rf_random, newdata = training_set, type = "raw")
rf_random_train_confusion_matrix <- confusionMatrix(as.factor(rf_random_train_predictions), as.factor(training_set$HeartDisease))
rf_random_train_accuracy <- rf_random_train_confusion_matrix$overall["Accuracy"]
rf_random_train_precision <- rf_random_train_confusion_matrix$byClass["Pos Pred Value"]
rf_random_train_recall <- rf_random_train_confusion_matrix$byClass["Sensitivity"]
rf_random_train_f1 <- rf_random_train_confusion_matrix$byClass["F1"]
rf_random_train_roc_curve <- roc(training_set$HeartDisease, as.numeric(rf_random_train_predictions))
rf_random_train_auc <- auc(rf_random_train_roc_curve)

cat("Random Search Tuned Training Confusion Matrix:\n")
print(rf_random_train_confusion_matrix)
cat("\nRandom Search Tuned Training Accuracy:", rf_random_train_accuracy)
cat("\nRandom Search Tuned Training Precision:", rf_random_train_precision)
cat("\nRandom Search Tuned Training Recall:", rf_random_train_recall)
cat("\nRandom Search Tuned Training F1 Score:", rf_random_train_f1)
cat("\nRandom Search Tuned Training AUC:", rf_random_train_auc)

# Test Set
rf_random_test_predictions <- predict(rf_random, newdata = test_set, type = "raw")
rf_random_test_confusion_matrix <- confusionMatrix(as.factor(rf_random_test_predictions), as.factor(test_set$HeartDisease))
rf_random_test_accuracy <- rf_random_test_confusion_matrix$overall["Accuracy"]
rf_random_test_precision <- rf_random_test_confusion_matrix$byClass["Pos Pred Value"]
rf_random_test_recall <- rf_random_test_confusion_matrix$byClass["Sensitivity"]
rf_random_test_f1 <- rf_random_test_confusion_matrix$byClass["F1"]
rf_random_test_roc_curve <- roc(test_set$HeartDisease, as.numeric(rf_random_test_predictions))
rf_random_test_auc <- auc(rf_random_test_roc_curve)

cat("\nRandom Search Tuned Test Confusion Matrix:\n")
print(rf_random_test_confusion_matrix)
cat("\nRandom Search Tuned Test Accuracy:", rf_random_test_accuracy)
cat("\nRandom Search Tuned Test Precision:", rf_random_test_precision)
cat("\nRandom Search Tuned Test Recall:", rf_random_test_recall)
cat("\nRandom Search Tuned Test F1 Score:", rf_random_test_f1)
cat("\nRandom Search Tuned Test AUC:", rf_random_test_auc)


# Analysis and Recommendations
# Display confusion matrix
confusionMatrix(train_tree_predictions, training_set$Class)
# Display ROC curve
plot(roc(training_set$Class, as.numeric(train_tree_predictions_prob)), main = "ROC Curve")

#-------------
