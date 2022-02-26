# ========================================================================================================
# Name: Arjun Aggarwal
# BC3409 Individual Assignment
# Results: 
#         1) Logistic Regression (Accuracy: 94.64 %)
#         2) Decision Tree - CART (Accuracy: 98.73 %)
#         3) Random Forest (Accuracy: 99.42 %)
#         4) XGBoost (Accuracy: 99.51 % )
#         5) Neural Network MLP (Accuracy: 98.64 %)         
# ========================================================================================================

# Import data
library(data.table)
setwd("~/Desktop")
dt_full = read.csv("Credit Card Data.csv", stringsAsFactors = TRUE)

# Data Wrangling

# Check if there is any missing data then remove from dataset
summary(dt_full)
dt <- dt_full[complete.cases(dt_full),] # There are no missing data in the dataset

summary(dt)
library("dplyr")
dt <- dt %>%
  filter(dt$age > 0)
# Minimium value of Age is a negative number, thus I removed the 3 records which had age as negative from the dataset

# Confirm that there is no missing data and the data makes sense
summary(dt)

# Identify and remove outliers
boxplot(dt$income)
boxplot(dt$age)
boxplot(dt$loan)
# There are no outliers in the dataset

# Make default into a categorical variable
dt$default <- as.factor(dt$default)
str(dt)

# Check collinearity of independent variables
cor(dt[,c('income', 'age', 'loan')])
# Correlation between the independent variables is not high and the problem of multi collinearity does not exist.

# Data Normalisation

# Some feature values differ from others substantially (e.g. Income is about 1000 larger than age). 
# The features with higher values will dominate the learning process. Hence, normalization is required.

# Create a function for normalisation
normalise <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalise all the variables except for 'Default' (which will not be used in the model)
dt[,c(1,2,3)] <- lapply(dt[,c(1,2,3)], normalise)
summary(dt)

# Train Test Split
library(caTools)

# Parameters for Train Test Split
set.seed(2021)
train_test_split = sample.split(dt$default, SplitRatio = 0.3)
test = dt[train_test_split,]
train = dt[!train_test_split,]

# Create confusion matrix function to make it easier to see the accuracy
ConfusionMatrix <- function(pred, Actual, prob){
  if (prob){
    cm <- table(Actual, ifelse(pred > 0.5,"Yes", "No"))
    print(prop.table(cm))
  } else{
    cm <- table(Actual, ifelse(pred == 1,"Yes", "No"))
    print(prop.table(cm))
  }
  print(cm)
  acc <- round((cm[1,1]+cm[2,2])/length(pred) * 100, digits  = 2)
  cat("Accuracy is",acc,"%")
}

# ========================================================================================================
# Logistic Regression
#
# Accuracy of model on test set: 94.64 %
#
# Assumptions:
# 1) The standard deviations of the errors should be constant (i.e. follow a normal distribution).
# 2) The independent variables should not be correlated with each other (i.e. no multicollinearity).
# 3) The mean of the errors should be zero.
#
# ========================================================================================================

#Building logistic model
logistic_model <- glm(default ~ income + age + loan, data = train, family = binomial)
summary(logistic_model)

# Predict 'default' using Logistic Regression on train set
pred_train_logistic <- predict(logistic_model, train, type = "response")

# Confusion Matrix for Logistic Regression on train set
ConfusionMatrix(pred_train_logistic, train$default, TRUE)

# actual         No            Yes
#      0   0.46580484    0.03377815
#      1   0.01542952    0.48498749

# actual     No    Yes
#      0   1117     81
#      1     37    1163
#
# Accuracy of model on train set is 95.08 %.

# Predict 'default' using Logistic Regression on test set
pred_logistic <- predict(logistic_model, test, type = "response")

# Confusion Matrix for Logistic Regression on test set
ConfusionMatrix(pred_logistic, test$default, TRUE)

# actual         No            Yes
#      0   0.46251217    0.03700097
#      1   0.01655307    0.48393379

# actual     No    Yes
#      0    475     38
#      1     17     497
#
# Accuracy of model on test set is 94.64 %.

# ========================================================================================================

# Decision Tree (CART)
#
# Accuracy of model on test set: 98.73 %
#
# Hyperparameters: minsplit = 15, cp = cp_opt, all other parameters set as default
#
# Assumptions:
# 1) No linearity is required.
# 2) All observations are assumed to fit into one of the nodes after splitting.
#
# ========================================================================================================
library(rpart)
library(rpart.plot)

# Grow the tree to the maximum
cart_model_max <- rpart(default ~ income + age + loan, data = train, method = "class", control = rpart.control(minsplit = 15, cp = 0))
rpart.plot(cart_model_max, nn= T, main = "Maximal Tree for Default")
printcp(cart_model_max)
plotcp(cart_model_max)

# Optimize the Complexity Parameter (CP)
cp_opt <- cart_model_max$cptable[which.min(cart_model_max$cptable[,"xerror"]),"CP"]

# Prune the tree
cart_model <- prune(cart_model_max, cp = cp_opt)
rpart.plot(cart_model_max, nn= T, main = "Optimal Tree for Default")

# Predict 'default' using the CART model on train set
pred_train_cart <- predict(cart_model, train, type = "class")
pred_train_cart <-as.numeric(as.character(pred_train_cart))

# Confusion Matrix for CART model on train set
ConfusionMatrix(pred_train_cart, train$default, FALSE)

# Actual             No           Yes
#      0    0.496246872   0.003336113
#      1    0.002919099   0.497497915

# Actual      No     Yes
#      0    1190       8
#      1       7     1193
# Accuracy is 99.37 %.

# Predict 'default' using the CART model on test set
pred_cart <- predict(cart_model, test, type = "class")
pred_cart <-as.numeric(as.character(pred_cart))

# Confusion Matrix for CART model on test set
ConfusionMatrix(pred_cart, test$default, FALSE)

# Actual             No           Yes
#      0    0.491723466   0.007789679
#      1    0.004868549   0.495618306

# Actual      No     Yes
#      0     505       8
#      1       5      509
# Accuracy is 98.73 %.

# ========================================================================================================
# Random Forest
#
# Accuracy of model on test set: 99.42 %
#
# Hyperparameters: ntree = 500, mtry = 3, all other parameters set as default
# 
# Assumptions:
# 1) There is no over-fitting of model generated by Random Forest.
#
# ========================================================================================================


library(randomForest)

# Building the Random Forest model
randomforest_model <- randomForest(default ~ income + age + loan, data = train, ntree = 500, mtry = 3, importance = T)

# Predict 'default' using the Random Forest model on train set
randomforest_train_pred <- predict(randomforest_model, train)

# Confusion Matrix for Random Forest model on train set
ConfusionMatrix(randomforest_train_pred, train$default, FALSE)

# Actual            No           Yes
#      0     0.8585786     0.0000000
#      1     0.0000000     0.1414214

# Actual    No   Yes
#      0  1196     0
#      1     0   197
# Accuracy is 100 %.

# Predict 'default' using the Random Forest model on test set
randomforest_pred <- predict(randomforest_model, test)

# Confusion Matrix for Random Forest model on test set
ConfusionMatrix(randomforest_pred, test$default, FALSE)

# Actual            No           Yes
#      0   0.4946445959   0.0048685492
#      1   0.0009737098   0.4995131451

# Actual    No   Yes
#      0   508     5
#      1     1    513
# Accuracy is 99.42 %.



# ========================================================================================================
# XGBoost
#
# Accuracy of model on test set: 99.51 %
#
# Hyperparameters: nrounds=100, max.depth=8, nthread=3
# 
# Assumptions:
# 1) Data can be collected from a sample that does not follow a specific distribution. Being a non-parametric model, it can handle skewed and multi-modal data.
# 2) Encoded integer value for each variable has ordinal relation
# 3) Model is not affected by multicollinearity as the model will only select one of the correlated features at each split.


# ========================================================================================================

library(tidyverse)
library(caret)
library(xgboost)

trained<-model.matrix(default~ . , data=train)
train_label<- train[,4]
train_label<- as.numeric(train_label)-1

tested<-model.matrix(default~ . , data=test)
test_label<- test[,4]
test_label<- as.numeric(test_label)-1

model<-xgboost(data=trained, label=train_label, objective="binary:logistic",  
               nrounds=100, max.depth=8, nthread=3)

# Variable Importance
xgb.plot.importance(xgb.importance(model=model))

# Predict 'default' using the XGBoost model on train set
pred_train <- predict(model, trained)
pred_train<- as.integer(pred_train>0.5)

# Confusion Matrix for XGBoost model on train set
confusionMatrix(as.factor(pred_train), as.factor(train_label))


# Actual      No     Yes
#      0     1198       0
#      1       0      1200
# Accuracy is 100 %.

# Predict 'default' using the XGBoost model on test set
pred_test <- predict(model, tested)
pred_test<- as.integer(pred_test>0.5)

# Confusion Matrix for XGBoost model on test set
confusionMatrix(as.factor(pred_test), as.factor(test_label))


# Actual      No     Yes
#      0     509       1
#      1       4      513
# Accuracy is 99.51 %.



# ========================================================================================================
# Neural Network MLP
#
# Accuracy of model on test set: 98.64 %
#
# Hyperparameters: size = 3
# 
# Assumptions:
# 1) Neurons within the same layer do not interact or communicate to each other and neurons at consecutive layers are densely connected.
# 2) Artificial Neurons are arranged in layers, which are sequentially arranged.
# 3) Every inter-connected neural network has it’s own weight and biased associated with it


# ========================================================================================================

library(nnet)


model = nnet(default ~ ., data=train, size = 3)

# Predict 'default' using the Neural Network MLP model on test set
pred_train<- predict(model,train)

# Confusion Matrix for Neural Network MLP model on test set
ConfusionMatrix(pred_train, train$default, FALSE)

# Actual      No     Yes
#      0     1198     0
#      1       25     1175
# Accuracy is 98.96 %.


# Predict 'default' using the Neural Network MLP model on test set
pred_test<- predict(model,test)

# Confusion Matrix for Neural Network MLP model on test set
ConfusionMatrix(pred_test, test$default, FALSE)


# Actual      No     Yes
#      0     513      0
#      1      14      500
# Accuracy is 98.64 %.

# ========================================================================================================



