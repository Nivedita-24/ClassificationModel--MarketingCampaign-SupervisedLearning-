## Project Summary and Task Detail

## Importing Libraries
library(ggplot2)
library(reshape)
library(forecast)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(smotefamily)
library(FNN)
library(e1071)

##--------------------------------------------------------------------------------------------------------------------------------------
# Reading the file
bank <- read.csv("bank-additional-full.csv", sep =";", header = TRUE, dec =".",stringsAsFactors = TRUE)

## Data Dimension
dim(bank) # 41,188 customer's data

## Looking at the dataset
str(bank)
head(bank)

## Statistical Summary
summary(bank)

## The dataset is imbalanced as the subscriptions are around 12% of total clients reached out during telemarketing campaign.

##--------------------------------------------------------------------------------------------------------------------------------------
# Data Cleaning

## Step 1: Checking for missing values

cbind(
  lapply(
    lapply(bank, is.na)
    , sum)
)

bank$HasSubscribed <- 1*(bank$y =="yes") # Creating a new variable HasSubscrbed

# There are no missing values

## Checking the data for outliers
boxplot(bank$age)
boxplot(bank$emp.var.rate)
boxplot(bank$cons.price.idx)
boxplot(bank$cons.conf.idx)
boxplot(bank$euribor3m)
boxplot(bank$nr.employed)

## Removing outliers
outliers <- boxplot(bank$cons.conf.idx, plot=FALSE)$out
length(outliers)
## Age variables contains outliers, the maximum age is 98, it is possible to have this age in the dataset.
## There are 447 values as 26.9, which cannot be a manual error, therefore not removing these outliers from the data.

##--------------------------------------------------------------------------------------------------------------------------------------
## Step2 : Exploratory Data Analysis (EDA)

## Analyzing continuous variables 
bank.num.df <- bank[c(1,16:20,22)]
str(bank.num.df)

# HeatMap to see the correlation between the independent variables and the output variable

cor.mat<-round(cor(bank.num.df),2)
melted.cor.mat<-melt(cor.mat)

p<-ggplot(melted.cor.mat, aes(x = X1, y= X2, fill = value))+ geom_tile()+geom_text(aes(x=X1, y= X2, label = value))
p + ggtitle("Correlation heatmap for variables") + theme(plot.title = element_text(hjust = 0.5))+ 
  xlab("") + ylab("")

## Analyzing categorical variables

data.for.plot1 <- aggregate(bank$HasSubscribed, by = list(bank$job), FUN = mean)
names(data.for.plot1) <- c("job", "Subscriptions")
plot2 <- ggplot(data.for.plot1) + geom_bar(aes(x = reorder(job, Subscriptions), y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot2

## Client's that has subscribed the term deposit are mostly students or retired.

data.for.plot2 <- aggregate(bank$HasSubscribed, by = list(bank$marital), FUN = mean)
names(data.for.plot2) <- c("marital", "Subscriptions")
plot3 <- ggplot(data.for.plot2) + geom_bar(aes(x = reorder(marital, Subscriptions), y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot3

## Client's that has subscribed the term deposit are mostly singles.

data.for.plot3 <- aggregate(bank$HasSubscribed, by = list(bank$education), FUN = mean)
names(data.for.plot3) <- c("education", "Subscriptions")
plot4 <- ggplot(data.for.plot3) + geom_bar(aes(x = reorder(education, Subscriptions), y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot4

## Client's that has subscribed the term deposit are mostly illiterate, university degree.

data.for.plot4 <- aggregate(bank$HasSubscribed, by = list(bank$default), FUN = mean)
names(data.for.plot4) <- c("default", "Subscriptions")
plot5 <- ggplot(data.for.plot4) + geom_bar(aes(x = default, y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot5

## Client's that has subscribed the term deposit do not have a credit in default.

data.for.plot5 <- aggregate(bank$HasSubscribed, by = list(bank$housing), FUN = mean)
names(data.for.plot5) <- c("housing", "Subscriptions")
plot6 <- ggplot(data.for.plot5) + geom_bar(aes(x = reorder(housing, Subscriptions), y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot6

## Client's that has subscribed the term deposit have a housing looan.


data.for.plot6 <- aggregate(bank$HasSubscribed, by = list(bank$loan), FUN = mean)
names(data.for.plot6) <- c("loan", "Subscriptions")
plot7 <- ggplot(data.for.plot6) + geom_bar(aes(x = reorder(loan, Subscriptions), y = Subscriptions), stat = "identity", color ="cyan4", fill = "lightcyan2")
plot7

## Client's that has subscribed the term deposit do not have a personal loan.

##--------------------------------------------------------------------------------------------------------------------------------------
## Feature Selection | Feature Transformation | Data Pre-processing


## Removing features which are irrelevant for our analysis
# contact, month, day_of_week, duration, campaign, pdays, previous, poutcome

bank.df <- bank[-c(8:15,21)]
str(bank.df)
dim(bank.df)


## we shall transform the non-numerical labels of the categorical variables to numerical ones and convert them to integers

cols <- c("job", "marital", "education", "default","housing","loan")
bank.df[cols] <- lapply(bank.df[cols], as.integer)
summary(bank.df)

## Oversampling using SMOTE (Synthetic Minority Oversampling Technique) Technique to handle imbalanced data.
set.seed(12345)
smote_result<- SMOTE(bank.df[,-13],target = bank.df[,13] , K = 5, dup_size = 0)

data_oversampled = smote_result$data
colnames(data_oversampled)[13] = "HasSubscribed"
table(data_oversampled$HasSubscribed)
data_oversampled$HasSubscribed <-as.factor(data_oversampled$HasSubscribed)

summary(bank.df)
summary(data_oversampled)

##--------------------------------------------------------------------------------------------------------------------------------------
##  PARTITIONING THE DATA IN TRAINING, VALIDATION AND TESTING SET.

# randomly sampling 50% of the row IDs for training, 30% serve as validation and the remaining 20% as testing data.
set.seed(12345)

train.index<-sample(rownames(data_oversampled),dim(data_oversampled)[1]*0.5)
valid.index<-sample(setdiff(rownames(data_oversampled),train.index), dim(data_oversampled)[1]*0.3)
test.index<-sample(setdiff(rownames(data_oversampled),union(train.index,valid.index)))

train.df<-data_oversampled[train.index,]
valid.df<-data_oversampled[valid.index,]
test.df<-data_oversampled[test.index,]
dim(train.df)
dim(valid.df)
dim(test.df)
summary(train.df$HasSubscribed)
summary(valid.df$HasSubscribed)
summary(test.df$HasSubscribed)


##--------------------------------------------------------------------------------------------------------------------------------------
## Linear Algorithms
## Logistic Regression

# using glm() (general linear model) with family = "binomial" to fit a logistic regression.
banklogit.reg <- glm(HasSubscribed ~ ., data = train.df, family = "binomial") 
options(scipen=999)
summary(banklogit.reg)

## Age, marital status, education, credit defaulter, employment variation rate, consumer price index , consumer confidence index and no. of employees are significant variables in the model.

bank.reg.pred <- predict(banklogit.reg, test.df[,-13], type = "response") 
confusionMatrix(as.factor(ifelse(bank.reg.pred > 0.5, 1, 0)), as.factor(test.df$HasSubscribed))

## Calculating Lift of Final Model.
bankgain <- gains(as.numeric(test.df$HasSubscribed) , bank.reg.pred, groups=10)
bankgain

## From gains table, we can observe that first 5 deciles have 57% of cumulative percentage of
## total response for the subscription.

# plot lift chart
plot(c(0,bankgain$cume.pct.of.total*sum(as.numeric(test.df$HasSubscribed)))~c(0,bankgain$cume.obs), 
     xlab="# cases", ylab="Cumulative", type="l")
lines(c(0,sum(as.numeric(test.df$HasSubscribed)))~c(0, dim(test.df)[1]), lty=2)

# The “lift” over the base curve indicates for a given number of cases (read on the x-axis), the additional prospective customers that we can identify by using the model.

# compute deciles and plot decile-wise chart
heights <- bankgain$mean.resp/mean(as.numeric(test.df$HasSubscribed))
heights
decileplot <- barplot(heights, names.arg = bankgain$depth, ylim = c(0,4), 
                      xlab = "Percentile", ylab = "Mean Response/Overall Mean", main = "Decile-wise lift chart")

# add labels to columns
text(decileplot, heights+0.5, labels=round(heights, 1), cex = 0.8)

# Taking 10% of the records that are ranked by the model as “most probable 1’s” yields 1.3 times as many 1’s as would simply selecting 10% of the records at random. 

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.logreg <- as.factor(as.factor(ifelse(bank.reg.pred > 0.5, 1, 0))) # factor of predictions

precision <- posPredValue(predictions.logreg, y, positive="1")
recall <- sensitivity(predictions.logreg, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

# Precision  = True positive / (True Positive + False Positive)
# Recall = True positive / (True Positive + False Negative)
## F1 Score = 2*(precision*recall)/(precision + recall)
## Accuracy is 73%, Precision is 72%,  Recall is 69% and F1 score is 70.38%

## Here our agenda is to have maximum recall, low precision is acceptable.

##--------------------------------------------------------------------------------------------------------------------------------------
## Non-Linear Algorithms
## K Nearest Neighbors

# initialize normalized training, validation data, complete data frames to originals
train.norm.df <- train.df
valid.norm.df <- valid.df
test.norm.df <-test.df

# use preProcess() from the caret package to normalize the variables
norm.values <- preProcess(train.df[, 1:12], method=c("center", "scale"))
train.norm.df[, 1:12] <- predict(norm.values, train.df[, 1:12])
valid.norm.df[, 1:12] <- predict(norm.values, valid.df[, 1:12])
test.norm.df[, 1:12] <- predict(norm.values, test.df[, 1:12])

# Choosing the best value for 'k'
accuracy.df <- data.frame( k = seq(1,20,1), accuracy = rep(0,20))
accuracy.df

for (i in 1:20){
  knn.pred <-knn(train = train.norm.df[,1:12],test = valid.norm.df[,1:12], cl=train.norm.df[,13]
                 , k=i)
  accuracy.df[i,2]<- confusionMatrix(factor(knn.pred), factor(valid.norm.df[,13]))$overall[1]
}

#Accuracy plot
plot(accuracy.df, type="b", xlab="K- Value",ylab="Accuracy level")

## accuracy is best at k=1; 87% on validation set.
## But We would recommend to take k=3 to avoid over fitting.

knn.pred.test <- knn(train.norm.df[,1:12],test.norm.df[,1:12], cl=train.norm.df[,13]
                     , k=5)
confusionMatrix(factor(knn.pred.test), factor(test.norm.df[,13]))

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.knn <- as.factor(knn.pred.test) # factor of predictions

precision <- posPredValue(predictions.knn, y, positive="1")
recall <- sensitivity(predictions.knn, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

## Accuracy is 83%, Precision is 82.8%, Recall is 82% and F1 score is 82%

##--------------------------------------------------------------------------------------------------------------------------------------
## CART (Classification and Regression Trees)

# Running a full grown classification tree
set.seed(7)
default_tree <- rpart(HasSubscribed ~., data=train.df, method='class')

# using prp to draw this tree
prp(default_tree, type = 1, extra = 1, split.font = 1, varlen = -10)

# Checking the accuracy on test data
bank.ct.predict <- predict(default_tree, test.df, type='class')
confusionMatrix(bank.ct.predict, as.factor(test.df$HasSubscribed))

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.cart <- as.factor(bank.ct.predict) # factor of predictions

precision <- posPredValue(predictions.cart, y, positive="1")
recall <- sensitivity(predictions.cart, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

## Accuracy is 83%, Precision is 83.5%, Recall is 76.6% and F1 score is 79.9%

##--------------------------------------------------------------------------------------------------------------------------------------
## Advnaced Algorithms

# BOOSTED TREES MODEL

set.seed(1)

bankboost <- boosting(HasSubscribed ~ ., data = train.df)

#Predict using test data

bankBoost.pred.test <- predict(bankboost,test.df,type = "class")

# generate confusion matrix for testing data
confusionMatrix(as.factor(bankBoost.pred.test$class), as.factor(test.df$HasSubscribed))

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.boost <- as.factor(bankBoost.pred.test$class) # factor of predictions

precision <- posPredValue(predictions.boost, y, positive="1")
recall <- sensitivity(predictions.boost, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

# Accuracy in prediction is 89.84 %, precision is 91%, recall is 86.87% and F1 score is 88%.

####--------------------------------------------------------------------------------------------------------------------------------------

## RANDOM FORESTS

bankrf <- randomForest(as.factor(HasSubscribed) ~ ., data = train.df, ntree = 500, 
                        mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(bankrf, type = 1)

# OBSERVATION
# The most important variables, in identifying whether a person would subscribe are age, job, education, marital status, jousing loan, personal loan etc.

## confusion matrix of the random forest above, use the test data

bankrf.pred.test <- predict(bankrf,test.df,type = "class")

# generate confusion matrix
confusionMatrix(bankrf.pred.test, as.factor(test.df$HasSubscribed))

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.rf <- as.factor(bankrf.pred.test) # factor of predictions

precision <- posPredValue(predictions.rf, y, positive="1")
recall <- sensitivity(predictions.rf, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

# Accuracy in prediction is 91%, precision is 93%, recall is 88% and F1 score is 90% using Random forest trees Model.

##--------------------------------------------------------------------------------------------------------------------------------------

## SVM

train.df$HasSubscribed <- as.factor(train.df$HasSubscribed)
valid.df$HasSubscribed <- as.factor(valid.df$HasSubscribed)
test.df$HasSubscribed <- as.factor(test.df$HasSubscribed)

## feature scaling
train.df[-13] <- scale(train.df[-13])
valid.df[-13] <- scale(valid.df[-13])
test.df[-13] <- scale(test.df[-13])

## fitting the SVM classifier

## hyperparameters tuning for c, values tried 0.001,0.01,0.1,1,10 and Gamma 1,10,100

svm.model <- svm(HasSubscribed ~ ., data = train.df, kernel = 'radial', cost = 10, gamma = 1)
svm.pred <- predict(svm.model, valid.df[,-13])

confusionMatrix(factor(svm.pred), factor(valid.df[,13]))

## Precision, Recall and F1
y <- as.factor(test.df[,13]) # factor of positive / negative cases
predictions.svm <- as.factor(svm.pred) # factor of predictions

precision <- posPredValue(predictions.svm, y, positive="1")
recall <- sensitivity(predictions.svm, y, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

# Accuracy in prediction is 85%, precision is 78%, recall is 66% and F1 score is 71% using Random forest trees Model.


##----------------------------------------------------------------------------------------------




