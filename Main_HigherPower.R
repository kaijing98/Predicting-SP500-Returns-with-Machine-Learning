rm(list=ls())

library(forecast)
library(fpp)
library(ISLR)
library(glmnet)
library(nnet)
library(hdm)
library(gbm)
library("readxl")
library(HDeconometrics)
library(psych) #install.packages("psych")
#install.packages("githubinstall") #this package is needed to install packages from GitHub (a popular code repository)
library(githubinstall)
githubinstall("HDeconometrics")
#Install the HDeconometrics package used in Medeiros et al. (2019) for convenient estimation
#of LASSO and ElNet using information criteria (basically uses glmnet, and selects on criterion)
library(HDeconometrics)
library(randomForest)
library(caret) 
#install.packages('e1071', dependencies=TRUE)

setwd("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/Project/")
source("func-ann.R")
source("func-lasso.R")
source("func-elastic.R")
source("func-ridge.R")
source("unconditional_mean.R")
source("func-boostedtree.R")
source("func-boostedtree-cat.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-rf.R")
source("func-rf-cat.R")
source("func-boostedtree.R")
source("unconditional_mean.R")
source("func-forecast.R")
source("func-boostedtree-cat.R")
source("func-rf-cat.R")

df <- read_excel("Data-SnP Returns-Final.xlsx", sheet = "Data")
df_cat <- df[,-2]
df <- df[,-1]

fb = formula(SP500R~poly(USCIPM,degree = 3,raw = TRUE)
             +poly(USPCE,degree = 3,raw = TRUE) + poly(USCPCE,degree = 3,raw = TRUE)
             +poly(USCINF,degree = 3,raw = TRUE) + poly(M1G,degree = 3,raw = TRUE)
             +poly(M2G,degree = 3,raw = TRUE) + poly(USCUN,degree = 3,raw = TRUE)
             +poly(USCFR,degree = 3,raw = TRUE) + poly(USCTBILL,degree = 3,raw = TRUE)
             +poly(EXRATE,degree = 3,raw = TRUE) + poly(DEBT,degree = 3,raw = TRUE)
             +poly(VAR,degree = 3,raw = TRUE) + poly(PE,degree = 3,raw = TRUE)
             +poly(DY,degree = 3,raw = TRUE))

#Assign x and y of the model 
x = model.matrix(fb, data = df)
SP500R = df$SP500R
Y = cbind(SP500R, x[,2:43])

#get variable we want to predict
yy = df$SP500R
nprev = 60 #data between 2015 to 2019 (test window)

Y_cat  = data.matrix(df)[,-2]
#get variable we want to predict
yy_cat = df_cat$SP500RC

#actual result for direction
actual_direction = c(tail(yy_cat,nprev))

###############################################################################
## Artificial Neural Network (ANN)
###############################################################################
set.seed(12345)
#Run forecasts for ANN (BIC)
ann1a=ann.rolling.window(Y,nprev,1,1)
ann3a=ann.rolling.window(Y,nprev,1,3)
ann6a=ann.rolling.window(Y,nprev,1,6)
ann12a=ann.rolling.window(Y,nprev,1,12)

#RIDGE(BIC) RMSE's
ann.mse1=ann1a$errors[1]
ann.mse1
ann.mse3=ann3a$errors[1]
ann.mse3
ann.mse6=ann6a$errors[1]
ann.mse6
ann.mse12=ann12a$errors[1]
ann.mse12

pred1a.ann = forecast(ann1a$model, tail(Y, 60), 1, 1, 1)
pred1a.ann
pred3a.ann = forecast(ann3a$model, tail(Y, 60), 1, 3, 1)
pred3a.ann
pred6a.ann = forecast(ann6a$model, tail(Y, 60), 1, 6, 1)
pred6a.ann
pred12a.ann = forecast(ann12a$model, tail(Y, 60), 1, 12, 1)
pred12a.ann

##########################################################
#standard helper funcs
#########################################################
#Auxiliary function to compute root MSE (same as MSE before, but with square root):
RMSE <- function(pred, truth){ #start and end body of the function by { } - same as a loop
  return(sqrt(mean((truth - pred)^2)))
} #end function with a return(output) statement. Here we can go straight to return because the object of interest is a simple function of inputs

### Accuracy function that takes in the prediction vector and truth vector, outputs the accuracy
calc_accuracy <- function(pred, truth) {
  num_correct = 0
  pred_actual = cbind(pred, truth)
  for (row in 1:nrow(pred_actual)) {
    if (pred_actual[row, 1] == pred_actual[row,2]) {
      num_correct = num_correct + 1
    }
  }
  accuracy = num_correct/60
  return(accuracy)
}

get_direction <- function(pred, direction) {
  direction_pred = matrix(NA,nprev, 1) 
  for (row in 1:nrow(pred)) {
    if (pred[row, 1] >= 0) {
      direction_pred[row, 1] = 1
    } else {
      direction_pred[row, 1] = 0
    }
  }
  return(direction_pred)
}

###############################################################################
## Lasso
###############################################################################
set.seed(12345)
alpha=1 #set alpha=1 for LASSO

#Run forecasts for LASSO (BIC)

lasso1a=lasso.rolling.window(Y,nprev,1,1,alpha,IC="bic", "gaussian")
lasso3a=lasso.rolling.window(Y,nprev,1,3,alpha,IC="bic", "gaussian")
lasso6a=lasso.rolling.window(Y,nprev,1,6,alpha,IC="bic", "gaussian")
lasso12a=lasso.rolling.window(Y,nprev,1,12,alpha,IC="bic", "gaussian")

lassoa.mse1=lasso1a$errors[2]
lassoa.mse3=lasso3a$errors[2]
lassoa.mse6=lasso6a$errors[2]
lassoa.mse12=lasso12a$errors[2]

lassoa.mse1
lassoa.mse3
lassoa.mse6
lassoa.mse12

pred1a = forecast(lasso1a$model, tail(Y, 60), 1, 1)
pred1a
pred3a = forecast(lasso3a$model, tail(Y, 60), 1, 3)
pred3a
pred6a = forecast(lasso6a$model, tail(Y, 60), 1, 6)
pred6a
pred12a = forecast(lasso12a$model, tail(Y, 60), 1, 12)
pred12a

pols.lasso1a=pols.rolling.window(Y,nprev,1,1,lasso1a$coef)
pols.lasso3a=pols.rolling.window(Y,nprev,1,3,lasso3a$coef)
pols.lasso6a=pols.rolling.window(Y,nprev,1,6,lasso6a$coef)
pols.lasso12a=pols.rolling.window(Y,nprev,1,12,lasso12a$coef)

#Post-LASSO RMSE's:
plasso.mse1=pols.lasso1a$errors[2]
plasso.mse3=pols.lasso3a$errors[2]
plasso.mse6=pols.lasso6a$errors[2]
plasso.mse12=pols.lasso12a$errors[2]

plasso.mse1
plasso.mse3
plasso.mse6
plasso.mse12

pred1a.plasso = forecast(pols.lasso1a$model, tail(Y, 60), 1, 1, 2)
pred1a.plasso
pred3a.plasso = forecast(pols.lasso3a$model, tail(Y, 60), 1, 3, 2)
pred3a.plasso
pred6a.plasso = forecast(pols.lasso6a$model, tail(Y, 60), 1, 6, 2)
pred6a.plasso
pred12a.plasso = forecast(pols.lasso12a$model, tail(Y, 60), 1, 12, 2)
pred12a.plasso

###############################################################################
## Elastic Net
###############################################################################
alpha=0.5

elastic1a=elastic.rolling.window(Y,nprev,1,1,alpha,IC="bic", "gaussian")
elastic3a=elastic.rolling.window(Y,nprev,1,3,alpha,IC="bic",  "gaussian")
elastic6a=elastic.rolling.window(Y,nprev,1,6,alpha,IC="bic",  "gaussian")
elastic12a=elastic.rolling.window(Y,nprev,1,12,alpha,IC="bic",  "gaussian")

#LASSO(BIC) MSE's
elastic.mse1=elastic1a$errors[1]
elastic.mse3=elastic3a$errors[1]
elastic.mse6=elastic6a$errors[1]
elastic.mse12=elastic12a$errors[1]

elastic.mse1
elastic.mse3
elastic.mse6
elastic.mse12

pred1a.elastic = forecast(elastic1a$model, tail(Y, 60), 1, 1)
pred1a.elastic
pred3a.elastic = forecast(elastic3a$model, tail(Y, 60), 1, 3)
pred3a.elastic
pred6a.elastic = forecast(elastic6a$model, tail(Y, 60), 1, 6)
pred6a.elastic
pred12a.elastic = forecast(elastic12a$model, tail(Y, 60), 1, 12)
pred12a.elastic

###############################################################################
## Ridge-regression
###############################################################################
#See the functions  in func-ridge.R
alpha=0

#Run forecasts for Ridge (BIC)
#note in this case indice is set to 15 as SP500EP is the 15th column
# SP500EP is the 1st column
ridge1a=ridge.rolling.window(Y,nprev,1,1,alpha,IC="bic", "gaussian")
ridge3a=ridge.rolling.window(Y,nprev,1,3,alpha,IC="bic", "gaussian")
ridge6a=ridge.rolling.window(Y,nprev,1,6,alpha,IC="bic", "gaussian")
ridge12a=ridge.rolling.window(Y,nprev,1,12,alpha,IC="bic", "gaussian")

#RIDGE(BIC) RMSE's
ridge.mse1=ridge1a$errors[1]
ridge.mse1
ridge.mse3=ridge3a$errors[1]
ridge.mse3
ridge.mse6=ridge6a$errors[1]
ridge.mse6
ridge.mse12=ridge12a$errors[1]
ridge.mse12

pred1a.ridge = forecast(ridge1a$model, tail(Y, 60), 1, 1)
pred1a.ridge
pred3a.ridge = forecast(ridge3a$model, tail(Y, 60), 1, 3)
pred3a.ridge
pred6a.ridge = forecast(ridge6a$model, tail(Y, 60), 1, 6)
pred6a.ridge
pred12a.ridge = forecast(ridge12a$model, tail(Y, 60), 1, 12)
pred12a.ridge

###############################################################################
## Boosted Tree
###############################################################################
# SP500EP is the 1st column
set.seed(12345)
boosted1a=boosted.rolling.window(Y,nprev,1,1)
boosted3a=boosted.rolling.window(Y,nprev,1,3)
boosted6a=boosted.rolling.window(Y,nprev,1,6)
boosted12a=boosted.rolling.window(Y,nprev,1,12)

boosted.mse1=boosted1a$errors[1]
boosted.mse1
boosted.mse3=boosted3a$errors[1]
boosted.mse3
boosted.mse6=boosted6a$errors[1]
boosted.mse6
boosted.mse12=boosted12a$errors[1]
boosted.mse12

pred1a.boosted = forecast(boosted1a$model, tail(Y, 60), 1, 1, 2)
pred1a.boosted
pred3a.boosted = forecast(boosted3a$model, tail(Y, 60), 1, 3, 2)
pred3a.boosted
pred6a.boosted = forecast(boosted6a$model, tail(Y, 60), 1, 6, 2)
pred6a.boosted
pred12a.boosted = forecast(boosted12a$model, tail(Y, 60), 1, 12, 2)
pred12a.boosted

boosted1ac=boostedcat.rolling.window(Y_cat,nprev,1,1)
boosted3ac=boostedcat.rolling.window(Y_cat,nprev,1,3)
boosted6ac=boostedcat.rolling.window(Y_cat,nprev,1,6)
boosted12ac=boostedcat.rolling.window(Y_cat,nprev,1,12)

###############################################################################
## Random Forest - Rolling period
###############################################################################
source("/Users/adelineandikko/Desktop/Adeline NUS/20:21 Sem 1/EC4308/R/func-rf.R")

#Rolling window forecast
rf1a=rf.rolling.window(Y,nprev,1,1) # 1 step forecast
rf3a=rf.rolling.window(Y,nprev,1,3) # 3 step forecast
rf6a=rf.rolling.window(Y,nprev,1,6) # 6 step forecast
rf12a=rf.rolling.window(Y,nprev,1,12) #12 step forecast

#See the MSE:
rf.mse1=rf1a$errors[1] # MSE =22.8 for 1 lag
rf.mse1
rf.mse3=rf3a$errors[1] # MSE =20.1 for 3 lags
rf.mse3
rf.mse6=rf6a$errors[1] # MSE = 19.3 for 6 lags
rf.mse6
rf.mse12=rf12a$errors[1] # MSE = 20.7 for 12 lags
rf.mse12

pred1a.rf = forecast(rf1a$model, tail(Y, 60), 1, 1, 1)
pred1a.rf
pred3a.rf = forecast(rf3a$model, tail(Y, 60), 1, 3, 1)
pred3a.rf
pred6a.rf = forecast(rf6a$model, tail(Y, 60), 1, 6, 1)
pred6a.rf
pred12a.rf = forecast(rf12a$model, tail(Y, 60), 1, 12, 1)
pred12a.rf

rf1cc=rfcat.rolling.window(Y_cat,nprev,1,1) # 1 step forecast
rf3cc=rfcat.rolling.window(Y_cat,nprev,1,3) # 3 step forecast
rf6cc=rfcat.rolling.window(Y_cat,nprev,1,6) # 6 step forecast
rf12cc=rfcat.rolling.window(Y_cat,nprev,1,12) #12 step forecast

###############################################################################
###Accuracy on SP500 Directions
###############################################################################
#Lasso
lasso1a_dir = get_direction(lasso1a$pred)
lasso3a_dir = get_direction(lasso3a$pred)
lasso6a_dir = get_direction(lasso6a$pred)
lasso12a_dir = get_direction(lasso12a$pred)

calc_accuracy(lasso1a_dir, actual_direction)
calc_accuracy(lasso3a_dir, actual_direction)
calc_accuracy(lasso6a_dir, actual_direction)
calc_accuracy(lasso12a_dir, actual_direction)

#Elastic Net
elastic1a_dir = get_direction(elastic1a$pred)
elastic3a_dir = get_direction(elastic3a$pred)
elastic6a_dir = get_direction(elastic6a$pred)
elastic12a_dir = get_direction(elastic12a$pred)

calc_accuracy(elastic1a_dir, actual_direction)
calc_accuracy(elastic3a_dir, actual_direction)
calc_accuracy(elastic6a_dir, actual_direction)
calc_accuracy(elastic12a_dir, actual_direction)

#Ridge
ridge1a_dir = get_direction(ridge1a$pred)
ridge3a_dir = get_direction(ridge3a$pred)
ridge6a_dir = get_direction(ridge6a$pred)
ridge12a_dir = get_direction(ridge12a$pred)

calc_accuracy(ridge1a_dir, actual_direction)
calc_accuracy(ridge3a_dir, actual_direction)
calc_accuracy(ridge6a_dir, actual_direction)
calc_accuracy(ridge12a_dir, actual_direction)

ann1a_dir = get_direction(ann1a$pred)
ann3a_dir = get_direction(ann3a$pred)
ann6a_dir = get_direction(ann6a$pred)
ann12a_dir = get_direction(ann12a$pred)

calc_accuracy(ann1a_dir, actual_direction)
calc_accuracy(ann3a_dir, actual_direction)
calc_accuracy(ann6a_dir, actual_direction)
calc_accuracy(ann12a_dir, actual_direction)

boosted1a_dir = get_direction(boosted1ac$pred)
boosted3a_dir = get_direction(boosted3ac$pred)
boosted6a_dir = get_direction(boosted6ac$pred)
boosted12a_dir = get_direction(boosted12ac$pred)

calc_accuracy(boosted1a_dir, actual_direction) 
calc_accuracy(boosted3a_dir, actual_direction) 
calc_accuracy(boosted6a_dir, actual_direction) 
calc_accuracy(boosted12a_dir, actual_direction)

calc_accuracy(get_direction(rf1a$pred),actual_direction) 
calc_accuracy(get_direction(rf3a$pred),actual_direction) 
calc_accuracy(get_direction(rf6a$pred),actual_direction) 
calc_accuracy(get_direction(rf12a$pred),actual_direction) 

###############################################################################
## Rolling Unconditional Mean as the predictor
###############################################################################
set.seed(12345)
mean1=mean.rolling.window(Y,nprev,1,1)
mean3=mean.rolling.window(Y,nprev,1,3)
mean6=mean.rolling.window(Y,nprev,1,6)
mean12=mean.rolling.window(Y,nprev,1,12)

mean.mse1=mean1$errors[2]
mean.mse3=mean3$errors[2]
mean.mse6=mean6$errors[2]
mean.mse12=mean12$errors[2]

mean.mse1
mean.mse3
mean.mse6
mean.mse12

###############################################################################
###manual testing code below for 1 step forecast  TO BE DELETED B4 SUBMISSION!!!
###############################################################################
test.window = Y_cat[(1):420,]
aux=embed(test.window,4+1) #create 4 lags + forecast horizon shift (=lag option)
y=aux[,15] #  Y variable aligned/adjusted for missing data due do lags (in this case our y variable is in the 15th column)
X=aux[,-c(1:(ncol(test.window)*1))]   # lags of Y (predictors) corresponding to forecast horizon
X.out=tail(aux,1)[1:ncol(X)] #retrieve the last  observations if one-step forecast

#Here we use the glmnet wrapper written by the authors that does selection on IC:
model=ic.glmnet(X,y,crit="bic",alpha = 0, distType="logistic") #fit the ridge model selected on IC
length(model$coefficients) #length of coefficients 
pred_probs = predict(model,X.out, type="response")
if (pred_probs > 0.5) {
  pred = cbind(c(1))
} else {
  pred = cbind(c(0))
}

###############################################################################
## Regression Tree (Base Case)
###############################################################################
install.packages("rpart.plot")
setwd("D:/NUS Files JX/Y5S1/EC4308")
df <- read.csv("ProjectData.csv")

train = df[1:420,]
test = df[421:486,]

library(rpart)
library(tictoc)
library(gbm)
library(randomForest)
setwd("D:/NUS Files JX/Y5S1/EC4308")
df <- read.csv("ProjectData.csv")

set.seed(12345)

ntrain = 366 #data between 1980 to 2009
nvalidation = 60 #data between 2010 to 2014
ntest = 60 #data between 2015 to 2019
train = df[1:ntrain,]
vald = df[ntrain+1:nvalidation+ntrain+1,]
test = df[nrow(df)-(ntrain+nvalidation):nrow(df),]

#A bit more advanced MSE function: computes both MSE and its standard error via linear regression of squared errors on a constant
MSE <- function(pred, truth){ #start and end body of the function by { } - same as a loop
  return(summary(lm((truth-pred)^2~1))$coef[1:2])}

#Create model matrices
x1 = model.matrix(SP500EP~.-1, data = train) #training data, -1 to remove intercept
x2 = model.matrix(SP500EP~.-1, data = vald) #validation data
x3 = model.matrix(SP500EP~.-1, data = test) #test data
y1=train$SP500EP #y for training data
y2=vald$SP500EP  #y for validation data
y3=test$SP500EP  #y for test data

#Fit a single pruned tree first using default settings:
big.tree = rpart(SP500EP~.,method="anova",data=train) #big tree

bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"] #get best penalty

best.tree = prune(big.tree,cp=bestcp) #get tree for best cp on CV

tree.pred = predict(best.tree, newdata=test) #predict on test data
MSE(tree.pred,y3)
# MSE: 24.606120  SE: 4.448134

#Fit boosted tree, with d=5 (takes a long time!!):
tic()
boost.fit = gbm(SP500EP~.,data=train,distribution='gaussian',interaction.depth=5,n.trees=10000,shrinkage=.01,cv.folds=10)
bestd5cv=gbm.perf(boost.fit, method="cv")
boost.pred = predict(boost.fit,newdata = test,n.trees = bestd5cv)
toc()
MSE(boost.pred, y3)
# MSE: 13.504419  SE: 2.691189

#Fit boosted tree, with d=2 (takes a long time!!):
tic()
boost.fit2 = gbm(SP500EP~.,data=train,distribution='gaussian',interaction.depth=2,n.trees=10000,shrinkage=.01,cv.folds=10)
bestd5cv2=gbm.perf(boost.fit2, method="cv")
boost.pred2 = predict(boost.fit2,newdata = test,n.trees = bestd5cv2)
toc()
MSE(boost.pred2, y3)
# MSE: 19.611683  SE: 3.972983

#Fit a random forest using the tuneRF() function to pick the optimal number of predictors (attention - takes a long time!):
#(takes a long time!)
tic()
rftune=tuneRF(x1, y1, mtryStart=floor(sqrt(ncol(x1))), stepFactor=2, improve=0.05, nodesize=5, ntree=5000, doBest=TRUE, plot=FALSE, trace=FALSE)
toc()
rftune$mtry #Optimal number of mtry (i.e. number of variables available for splitting at each tree) is 6 (for OOB)

rft.pred = predict(rftune, newdata=x3)
MSE(rft.pred,y3)
# MSE: 4.1052612 SE: 0.826929

###############################################################################
## Random Forest - Rolling period
###############################################################################
set.seed(12345)
source("/Users/adelineandikko/Desktop/Adeline NUS/20:21 Sem 1/EC4308/R/func-rf.R")

#Rolling window forecast
rf1a=rf.rolling.window(Y,nprev,1,1) # 1 step forecast
rf3a=rf.rolling.window(Y,nprev,1,3) # 3 step forecast
rf6a=rf.rolling.window(Y,nprev,1,6) # 6 step forecast
rf12a=rf.rolling.window(Y,nprev,1,12) #12 step forecast

#See the MSE:
rf.mse1=rf1a$errors[1] # MSE =22.8 for 1 lag
rf.mse1
rf.mse3=rf3a$errors[1] # MSE =20.1 for 3 lags
rf.mse3
rf.mse6=rf6a$errors[1] # MSE = 19.3 for 6 lags
rf.mse6
rf.mse12=rf12a$errors[1] # MSE = 20.7 for 12 lags
rf.mse12

pred1a.rf = forecast(rf1a$model, tail(Y, 60), 1, 1, 1)
pred1a.rf
pred3a.rf = forecast(rf3a$model, tail(Y, 60), 1, 3, 1)
pred3a.rf
pred6a.rf = forecast(rf6a$model, tail(Y, 60), 1, 6, 1)
pred6a.rf
pred12a.rf = forecast(rf12a$model, tail(Y, 60), 1, 12, 1)
pred12a.rf

###############################################################################
## Predicting direction of returns
###############################################################################

## RANDOM FOREST
install.packages("caret")
install.packages("readxl")
install.packages("e1071",dependencies = TRUE)
library(readxl)
library(randomForest)
library(caret)
library(e1071)
#note in this case indice is set to 15 as SP500EP is the 15th column
#read the file: df is the original data file for predicting actual returns
df_cat <- read_excel("D:/NUS Files JX/Y5S1/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/Data-Categorical.xlsx", sheet = "Data")

##New Data: to predict direction of return (0: negative return; 1: positive/no change in return)
#drop original SP500EP variable so it doesn't get passed in as a predictor
Y_cat  = data.matrix(df_cat)[,-15]
#get variable we want to predict
yy_cat = df_cat$SP500EPC

nprev=60

rf1c=rf.rolling.window(Y_cat,nprev,15,1) # 1 step forecast
rf3c=rf.rolling.window(Y_cat,nprev,15,3) # 3 step forecast
rf6c=rf.rolling.window(Y_cat,nprev,15,6) # 6 step forecast
rf12c=rf.rolling.window(Y_cat,nprev,15,12) #12 step forecast

#See the MSE:
rf.mse1=rf1c$errors[1] # MSE = 0.2713699 for 1 lag
rf.mse1
rf.mse3=rf3c$errors[1] # MSE = 0.2689571 for 3 lags
rf.mse3
rf.mse6=rf6c$errors[1] # MSE = 0.270481 for 6 lags
rf.mse6
rf.mse12=rf12c$errors[1] # MSE = 0.2580554 for 12 lags
rf.mse12


########################################################
###Diebold-Mariano (DM) tests
#########################################################
oosy = tail(yy,nprev)

#####################################################
#FIRST PART: Set up for the tests of pairwise comparison between each method vs. unconditional mean
#####################################################

#1. Compute squared loss for different horizons (Lasso)
lasso1c=(oosy-lasso1a$pred)^2
lasso3c=(oosy-lasso3a$pred)^2
lasso6c=(oosy-lasso6a$pred)^2
lasso12c=(oosy-lasso12a$pred)^2

#2. Compute squared loss for different horizons (Elastic net)
elastic1c=(oosy-elastic1a$pred)^2
elastic3c=(oosy-elastic3a$pred)^2
elastic6c=(oosy-elastic6a$pred)^2
elastic12c=(oosy-elastic12a$pred)^2

#3. Compute squared loss for different horizons (Ridge)
ridge1c=(oosy-ridge1a$pred)^2
ridge3c=(oosy-ridge3a$pred)^2
ridge6c=(oosy-ridge6a$pred)^2
ridge12c=(oosy-ridge12a$pred)^2

#4. Compute squared loss for different horizons (Random Forest)
lrf1c=(oosy-rf1a$pred)^2
lrf3c=(oosy-rf3a$pred)^2
lrf6c=(oosy-rf6a$pred)^2
lrf12c=(oosy-rf12a$pred)^2

#5. Compute squared loss for different horizons (Boosted)
boosted1c=(oosy-boosted1a$pred)^2
boosted3c=(oosy-boosted3a$pred)^2
boosted6c=(oosy-boosted6a$pred)^2
boosted12c=(oosy-boosted12a$pred)^2

#6. Compute squared loss for different horizons (ANN)
ann1c=(oosy-ann1a$pred)^2
ann3c=(oosy-ann3a$pred)^2
ann6c=(oosy-ann6a$pred)^2
ann12c=(oosy-ann12a$pred)^2

#7. Compute squared loss for different horizons (Post-Lasso)
post1c=(oosy-pols.lasso1a$pred)^2
post3c=(oosy-pols.lasso3a$pred)^2
post6c=(oosy-pols.lasso6a$pred)^2
post12c=(oosy-pols.lasso12a$pred)^2

#8. Compute squared loss for different horizons (Benchmark - Unconditional Mean (UM))
mean1=mean.rolling.window(Y,nprev,1,1)
mean3=mean.rolling.window(Y,nprev,1,3)
mean6=mean.rolling.window(Y,nprev,1,6)
mean12=mean.rolling.window(Y,nprev,1,12)


## Rolling Unconditional Mean as the predictor

mean1=mean.rolling.window(Y,nprev,1,1)
mean3=mean.rolling.window(Y,nprev,1,3)
mean6=mean.rolling.window(Y,nprev,1,6)
mean12=mean.rolling.window(Y,nprev,1,12)

mean1c=(oosy-mean1$pred)^2
mean3c=(oosy-mean3$pred)^2
mean6c=(oosy-mean6$pred)^2
mean12c=(oosy-mean12$pred)^2


#1. Compute loss differentials (d_t) for different horizons (UM- Lasso)
dumlasso1=mean1c-lasso1c
dumlasso3=mean3c-lasso3c
dumlasso6=mean6c-lasso6c
dumlasso12=mean12c-lasso12c

#2. Compute loss differentials (d_t) for different horizons (UM-Elastic)
dumelastic1=mean1c-elastic1c
dumelastic3=mean3c-elastic3c
dumelastic6=mean6c-elastic6c
dumelastic12=mean12c-elastic12c

#3. Compute loss differentials (d_t) for different horizons (UM-Ridge)
dumridge1=mean1c-ridge1c
dumridge3=mean3c-ridge3c
dumridge6=mean6c-ridge6c
dumridge12=mean12c-ridge12c

#4. Compute loss differentials (d_t) for different horizons (UM-Post Lasso)
dumpost1=mean1c-post1c
dumpost3=mean3c-post3c
dumpost6=mean6c-post6c
dumpost12=mean12c-post12c

#5. Compute loss differentials (d_t) for different horizons (UM-RF)
dumrf1=mean1c-lrf1c
dumrf3=mean3c-lrf3c
dumrf6=mean6c-lrf6c
dumrf12=mean12c-lrf12c

#6. Compute loss differentials (d_t) for different horizons (UN-Boosted)
dumboosted1=mean1c-boosted1c
dumboosted3=mean3c-boosted3c
dumboosted6=mean6c-boosted6c
dumboosted12=mean12c-boosted12c

#7. Compute loss differentials (d_t) for different horizons (UM-ANN)
dumann1=mean1c-ann1c
dumann3=mean3c-ann3c
dumann6=mean6c-ann6c
dumann12=mean12c-ann12c

#1. Create ts object containing loss differentials(UM- Lasso)
dtumlasso.ts=ts(cbind(dumlasso1,dumlasso3,dumlasso6,dumlasso12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumlasso.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumlasso.ts, main="Loss differential UM- Lasso",cex.axis=1.2)

#2. Create ts object containing loss differentials(UM-Elastic)
dtumelastic.ts=ts(cbind(dumelastic1,dumelastic3,dumelastic6,dumelastic12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumelastic.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumelastic.ts, main="Loss differential UM-Elastic",cex.axis=1.2)

#3. Create ts object containing loss differentials(UM-Ridge)
dtumridge.ts=ts(cbind(dumridge1,dumridge3,dumridge6,dumridge12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumridge.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumridge.ts, main="Loss differential UM-Ridge",cex.axis=1.2)

#4. Create ts object containing loss differentials(UM-Post Lasso)
dtumpost.ts=ts(cbind(dumpost1,dumpost3,dumpost6,dumpost12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumpost.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumpost.ts, main="Loss differential UM-Post Lasso",cex.axis=1.2)

#5. Create ts object containing loss differentials(UM-RF)
dtumrf.ts=ts(cbind(dumrf1,dumrf3,dumrf6,dumrf12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumrf.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumrf.ts, main="Loss differential UM-RF",cex.axis=1.2)

#6. Create ts object containing loss differentials(UN-Boosted)
dtumboosted.ts=ts(cbind(dumboosted1,dumboosted3,dumboosted6,dumboosted12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumboosted.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumboosted.ts, main="Loss differential UM-Boosted",cex.axis=1.2)

#7. Create ts object containing loss differentials(UM-ANN)
dtumann.ts=ts(cbind(dumann1,dumann3,dumann6,dumann12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumann.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumann.ts, main="Loss differential UM-ANN",cex.axis=1.2)

#1. DM regressions(UM- Lasso)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumlasso1=lm(dumlasso1~1) #regression
acf(dmumlasso1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumlasso1$coefficients/sqrt(NeweyWest(dmumlasso1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumlasso3=lm(dumlasso3~1)
acf(dmumlasso3$residuals)
dmumlasso3$coefficients/sqrt(NeweyWest(dmumlasso3,lag=4))
#6-step forecast test
dmumlasso6=lm(dumlasso6~1)
acf(dmumlasso6$residuals)
dmumlasso6$coefficients/sqrt(NeweyWest(dmumlasso6,lag=4))
#12-step forecast test
dmumlasso12=lm(dumlasso12~1)
acf(dmumlasso12$residuals)
dmumlasso12$coefficients/sqrt(NeweyWest(dmumlasso12,lag=4))

#2. DM regressions(UM- Elastic)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumelastic1=lm(dumelastic1~1) #regression
acf(dmumelastic1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumelastic1$coefficients/sqrt(NeweyWest(dmumelastic1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumelastic3=lm(dumelastic3~1)
acf(dmumelastic3$residuals)
dmumelastic3$coefficients/sqrt(NeweyWest(dmumelastic3,lag=4))
#6-step forecast test
dmumelastic6=lm(dumelastic6~1)
acf(dmumelastic6$residuals)
dmumelastic6$coefficients/sqrt(NeweyWest(dmumelastic6,lag=4))
#12-step forecast test
dmumelastic12=lm(dumelastic12~1)
acf(dmumelastic12$residuals)
dmumelastic12$coefficients/sqrt(NeweyWest(dmumelastic12,lag=4))

#3. DM regressions(UM- Ridge)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumridge1=lm(dumridge1~1) #regression
acf(dmumridge1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumridge1$coefficients/sqrt(NeweyWest(dmumridge1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumridge3=lm(dumridge3~1)
acf(dmumridge3$residuals)
dmumridge3$coefficients/sqrt(NeweyWest(dmumridge3,lag=4))
#6-step forecast test
dmumridge6=lm(dumridge6~1)
acf(dmumridge6$residuals)
dmumridge6$coefficients/sqrt(NeweyWest(dmumridge6,lag=4))
#12-step forecast test
dmumridge12=lm(dumridge12~1)
acf(dmumridge12$residuals)
dmumridge12$coefficients/sqrt(NeweyWest(dmumridge12,lag=4))

#4. DM regressions(UM- Post)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumpost1=lm(dumpost1~1) #regression
acf(dmumpost1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumpost1$coefficients/sqrt(NeweyWest(dmumpost1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumpost3=lm(dumpost3~1)
acf(dmumpost3$residuals)
dmumpost3$coefficients/sqrt(NeweyWest(dmumpost3,lag=4))
#6-step forecast test
dmumpost6=lm(dumpost6~1)
acf(dmumpost6$residuals)
dmumpost6$coefficients/sqrt(NeweyWest(dmumpost6,lag=4))
#12-step forecast test
dmumpost12=lm(dumpost12~1)
acf(dmumpost12$residuals)
dmumpost12$coefficients/sqrt(NeweyWest(dmumpost12,lag=4))

#5. DM regressions(UM- RF)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumrf1=lm(dumrf1~1) #regression
acf(dmumrf1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumrf1$coefficients/sqrt(NeweyWest(dmumrf1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumrf3=lm(dumrf3~1)
acf(dmumrf3$residuals)
dmumrf3$coefficients/sqrt(NeweyWest(dmumrf3,lag=4))
#6-step forecast test
dmumrf6=lm(dumrf6~1)
acf(dmumrf6$residuals)
dmumrf6$coefficients/sqrt(NeweyWest(dmumrf6,lag=4))
#12-step forecast test
dmumrf12=lm(dumrf12~1)
acf(dmumrf12$residuals)
dmumrf12$coefficients/sqrt(NeweyWest(dmumrf12,lag=4))

#6. DM regressions(UM- Boosted)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumboosted1=lm(dumboosted1~1) #regression
acf(dmumboosted1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumboosted1$coefficients/sqrt(NeweyWest(dmumboosted1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumboosted3=lm(dumboosted3~1)
acf(dmumboosted3$residuals)
dmumboosted3$coefficients/sqrt(NeweyWest(dmumboosted3,lag=4))
#6-step forecast test
dmumboosted6=lm(dumboosted6~1)
acf(dmumboosted6$residuals)
dmumboosted6$coefficients/sqrt(NeweyWest(dmumboosted6,lag=4))
#12-step forecast test
dmumboosted12=lm(dumboosted12~1)
acf(dmumboosted12$residuals)
dmumboosted12$coefficients/sqrt(NeweyWest(dmumboosted12,lag=4))

#7. DM regressions(UM- ANN)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumann1=lm(dumann1~1) #regression
acf(dmumann1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumann1$coefficients/sqrt(NeweyWest(dmumann1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumann3=lm(dumann3~1)
acf(dmumann3$residuals)
dmumann3$coefficients/sqrt(NeweyWest(dmumann3,lag=4))
#6-step forecast test
dmumann6=lm(dumann6~1)
acf(dmumann6$residuals)
dmumann6$coefficients/sqrt(NeweyWest(dmumann6,lag=4))
#12-step forecast test
dmumann12=lm(dumann12~1)
acf(dmumann12$residuals)
dmumann12$coefficients/sqrt(NeweyWest(dmumann12,lag=4))

###############################################################################
## Rolling Unconditional Mean as the predictor
###############################################################################

mean1=mean.rolling.window(Y,nprev,1,1)
mean3=mean.rolling.window(Y,nprev,1,3)
mean6=mean.rolling.window(Y,nprev,1,6)
mean12=mean.rolling.window(Y,nprev,1,12)

mean.mse1=mean1$errors[2]
mean.mse3=mean3$errors[2]
mean.mse6=mean6$errors[2]
mean.mse12=mean12$errors[2]

mean.mse1
mean.mse3
mean.mse6
mean.mse12


###############################################################################
## Predicting direction of returns
###############################################################################
set.seed(12345)
## RANDOM FOREST
install.packages("caret")
install.packages("readxl")
install.packages("e1071",dependencies = TRUE)
library(readxl)
library(randomForest)
library(caret)
library(e1071)
#note in this case indice is set to 15 as SP500EP is the 15th column
#read the file: df is the original data file for predicting actual returns
df_cat <- read_excel("D:/NUS Files JX/Y5S1/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/Data-Categorical.xlsx", sheet = "Data")

##New Data: to predict direction of return (0: negative return; 1: positive/no change in return)
#drop original SP500EP variable so it doesn't get passed in as a predictor
Y_cat  = data.matrix(df_cat)[,-15]
#get variable we want to predict
yy_cat = df_cat$SP500EPC

nprev=60

rf1c=rf.rolling.window(Y_cat,nprev,15,1) # 1 step forecast
rf3c=rf.rolling.window(Y_cat,nprev,15,3) # 3 step forecast
rf6c=rf.rolling.window(Y_cat,nprev,15,6) # 6 step forecast
rf12c=rf.rolling.window(Y_cat,nprev,15,12) #12 step forecast

#See the MSE:
rf.mse1=rf1c$errors[1] # MSE = 0.2713699 for 1 lag
rf.mse1
rf.mse3=rf3c$errors[1] # MSE = 0.2689571 for 3 lags
rf.mse3
rf.mse6=rf6c$errors[1] # MSE = 0.270481 for 6 lags
rf.mse6
rf.mse12=rf12c$errors[1] # MSE = 0.2580554 for 12 lags
rf.mse12

rfTab1c = table(rf1c$pred, tail(yy_cat,nprev)) 
confusionMatrix(rfTab1c) 
rfTab3c = table(rf3c$pred, tail(yy_cat,nprev))
confusionMatrix(rfTab3c)
rfTab6c = table(rf6c$pred, tail(yy_cat,nprev))
confusionMatrix(rfTab6c)
rfTab12c = table(rf12c$pred, tail(yy_cat,nprev))
confusionMatrix(rfTab12c)

