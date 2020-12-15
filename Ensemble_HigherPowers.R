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
library(lsei)
library(sandwich)

########replace with ur own source paths, delete before submission!!
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-ann.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-lasso.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-elastic.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-ridge.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/unconditional_mean.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-boostedtree.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-rf.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/unconditional_mean.R")
source("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/func-forecast.R")

##########################################################
#read data 
#########################################################
#read the file: df is the original data file for predicting actual returns
df <- read_excel("/Users/TuanDingWei/Dropbox/EC4308/Project/Empirical-Performance-of-Equity-Premium-Prediction-on-Various-Machine-Learning/Data-SnP Returns-Final.xlsx", sheet = "Data")
nvalidation = 60 #data between 2010 to 2014 (validation window)
ntest = 60 #data between 2015 to 2019

df_cat <- df[,-2]
yy_cat = df_cat$SP500RC
#actual result for direction
actual_direction = c(tail(yy_cat,ntest))

fb = formula(SP500R~poly(USCIPM,degree = 3,raw = TRUE)
             +poly(USPCE,degree = 3,raw = TRUE) + poly(USCPCE,degree = 3,raw = TRUE)
             +poly(USCINF,degree = 3,raw = TRUE) + poly(M1G,degree = 3,raw = TRUE)
             +poly(M2G,degree = 3,raw = TRUE) + poly(USCUN,degree = 3,raw = TRUE)
             +poly(USCFR,degree = 3,raw = TRUE) + poly(USCTBILL,degree = 3,raw = TRUE)
             +poly(EXRATE,degree = 3,raw = TRUE) + poly(DEBT,degree = 3,raw = TRUE)
             +poly(VAR,degree = 3,raw = TRUE) + poly(PE,degree = 3,raw = TRUE)
             +poly(DY,degree = 3,raw = TRUE))

#assign x and y of the model 
x = model.matrix(fb, data = df)
SP500R = df$SP500R
fullset = cbind(SP500R, x[,2:43])

Y = data.matrix(fullset)[1:(nrow(df)-ntest),]
yy = df$SP500R

Y_normal = data.matrix(fullset)
oosy = tail(yy,ntest)

##########################################################
#standard helper funcs
#########################################################
#Auxiliary function to compute root MSE (same as MSE before, but with square root):
RMSE <- function(pred, truth){ #start and end body of the function by { } - same as a loop
  return(sqrt(mean((truth - pred)^2)))
} #end function with a return(output) statement. Here we can go straight to return because the object of interest is a simple function of inputs

MSE <- function(pred, truth){ #start and end body of the function by { } - same as a loop
  return(mean((truth - pred)^2))
} #end function with a return(output) statement. Here we can go straight to return because the object of interest is a simple function of inputs

###############################################################################
## Artificial Neural Network (ANN)
###############################################################################

#Run forecasts for ANN (BIC)
#note in this case indice is set to 15 as SP500EP is the 43rd column
ann1a=ann.rolling.window(Y,nvalidation,1,1)
ann3a=ann.rolling.window(Y,nvalidation,1,3)
ann6a=ann.rolling.window(Y,nvalidation,1,6)
ann12a=ann.rolling.window(Y,nvalidation,1,12)

#RIDGE(BIC) MSEs
ann.mse1=ann1a$errors[1]
ann.mse1
ann.mse3=ann3a$errors[1]
ann.mse3
ann.mse6=ann6a$errors[1]
ann.mse6
ann.mse12=ann12a$errors[1]
ann.mse12

write.table(as.data.frame(cbind(ann1a$pred, ann3a$pred, ann6a$pred, ann12a$pred)), file = "Predictions/E_ann_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

pred1a.ann = forecast(ann1a$model, tail(Y, 60), 1, 1, 1)
pred1a.ann
pred3a.ann = forecast(ann3a$model, tail(Y, 60), 1, 3, 1)
pred3a.ann
pred6a.ann = forecast(ann6a$model, tail(Y, 60), 1, 6, 1)
pred6a.ann
pred12a.ann = forecast(ann12a$model, tail(Y, 60), 1, 12, 1)
pred12a.ann

###############################################################################
## Lasso
###############################################################################

alpha=1 #set alpha=1 for LASSO

#Run forecasts for LASSO (BIC)

lasso1a=lasso.rolling.window(Y,nvalidation,1,1,alpha,IC="bic", "gaussian")
lasso3a=lasso.rolling.window(Y,nvalidation,1,3,alpha,IC="bic", "gaussian")
lasso6a=lasso.rolling.window(Y,nvalidation,1,6,alpha,IC="bic", "gaussian")
lasso12a=lasso.rolling.window(Y,nvalidation,1,12,alpha,IC="bic", "gaussian")

lassoa.mse1=lasso1a$errors[2]
lassoa.mse3=lasso3a$errors[2]
lassoa.mse6=lasso6a$errors[2]
lassoa.mse12=lasso12a$errors[2]

lassoa.mse1
lassoa.mse3
lassoa.mse6
lassoa.mse12

write.table(as.data.frame(cbind(lasso1a$pred, lasso3a$pred, lasso6a$pred, lasso12a$pred)), file = "Predictions/E_lasso_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

pred1a.lasso = forecast(lasso1a$model, tail(Y, 60), 1, 1)
pred1a.lasso
pred3a.lasso = forecast(lasso3a$model, tail(Y, 60), 1, 3)
pred3a.lasso
pred6a.lasso = forecast(lasso6a$model, tail(Y, 60), 1, 6)
pred6a.lasso
pred12a.lasso = forecast(lasso12a$model, tail(Y, 60), 1, 12)
pred12a.lasso

###############################################################################
## Elastic Net
###############################################################################

alpha=0.5

elastic1a=elastic.rolling.window(Y,nvalidation,1,1,alpha,IC="bic", "gaussian")
elastic3a=elastic.rolling.window(Y,nvalidation,1,3,alpha,IC="bic",  "gaussian")
elastic6a=elastic.rolling.window(Y,nvalidation,1,6,alpha,IC="bic",  "gaussian")
elastic12a=elastic.rolling.window(Y,nvalidation,1,12,alpha,IC="bic",  "gaussian")

#LASSO(BIC) MSE's
elastic.mse1=elastic1a$errors[1]
elastic.mse3=elastic3a$errors[1]
elastic.mse6=elastic6a$errors[1]
elastic.mse12=elastic12a$errors[1]

elastic.mse1
elastic.mse3
elastic.mse6
elastic.mse12

write.table(as.data.frame(cbind(elastic1a$pred, elastic3a$pred, elastic6a$pred, elastic12a$pred)), file = "Predictions/E_elastic_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

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
ridge1a=ridge.rolling.window(Y,nvalidation,1,1,alpha,IC="bic", "gaussian")
ridge3a=ridge.rolling.window(Y,nvalidation,1,3,alpha,IC="bic", "gaussian")
ridge6a=ridge.rolling.window(Y,nvalidation,1,6,alpha,IC="bic", "gaussian")
ridge12a=ridge.rolling.window(Y,nvalidation,1,12,alpha,IC="bic", "gaussian")

#RIDGE(BIC) RMSE's
ridge.mse1=ridge1a$errors[1]
ridge.mse1
ridge.mse3=ridge3a$errors[1]
ridge.mse3
ridge.mse6=ridge6a$errors[1]
ridge.mse6
ridge.mse12=ridge12a$errors[1]
ridge.mse12

write.table(as.data.frame(cbind(ridge1a$pred, ridge3a$pred, ridge6a$pred, ridge12a$pred)), file = "Predictions/E_ridge_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

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
boosted1a=boosted.rolling.window(Y,nvalidation,1,1)
boosted3a=boosted.rolling.window(Y,nvalidation,1,3)
boosted6a=boosted.rolling.window(Y,nvalidation,1,6)
boosted12a=boosted.rolling.window(Y,nvalidation,1,12)

boosted.mse1=boosted1a$errors[1]
boosted.mse1
boosted.mse3=boosted3a$errors[1]
boosted.mse3
boosted.mse6=boosted6a$errors[1]
boosted.mse6
boosted.mse12=boosted12a$errors[1]
boosted.mse12

write.table(as.data.frame(cbind(boosted1a$pred, boosted3a$pred, boosted6a$pred, boosted12a$pred)), file = "Predictions/E_boosted_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

pred1a.boosted = forecast(boosted1a$model, tail(Y, 60), 1, 1, 1)
pred1a.boosted
pred3a.boosted = forecast(boosted3a$model, tail(Y, 60), 1, 3, 1)
pred3a.boosted
pred6a.boosted = forecast(boosted6a$model, tail(Y, 60), 1, 6, 1)
pred6a.boosted
pred12a.boosted = forecast(boosted12a$model, tail(Y, 60), 1, 12, 1)
pred12a.boosted

###############################################################################
## Random Forest - Rolling period
###############################################################################

#Rolling window forecast
rf1a=rf.rolling.window(Y,nvalidation,1,1) # 1 step forecast
rf3a=rf.rolling.window(Y,nvalidation,1,3) # 3 step forecast
rf6a=rf.rolling.window(Y,nvalidation,1,6) # 6 step forecast
rf12a=rf.rolling.window(Y,nvalidation,1,12) #12 step forecast

#See the MSE:
rf.mse1=rf1a$errors[1] # MSE =22.8 for 1 lag
rf.mse1
rf.mse3=rf3a$errors[1] # MSE =20.1 for 3 lags
rf.mse3
rf.mse6=rf6a$errors[1] # MSE = 19.3 for 6 lags
rf.mse6
rf.mse12=rf12a$errors[1] # MSE = 20.7 for 12 lags
rf.mse12

write.table(as.data.frame(cbind(rf1a$pred, rf3a$pred, rf6a$pred, rf12a$pred)), file = "Predictions/E_rf_pred.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

pred1a.rf = forecast(rf1a$model, tail(Y, 60), 1, 1, "tree")
pred1a.rf
pred3a.rf = forecast(rf3a$model, tail(Y, 60), 1, 3, "tree")
pred3a.rf
pred6a.rf = forecast(rf6a$model, tail(Y, 60), 1, 6, "tree")
pred6a.rf
pred12a.rf = forecast(rf12a$model, tail(Y, 60), 1, 12, "tree")
pred12a.rf

###############################################################################
## Rolling Unconditional Mean as the predictor
###############################################################################

mean1=mean.rolling.window(Y_normal,ntest,1,1)
mean3=mean.rolling.window(Y_normal,ntest,1,3)
mean6=mean.rolling.window(Y_normal,ntest,1,6)
mean12=mean.rolling.window(Y_normal,ntest,1,12)

mean.mse1=mean1$errors[2]
mean.mse3=mean3$errors[2]
mean.mse6=mean6$errors[2]
mean.mse12=mean12$errors[2]

mean.mse1
mean.mse3
mean.mse6
mean.mse12

###############################################################################
## Granger-Ramanathan combinations (Ensemble Learning)
###############################################################################

y_test = tail(data.matrix(yy),60)
y_validation = tail(data.matrix(yy)[1:(nrow(df)-ntest),],60)
#GR weights, no constant, all restrictions in place
##1-step ahead forecast
fmat1=cbind(ridge1a$pred, lasso1a$pred, elastic1a$pred, ann1a$pred, boosted1a$pred, rf1a$pred)
nregressors = ncol(fmat1)

gru1=lsei(fmat1, y_test, c=rep(1,nregressors), d=1, e=diag(nregressors), f=rep(0,nregressors))
View(gru1) #Examine weights; only ridge has a positive coefficient of 1

#Combine the forecasts with nonzero weights:
gre.pred1a=gru1[2]*lasso1a$pred+gru1[3]*elastic1a$pred+gru1[4]*ann1a$pred+gru1[6]*rf1a$pred
#MSE is the same as ridge regression
GRE.MSE1 = MSE(y_test,gre.pred1a)

write.table(as.data.frame(gre.pred1a), file = "Predictions/Ensemble_pred1.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

##3-step ahead forecast
fmat3=cbind(ridge3a$pred, lasso3a$pred, elastic3a$pred, ann3a$pred, boosted3a$pred, rf3a$pred)
gru3=lsei(fmat3, y_test, c=rep(1,nregressors), d=1, e=diag(nregressors), f=rep(0,nregressors))
View(gru3) #Examine weights; only ridge has a positive coefficient of 1

#Combine the forecasts with nonzero weights:
gre.pred3a=gru3[1]*ridge3a$pred+gru3[2]*lasso3a$pred+gru3[3]*elastic3a$pred+gru3[6]*rf3a$pred
GRE.MSE3 = MSE(y_test,gre.pred3a)

write.table(as.data.frame(gre.pred3a), file = "Predictions/Ensemble_pred3.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

##6-step ahead forecast
fmat6=cbind(ridge6a$pred, lasso6a$pred, elastic6a$pred, ann6a$pred, boosted6a$pred, rf6a$pred)
nregressors = ncol(fmat6)
gru6=lsei(fmat6, y_test, c=rep(1,nregressors), d=1, e=diag(nregressors), f=rep(0,nregressors))
View(gru6) #Examine weights; only ridge has a positive coefficient of 1

#Combine the forecasts with nonzero weights:
gre.pred6a=gru6[1]*ridge6a$pred+gru6[2]*lasso6a$pred+gru6[3]*elastic6a$pred+gru6[5]*boosted6a$pred
GRE.MSE6 = MSE(y_test,gre.pred6a)

write.table(as.data.frame(gre.pred6a), file = "Predictions/Ensemble_pred6.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

##12-step ahead forecast
fmat12=cbind(ridge12a$pred, lasso12a$pred, elastic12a$pred, ann12a$pred, boosted12a$pred, rf12a$pred)
nregressors = ncol(fmat12)
gru12=lsei(fmat12, y_test, c=rep(1,nregressors), d=1, e=diag(nregressors), f=rep(0,nregressors))
View(gru12) #Examine weights; only ridge has a positive coefficient of 1

#Combine the forecasts with nonzero weights:
gre.pred12a=gru12[1]*ridge12a$pred+gru12[2]*lasso12a$pred+gru12[3]*elastic12a$pred
GRE.MSE12=MSE(y_test,gre.pred12a)

write.table(as.data.frame(gre.pred12a), file = "Predictions/Ensemble_pred12.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

#1-step ahead forecast
rlassocomb.fit = rlasso(y_validation~fmat1,  post=FALSE,intercept=FALSE)
#Form combination:
comblasso1=cbind(ridge1a$pred, lasso1a$pred, elastic1a$pred, ann1a$pred, boosted1a$pred, rf1a$pred)%*%rlassocomb.fit$coefficients
LGRE.MSE1 = MSE(y_test,comblasso1)

#3-step ahead forecast
rlassocomb.fit = rlasso(y_validation~fmat3,  post=FALSE,intercept=FALSE)
#Form combination:
comblasso3=cbind(ridge3a$pred, lasso3a$pred, elastic3a$pred, ann3a$pred, boosted3a$pred, rf3a$pred)%*%rlassocomb.fit$coefficients
LGRE.MSE3 = MSE(y_test,comblasso3)

#6-step ahead forecast
rlassocomb.fit = rlasso(y_validation~fmat6,  post=FALSE,intercept=FALSE)
#Form combination:
comblasso6=cbind(ridge6a$pred, lasso6a$pred, elastic6a$pred, ann6a$pred, boosted6a$pred, rf6a$pred)%*%rlassocomb.fit$coefficients
LGRE.MSE6 = MSE(y_test,comblasso6)

#12-step ahead forecast
rlassocomb.fit = rlasso(y_validation~fmat12,  post=FALSE,intercept=FALSE)
#Form combination:
comblasso12=cbind(ridge12a$pred, lasso12a$pred, elastic12a$pred, ann12a$pred, boosted12a$pred, rf12a$pred)%*%rlassocomb.fit$coefficients
LGRE.MSE12 = MSE(y_test,comblasso12)

###############################################################################
## Diebold-Mariano (DM) tests
###############################################################################

#8. Compute loss differentials (d_t) for different horizons (UM- GRE)
GRE1c=(oosy-gre.pred1a)^2
GRE3c=(oosy-gre.pred3a)^2
GRE6c=(oosy-gre.pred6a)^2
GRE12c=(oosy-gre.pred12a)^2

mean1c=(oosy-mean1$pred)^2
mean3c=(oosy-mean3$pred)^2
mean6c=(oosy-mean6$pred)^2
mean12c=(oosy-mean12$pred)^2

dumGRE1=mean1c-GRE1c
dumGRE3=mean3c-GRE3c
dumGRE6=mean6c-GRE6c
dumGRE12=mean12c-GRE12c

dtumGRE.ts=ts(cbind(dumGRE1,dumGRE3,dumGRE6,dumGRE12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumGRE.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumGRE.ts, main="Loss differential UM - GRE",cex.axis=1.2)

#8. DM regressions(UM-GR)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumGRE1=lm(dumGRE1~1) #regression
acf(dmumGRE1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumGRE1$coefficients/sqrt(NeweyWest(dmumGRE1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumGRE3=lm(dumGRE3~1)
acf(dmumGRE3$residuals)
dmumGRE3$coefficients/sqrt(NeweyWest(dmumGRE3,lag=4))
#6-step forecast test
dmumGRE6=lm(dumGRE6~1)
acf(dmumGRE6$residuals)
dmumGRE6$coefficients/sqrt(NeweyWest(dmumGRE6,lag=4))
#12-step forecast test
dmumGRE12=lm(dumGRE12~1)
acf(dmumGRE12$residuals)
dmumGRE12$coefficients/sqrt(NeweyWest(dmumGRE12,lag=4))


LGRE1c=(oosy-comblasso1)^2
LGRE3c=(oosy-comblasso3)^2
LGRE6c=(oosy-comblasso6)^2
LGRE12c=(oosy-comblasso12)^2

dumLGRE1=mean1c-LGRE1c
dumLGRE3=mean3c-LGRE3c
dumLGRE6=mean6c-LGRE6c
dumLGRE12=mean12c-LGRE12c

dtumLGRE.ts=ts(cbind(dumLGRE1,dumLGRE3,dumLGRE6,dumLGRE12), start=c(2015,1), end=c(2019,12), freq=12)
colnames(dtumLGRE.ts)=c("1-step dt","3-step dt","6-step dt","12-step dt")
#Plot them to examine stationarity:
plot.ts(dtumLGRE.ts, main="Loss differential UM - Lasso(GRE)",cex.axis=1.2)

#9. DM regressions(UM-GR)
#1-step forecasts on a constant - get estimate of mean(d_t)
dmumLGRE1=lm(dumLGRE1~1) #regression
acf(dmumLGRE1$residuals) #check serial correlation of residuals - number of significant autocorrelations is a good guess for number lags included in the HAC variance estimator
dmumLGRE1$coefficients/sqrt(NeweyWest(dmumLGRE1,lag=4)) #form the DM t-statistic
#3-step forecast test
dmumLGRE3=lm(dumLGRE3~1)
acf(dmumLGRE3$residuals)
dmumLGRE3$coefficients/sqrt(NeweyWest(dmumLGRE3,lag=4))
#6-step forecast test
dmumLGRE6=lm(dumLGRE6~1)
acf(dmumLGRE6$residuals)
dmumLGRE6$coefficients/sqrt(NeweyWest(dmumLGRE6,lag=4))
#12-step forecast test
dmumLGRE12=lm(dumLGRE12~1)
acf(dmumLGRE12$residuals)
dmumLGRE12$coefficients/sqrt(NeweyWest(dmumLGRE12,lag=4))

###############################################################################
## 2020 Predictions
###############################################################################

pred1a.gre=gru1[2]*pred1a.lasso+gru1[3]*pred1a.elastic+gru1[4]*pred1a.ann+gru1[6]*pred1a.rf
pred1a.gre

pred3a.gre=gru3[1]*pred3a.ridge+gru3[2]*pred3a.lasso+gru3[3]*pred3a.elastic+gru3[6]*pred3a.rf
pred3a.gre

pred6a.gre=gru6[1]*pred6a.ridge+gru6[2]*pred6a.lasso+gru6[3]*pred6a.elastic+gru6[5]*pred6a.boosted
pred6a.gre

pred12a.gre=gru12[1]*pred3a.ridge+gru12[2]*pred12a.lasso+gru12[3]*pred12a.elastic
pred12a.gre

comblasso1.pred=cbind(pred1a.ridge, pred1a.lasso, pred1a.elastic, pred1a.ann, pred1a.boosted, pred1a.rf)%*%rlassocomb.fit$coefficients
comblasso1.pred

comblasso3.pred=cbind(pred3a.ridge, pred3a.lasso, pred3a.elastic, pred3a.ann, pred3a.boosted, pred3a.rf)%*%rlassocomb.fit$coefficients
comblasso3.pred

comblasso6.pred=cbind(pred6a.ridge, pred6a.lasso, pred6a.elastic, pred6a.ann, pred6a.boosted, pred6a.rf)%*%rlassocomb.fit$coefficients
comblasso6.pred

comblasso12.pred=cbind(pred12a.ridge, pred12a.lasso, pred12a.elastic, pred12a.ann, pred12a.boosted, pred12a.rf)%*%rlassocomb.fit$coefficients
comblasso12.pred

###############################################################################
## Directions
###############################################################################

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

get_direction <- function(pred) {
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

nprev=60

gre1a_dir = get_direction(gre.pred1a)
gre3a_dir = get_direction(gre.pred3a)
gre6a_dir = get_direction(gre.pred6a)
gre12a_dir = get_direction(gre.pred12a)

calc_accuracy(gre1a_dir, actual_direction) 
calc_accuracy(gre3a_dir, actual_direction) 
calc_accuracy(gre6a_dir, actual_direction) 
calc_accuracy(gre12a_dir, actual_direction) 


comblasso1a_dir = get_direction(comblasso1)
comblasso3a_dir = get_direction(comblasso3)
comblasso6a_dir = get_direction(comblasso6)
comblasso12a_dir = get_direction(comblasso12)

calc_accuracy(comblasso1a_dir, actual_direction) 
calc_accuracy(comblasso3a_dir, actual_direction) 
calc_accuracy(comblasso6a_dir, actual_direction) 
calc_accuracy(comblasso12a_dir, actual_direction) 
