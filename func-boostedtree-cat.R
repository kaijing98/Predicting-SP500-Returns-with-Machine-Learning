#Here we follow Medeiros et al. (2019) in defining two functions:

#One for forming forecasts using AR(p) model selected on BIC, which will be called
#on each iteration of the rolling window forecasting exercise.

#The other one for producing the series of h-step forecasts using rolling window.

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#3) lag - the forecast horizon

runboostedcat=function(Y,indice,lag){
  
  comp=princomp(scale(Y,scale=FALSE)) # compute principal components to add as predictors
  Y2=cbind(Y,comp$scores[,1:4]) #augment predictors by the first 4 principal components
  aux=embed(Y2,4+lag) #create 4 lags + forecast horizon shift (=lag option)
  SP500RC=aux[,indice] #  Y variable aligned/adjusted for missing data due do lags
  X=aux[,-c(1:(ncol(Y2)*lag))]  # lags of Y (predictors) corresponding to forecast horizon 
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]   #retrieve the last  observations if one-step forecast
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))] #delete first (h-1) columns of aux,
    X.out=tail(X.out,1)[1:ncol(X)]  #last observations: y_T,y_t-1...y_t-h
  }
  
  col_name = c()
  for(col in 1:ncol(X)) {
    col_name[col] <- paste("r", toString(col))
  }
  
  X.out = t(as.matrix(X.out))
  colnames(X.out)<-col_name
  colnames(X)<-col_name
  
  model= gbm(SP500RC~.,data=as.data.frame(cbind(SP500RC,X)),distribution='bernoulli',
             interaction.depth=5,n.trees=10000,shrinkage=.01)#fit the random forest on default settings
  pred_probs= predict(model,as.data.frame(X.out),type = "response") #generate forecast
    if (pred_probs > 0.5) {    
      pred = cbind(1)
    } else {  
      pred = cbind(0)
    }  
  
  return(list("model"=model,"pred"=pred)) #return the estimated model and h-step forecast
}


#This function will repeatedly call the previous function in the rolling window h-step forecasting

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) nprev - number of out-of-sample observations (at the end of the sample)

#3) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#4) lag - the forecast horizon

boostedcat.rolling.window=function(Y,nprev,indice=1,lag=1){
  
  #save.importance=list() #blank for saving variable importance
  save.pred=matrix(NA,nprev,1) ##blank for forecasts
  for(i in nprev:1){#NB: backwards FOR loop: going from 180 down to 1
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] #define the estimation window (first one: 1 to 491, then 2 to 492 etc.)
    boosted=runboostedcat(Y.window,indice,lag)#call the function to fit the Random Forest and generate h-step forecast
    save.pred[(1+nprev-i),]=boosted$pred #save the forecast
    #save.importance[[i]]=importance(boosted$model) #save variable importance
    cat("iteration",(1+nprev-i),"\n") #display iteration number
  }
  #Some helpful stuff:
  real=Y[,indice]#get actual values
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red") #padded with NA for blanks, plot predictions vs. actual
  
  mse=mean((tail(real,nprev)-save.pred)^2) #compute MSE
  mae=mean(abs(tail(real,nprev)-save.pred)) #compute MAE (Mean Absolute Error)
  errors=c("mse"=mse,"mae"=mae) #stack errors in a vector
  
  return(list("pred"=save.pred,"errors"=errors)) #return forecasts, history of variable importance, and RMSE and MAE for the period.
}

