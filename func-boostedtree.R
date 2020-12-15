#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#3) lag - the forecast horizon

runboosted=function(Y,indice,lag){
  
  comp=princomp(scale(Y,scale=FALSE)) # compute principal components to add as predictors
  Y2=cbind(Y,comp$scores[,1:4]) #augment predictors by the first 4 principal components
  aux=embed(Y2,4+lag) #create 4 lags + forecast horizon shift (=lag option)
  SP500R=aux[,indice] #  Y variable aligned/adjusted for missing data due do lags
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
  
  model= gbm(SP500R~.,data=as.data.frame(cbind(SP500R,X)),distribution='gaussian',
             interaction.depth=5,n.trees=10000,shrinkage=.01)#fit the random forest on default settings
  pred=predict(model,as.data.frame(X.out)) #generate forecast
  
  return(list("model"=model,"pred"=pred)) #return the estimated model and h-step forecast
}


#This function will repeatedly call the previous function in the rolling window h-step forecasting

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) nprev - number of out-of-sample observations (at the end of the sample)

#3) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#4) lag - the forecast horizon

boosted.rolling.window=function(Y,nprev,indice=1,lag=1){
  
  #save.importance=list() #blank for saving variable importance
  save.pred=matrix(NA,nprev,1) ##blank for forecasts
  model = NULL #create variable to store model
  for(i in nprev:1){#NB: backwards FOR loop: going from 180 down to 1
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] #define the estimation window (first one: 1 to 491, then 2 to 492 etc.)
    boosted=runboosted(Y.window,indice,lag)#call the function to fit the Random Forest and generate h-step forecast
    print(length(boosted$model$coef))
    save.pred[(1+nprev-i),]=boosted$pred #save the forecast
    #save.importance[[i]]=importance(boosted$model) #save variable importance
    model = boosted$model
    cat("iteration",(1+nprev-i),"\n") #display iteration number
  }
  #Some helpful stuff:
  real=Y[,indice]#get actual values
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red") #padded with NA for blanks, plot predictions vs. actual
  
  mse=mean((tail(real,nprev)-save.pred)^2) #compute MSE
  mae=mean(abs(tail(real,nprev)-save.pred)) #compute MAE (Mean Absolute Error)
  errors=c("mse"=mse,"mae"=mae) #stack errors in a vector
  
  return(list("pred"=save.pred,"errors"=errors, "model"=model)) #return forecasts, history of variable importance, and RMSE and MAE for the period.
}

