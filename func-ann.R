#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#3) lag - the forecast horizon

runann=function(Y,indice,lag) {

  comp=princomp(scale(Y,scale=FALSE)) # compute principal components to add as predictors
  Y2=cbind(Y,comp$scores[,1:4]) #augment predictors by the first 4 principal components
  aux=embed(Y2,4+lag) #create 2 lags + forecast horizon shift (=lag option)
  SP500R=aux[,indice] #  Y variable aligned/adjusted for missing data due do lags
  X=aux[,-c(1:(ncol(Y2)*lag))]   # lags of Y (predictors) corresponding to forecast horizon
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)] #retrieve the last  observations if one-step forecast
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))] #delete first (h-1) columns of aux,
    X.out=tail(X.out,1)[1:ncol(X)] #last observations: y_T,y_t-1...y_t-h
  }
  col_name = c()
  for(col in 1:ncol(X)) {
    col_name[col] <- paste("r", toString(col))
  }
  
  X.out = t(as.matrix(X.out))
  colnames(X.out)<-col_name
  colnames(X)<-col_name
  
  #Functional Checks
  #print(col_name)
  #print(X)
  #print(X.out)
  
  #here we use the recommened configurations
  model=nnet(SP500R~., data=cbind(SP500R,X), size=8,  maxit=1000, decay=0.01, linout = TRUE, trace=FALSE, MaxNWts=4000)
  pred=predict(model,X.out)
  
  return(list("model"=model,"pred"=pred)) #return the estimated model and h-step forecast
}

#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) nprev - number of out-of-sample observations (at the end of the sample)

#3) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#4) lag - the forecast horizon

ann.rolling.window=function(Y,nprev,indice,lag){

  save.pred=matrix(NA,nprev,1) #blank for forecasts
  model = NULL #create variable to store model
  for(i in nprev:1){ #NB: backwards FOR loop: going from 60 down to 1
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] #define the estimation window (first one: 1 to 420, then 2 to 421 etc.)
    ann=runann(Y.window,indice,lag) #call the function to fit the ann selected on IC and generate h-step forecast
    save.pred[(1+nprev-i),]=ann$pred #save the forecast
    model = ann$model
    cat("iteration",(1+nprev-i),"\n") #display iteration number
  }
  #Some helpful stuff:
  real=Y[,indice] #get actual values
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red") #padded with NA for blanks, plot predictions vs. actual

  mse=mean((tail(real,nprev)-save.pred)^2) #compute MSE
  mae=mean(abs(tail(real,nprev)-save.pred)) #compute MAE (Mean Absolute Error)
  errors=c("mse"=mse,"mae"=mae) #stack errors in a vector
  
  return(list("pred"=save.pred,"errors"=errors, "model"=model)) #return forecasts, history of estimated coefficients, and RMSE and MAE for the period.
}