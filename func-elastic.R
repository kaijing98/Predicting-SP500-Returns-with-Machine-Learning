#Inputs for the function:

#1) Data matrix Y: includes all variables

#2) indice - index for dependent variable: 1 for CPI inflation, 2 for PCE inflation

#3) lag - the forecast horizon

#4) alpha - the alpha value for glmnet

#5) IC - information criteria

#6) family - distribution

runelastic=function(Y,indice,lag,alpha=0.5,IC="bic", family){
  
  comp=princomp(scale(Y,scale=FALSE)) # compute principal components to add as predictors
  Y2=cbind(Y,comp$scores[,1:4]) #augment predictors by the first 4 principal components
  aux=embed(Y2,4+lag) #create 3 lags + forecast horizon shift (=lag option)
  y=aux[,indice] #  Y variable aligned/adjusted for missing data due to lags
  X=aux[,-c(1:(ncol(Y2)*lag))]   # lags of Y (predictors) corresponding to forecast horizon   
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)] #retrieve the last  observations if one-step forecast  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))] #delete first (h-1) columns of aux,
    X.out=tail(X.out,1)[1:ncol(X)] #last observations: y_T,y_t-1...y_t-h
  }
  
  #Here we use the glmnet wrapper written by the authors that does selection on IC:
  model=ic.glmnet(X,y,crit=IC,alpha = alpha, distType=family) #fit the elastic/ElNet model selected on IC

  #family = gaussian
  pred=predict(model,X.out) #generate the forecast (note c(X.out,0) gives the last observations on X's and the dummy (the zero))
    
  return(list("model"=model,"pred"=pred)) #return the estimated model and h-step forecast
}

elastic.rolling.window=function(Y,nprev,indice=15,lag=1,alpha=0.5,IC="bic", family){
  
  save.coef=matrix(NA,nprev, 21 + ncol(Y[,-indice])*4) #blank matrix for coefficients at each iteration
  save.pred=matrix(NA,nprev,1) #blank for forecasts
  model = NULL #create variable to store model
  for(i in nprev:1){ #NB: backwards FOR loop: going from 60 down to 1
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] #define the train window
    elastic=runelastic(Y.window,indice,lag,alpha,IC, family) #call the function to fit the elastic/ElNET selected on IC and generate h-step forecast
    save.coef[(1+nprev-i),]=elastic$model$coef #save estimated coefficients
    save.pred[(1+nprev-i),]=elastic$pred #save the forecast
    model=elastic$model
    cat("iteration",(1+nprev-i),"\n") #display iteration number
  }
  #Some helpful stuff:
  real=Y[,indice] #get actual values
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red") #padded with NA for blanks, plot predictions vs. actual
  
  mse=mean((tail(real,nprev)-save.pred)^2) #compute MSE
  mae=mean(abs(tail(real,nprev)-save.pred)) #compute MAE (Mean Absolute Error)
  errors=c("mse"=mse,"mae"=mae) #stack errors in a vector
  
  return(list("pred"=save.pred,"coef"=save.coef,"errors"=errors, "model"=model)) #return forecasts, history of estimated coefficients, and RMSE and MAE for the period.
}
