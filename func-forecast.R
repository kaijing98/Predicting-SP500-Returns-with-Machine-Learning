forecast=function(model,Y,indice,lag, formula = 0) {

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
  
  pred=NULL
  if(identical(formula, 1)) { #for ANN and rboosted
    col_name = c()
    for(col in 1:ncol(X)) {
      col_name[col] <- paste("r", toString(col))
    }
    X.out = t(as.matrix(X.out))
    colnames(X.out)<-col_name
    
    pred=predict(model,as.data.frame(X.out)) 
  } else if (identical(formula, 2)) { #for post-lasso
    X.out=c(X.out) #last observations
    coeff=coef(model) #vector of OLS coefficients
    coeff[is.na(coeff)]=0 #if NA set to 0
    pred=c(1,X.out[which(coeff[-1]!=0)])%*%coeff
  } else {
    pred=predict(model,X.out) 
  }
  
  return(pred)
}
