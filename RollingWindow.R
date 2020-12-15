tsrollingwindow <- function(model, ntrain, ntest) {
  
  for (i in 1:ntest) {
    left = i
    right = length(train) + i - 1
    grid = 10^seq(10, -2, length = 100)
    lasso.mod <- glmnet(xtrain, ytrain, alpha = 1, lambda = grid)
  }
}

MSE <- function(pred, truth) { 
  return(mean((truth - pred)^2)) 
}
