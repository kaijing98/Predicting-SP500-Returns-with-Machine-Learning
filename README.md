# Evaluating the Empirical Performance of S&P 500 Returns on Various Machine Learning Algorithms to beat the Unconditional Mean of Returns

**Goal of Project:** Attempt to predict the S&P500 Index using macroeconomics variables and fundamental indicators.

**Dependent Variable:** Returns on S&P500 

**Brief Outline of Data:**
We propose using monthly data for all regressors and dependent variables (i.e. predict on the last date of every month). Given the developments in the financial markets, a comprehensive set of data points that range between the 1980s and 2019 are used. 

**Initial Regressors (seasonally adjusted)**:
Macro Indicators: GDP growth rates, Inflation rates, M1 growth, M2 growth, Change in unemployment rates, Changes in federal funds rates (short term interest rate), Changes in 10Y Treasury Bill Rate (long term interest rate), Changes in exchange rates (DXY), Changes in public debt, Change in Stock Variance (VAR)  - Realised volatility,  Change in VIX  - Implied volatility
Fundamental Indicators: Dividend yields, PE ratio, EV/EBITDA ratio

**Suggested Evaluation Methods:**
Method 1: Information Criterion (AIC/BIC) 
Method 2: Recursive CV

**Analytical Tools**:
  * Lasso
  * Ridge Regression
  * Elastic Net
  * Boosted Regression Tree
  * Neural Network 
  * Sentiment Analysis

**Beat the Unconditional Mean**
Beat the MSE of unconditional mean as the predictor

**Predicting the Direction of S&P500 Index**
Compare the proportion of cases where the sign of equity premium is forecasted correctly compared to the proportion of cases where the sign is incorrect. Direction is more important in practice than getting the value exact (and hence not so much about MSE) because traders most of the time care whether the excess return on the stock is positive or negative when we make portfolio decisions. The natural benchmark here is a coin toss (50% probability of guessing right) – so if some model predicts the sign much better than this, it may be most useful even if it may not have the lowest MSE.


**Project References:**
* Predicting the bear stock market: Macroeconomic variables as leading indicators (Shiu-Sheng Chen)
* Forecasting Individual Stock Returns Using Macroeconomic and Technical Variables (Hui Zeng,	Ben R. Marshall, Nhut H. Nguyen, Nuttawat Visaltanachoti)
* A Comprehensive Look at The Empirical Performance of Equity Premium Prediction (Ivo Welch, Amit Goyal)
