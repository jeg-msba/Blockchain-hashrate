# Using Hash Rates to Predict Blockchain Performance

Hash Rates influence mining rewards, blockchain security, network congestion, and electricity usage. 
In this study we created a variety of Hash Rate Time Series Models including multivariate models in an attempt to to predict bitcoin valuation.
Future work could attempt to predict network congestion, electricity usage, and other factors dependent on blockchain transactions.
We decomposed the hash rate time series to identify trend, seasonality and variance. We created a baseline model (naive), Holt, Holt-Winters, 
SARIMA, Neural Net, and Autoregressive Models. We chose the model with the lowest RMSE, compared training and test data, then corrected for non constant variance

![image](https://github.com/jeg-msba/Blockchain-hashrate/assets/111711622/291a54a7-deb0-4a08-841a-9582e93b4f9f)



Hash-Rate-Model-presentation.pdf is the PowerPoint presentation describing the project, models used, results, and conclusion. <br/>
hashrate.ts.R is the R code used to create the time series models. <br/>
BTC-USD.csv, hashrate.csv, NVDIA.csv are table data listing daily bitcoin prices, NVIDIA stock price, and hash rates. BTC and Hash Rates are 7 days a week, NVIDIA is 5 days a week. <br/>


Sources

# Yahoo Finance
