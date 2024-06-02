################################################################################################
# This program will create time series models for several blockchain related data sets.
# The main data set is blockchain hash-rate, which is a measure of the computational power
# of a blockchain network. Others include bitcoin price and NVDIA stock price, which will be
# used to test for multivariate autoregression. The data will be split into training and test. 
# Cross Validation will be used on the Neural Network algorithm to see if the model is over fit.
################################################################################################

# Import libraries needed
library(forecast)
library(MTS)
library(FinTS)
library(rugarch)
library(tseries)
library(vars)

########################################################
############--Data Import and Cleaning--################
########################################################

setwd("~/Documents/Bentley Courses/MA 611")
nvdia_data = read.csv('nvdia.csv')
hash_data = read.csv('hashrate.csv')
btc_data = read.csv('BTC-USD.csv')
btc_data$Date = as.Date(btc_data$Date)
hash_data$Date <- as.Date(hash_data$Date)

# Rename columns, drop those not needed
nvdia_data = subset(nvdia_data, select = c('Date', 'Adj.Close'))
hash_data = subset(hash_data, select = -c(X, X.1))
btc_data = subset(btc_data, select = c('Date','Adj.Close'))

# Create a unified data.frame with both btc and hashrate. Merge on date.
# Do the same for btc and nvdia
data = merge(hash_data, btc_data,by = 'Date')
data2 = merge(btc_data, nvdia_data, by = 'Date')
names(data) = c('Date','hash', 'btc') 
names(data2) = c('Date', 'btc', 'nvdia')
rownames(data) = data$Date
rownames(data2) = data2$Date
data = data[, -1]
data2 = data2[, -1]

# Split the data into training and test
data.train = data[1:1080,]
data.test  = data[1081:1094,]

# Create time series for all the data, training data, and test data
hash = ts(data$hash, frequency = 7)
hash_train = ts(data.train$hash, frequency = 7)
hash_test = ts(data.test$hash, frequency = 7, start = c(155,3))
btc = ts(data$btc, frequency = 7)
btc_train = ts(data.train$btc, frequency = 7)
btc_test = ts(data.test$btc, frequency = 7)
btc_5days = ts(data2$btc, frequency = 5)
nvdia = ts(data2$nvdia, frequency = 5)

#scale the data
hash = hash/10^6
hash_train = hash_train/10^6
hash_test = hash_test/10^6
btc = btc/100
btc_train = btc_train/100
btc_test = btc_test/100

# Investigate the time series. Is there trend? seasonality? white noise? Yes, no, no
autoplot(hash) + autolayer(btc) # They do not seem to be  correlated
cor(btc, hash) # .12 correlation
hash.decompose = decompose(hash)
plot(hash.decompose)

# STL is useful when the seasonality is not constant
stl(hash, s.window = 7)
plot(stl(hash, s.window = 365)) # Looks like non-constant variance

# Calculate Strength of Trend and Seasonality
strength_trend_hash = 1 - var(hash.decompose$random, na.rm=TRUE) / (var(hash.decompose$random + hash.decompose$trend, na.rm=TRUE))
strength_season_hash = 1 - var(hash.decompose$random, na.rm=TRUE) / (var(hash.decompose$random + hash.decompose$seasonal, na.rm=TRUE))
strength_trend_hash
strength_season_hash

########################################
############--Modeling--################
########################################

# USING TRAIN VS TEST DATA

###################################################################
# Naive - baseline model                                                    #
###################################################################
hash.naive = naive(hash_train, h=14)
autoplot(hash.naive) + autolayer(fitted(hash.naive))
ggAcf(resid(hash.naive))

# Calculate accuracy for training and test
test_RMSE.naive = accuracy(hash.naive, hash_test)
test_RMSE.naive
# RMSE training = 39.63294
# RMSE on test: 52.19590

############################################################################
# Holt - Model trend but not seasonality. Plot in-sample and forecast. ACF #
############################################################################
hash.holt = holt(hash_train,h = 14)
autoplot(hash.holt) + autolayer(hash_test)
ggAcf(resid(hash.holt))

# Calculate accuracy for training and test
accuracy(hash.holt)
test_RMSE.h = accuracy(hash.holt, hash_test)
test_RMSE.h
#RMSE = 29.88099
#RMSE on test: 49.71361

#################################################################################
# Holt-Winters - Model trend and seasonality.. Plot in-sample and forecast. ACF #
#################################################################################
hash.hw = hw(hash_train, h=14)
autoplot(hash.hw) + autolayer(fitted(hash.hw))
ggAcf(resid(hash.hw))

# Calculate accuracy for training and test
test_RMSE.hw = accuracy(hash.hw, hash_test)
test_RMSE.hw
# RMSE = 29.73498
# RMSE on test: 47.90546

###################################################################
# ARIMA model - Plot in-sample and forecast. ACF                  #
###################################################################
hash.arima = auto.arima(hash_train, stepwise = FALSE)
arima.forecast = forecast(hash.arima, h = 14)
autoplot(arima.forecast) + autolayer(hash_test)
ggAcf(resid(hash.arima))

# Calculate accuracy for training and test
test_RMSE.arima = accuracy(arima.forecast, hash_test)
test_RMSE.arima
#RMSE = 29.91551
#RMSE on test: 52.89309

###################################################################
# Neural Nets. Plot in-sample and forecast. ACF
###################################################################
nnetar_p = 10
nnetar_size = 20
t_start = 910

hash.nn = nnetar(hash_train, p=nnetar_p, P=1, size=nnetar_size)
hash.nn
nn.forecast = forecast(hash.nn, h = 14)
autoplot(nn.forecast) + autolayer(hash_test)
ggAcf(resid(hash.nn))

# Calculate accuracy for training and test
test_RMSE.nn = accuracy(nn.forecast, hash_test)
test_RMSE.nn
# RMSE = 21.16962
# RMSE on test: 56.31455

# Now use cross validation to improve the model
# NNAR trained on all data, with in-sample preds and RMSE t_start to end
hash.nnetar = nnetar(hash, p=nnetar_p, P=1, size=nnetar_size)		# 41 weights
hash.nnetar.resids = resid(hash.nnetar)
hash.nnetar.resids[1:t_start] = NA		# nuke preds/resids up to t_start
hash.nnetar.preds.rmse = sqrt(mean(hash.nnetar.resids^2, na.rm=T))
hash.nnetar.preds = hash - hash.nnetar.resids
cat('NNAR RMSE:', hash.nnetar.preds.rmse, '\n')
# RMSE: 32.47672
hash.nnetar

hash_forecast_function = function(x, h) {
  forecast(nnetar(x, p=nnetar_p, P=1, size=nnetar_size), h = 1)
}

hash.nnetar.cv.resids = tsCV(hash, hash_forecast_function, initial=t_start)
hash.nnetar.cv.rmse = sqrt(mean(hash.nnetar.cv.resids^2, na.rm=T))
hash.nnetar.cv.preds = hash - stats::lag(hash.nnetar.cv.resids, -1)		# CV resids are annoying lag-1 back

cat('NNAR CV RMSE:', hash.nnetar.cv.rmse)
# RMSE: 58.55193
# RMSE: 59.84547
# RMSE: 60.93181

fig = autoplot(bpd) +
  autolayer(bpd.nnetar.preds, series='NNETAR') +
  autolayer(bpd.nnetar.cv.preds, series='NNETAR CV')
print(fig)

###################################################################
# Linear model
###################################################################
hash.lm = tslm(hash_train ~ poly(trend, 3) + season)
lm.forecast = forecast(hash.lm, h = 14)
autoplot(lm.forecast) + autolayer(fitted(hash.lm)) + autolayer(hash_test)
ggAcf(resid(hash.lm))

# Calculate accuracy for training and test
test_RMSE.lm = accuracy(lm.forecast, hash_test)
test_RMSE.lm
#RMSE = 32.58723
#RMSE on test: 46.22329

###################################################################
# VAR - Create a multivariate time series for   BTC and Hash Rate 
###################################################################
prices = data.frame(btc_train,hash_train)
mod2 = VAR(prices, p=1)

var.forecast <- predict(mod2, n.ahead = 14)
vf <- data.frame(var.forecast$fcst)
RMSE <- sqrt(mean((vf[,5] - hash_test)^2))
RMSE  #RMSE: 116.2483
summary(mod2)
ggAcf(resid(mod2))

##########################################################################
# Holt-Winters was the winner, by a slim margin. Check for heteroscedacity
##########################################################################
ArchTest(resid(hash.hw)^2) # Yes, non-constant variance
autoplot(resid(hash.hw))
ggAcf(resid(hash.hw))
ggAcf(resid(hash.hw)^2)
hash.resid.garch = garch(resid(hash.hw), order=c(1,1),trace=FALSE)
hash.resid.garch
autoplot(hash.hw) +
  autolayer(fitted(hash.hw) + 2*abs(fitted(hash.resid.garch)),color = 'red', series="CI") +
  autolayer(fitted(hash.hw) -2*abs(fitted(hash.resid.garch)),color = 'red', series="CI") +
  autolayer(fitted(hash.hw), color = 'green')


##############################################################
##############################################################
##############################################################
##############-- Other models we tried --#####################
##############################################################
##############################################################
##############################################################

# nvdia data
nvdia.decompose = decompose(nvdia)
plot(nvdia.decompose)
stl(nvdia, s.window = 7)
plot(stl(nvdia, s.window = 365))

# Strength of seasonality and trend
#nvdia
strength_trend = 1 - var(nvdia.decompose$random, na.rm=TRUE) / (var(nvdia.decompose$random + nvdia.decompose$trend, na.rm=TRUE))
strength_season = 1 - var(nvdia.decompose$random, na.rm=TRUE) / (var(nvdia.decompose$random + nvdia.decompose$seasonal, na.rm=TRUE))
strength_trend
strength_season

# Holt, HW, ARIMA
nvdia.naive = naive(nvdia)
accuracy(nvdia.naive)
#RMSE = 10.63365
ggAcf(resid(nvdia.naive))

nvdia.holt = holt(nvdia)
accuracy(nvdia.holt) #RMSE = 10.83134
ggAcf(resid(nvdia.holt))

nvdia.hw = hw(nvdia)
accuracy(nvdia.hw) #RMSE = 10.81
ggAcf(resid(nvdia.hw))

nvdia.arima = auto.arima(nvdia, stepwise = FALSE)
accuracy(nvdia.arima) #RMSE = 10.70756
ggAcf(resid(nvdia.arima))

#############################################
# BTC data decomposed
btc.decompose = decompose(btc)
plot(btc.decompose)

#seasonality vs trend
strength_trend_btc = 1 - var(btc.decompose$random, na.rm=TRUE) / (var(btc.decompose$random + btc.decompose$trend, na.rm=TRUE))
strength_season_btc = 1 - var(btc.decompose$random, na.rm=TRUE) / (var(btc.decompose$random + btc.decompose$seasonal, na.rm=TRUE))
strength_trend_btc
strength_season_btc

# Holt, HW, ARIMA
btc.holt = holt(btc)
accuracy(btc.holt)
#RMSE = 12.80882

btc.hw = hw(btc)
accuracy(btc.hw)
#RMSE = 12.77784

btc.arima = auto.arima(btc, stepwise = FALSE)
accuracy(btc.arima)
ggAcf(resid(btc.arima))
#RMSE = 12.80882

###################################################################
# VAR - Create a multivariate time series for BTC and Hash Rate 
###################################################################
prices.btc.nvdia = data.frame(btc_5days,nvdia)
mod_btc_nvdia = VAR(prices.btc.nvdia, p=1)
var.fcast.btc.nvdia <- predict(mod_btc_nvdia, n.ahead = 14)
vf.btc.nvdia <- data.frame(var.fcast.btc.nvdia$fcst)
RMSE <- sqrt(mean((vf.btc.nvdia[,5] - hash_test)^2))
RMSE  #RMSE: 116.2483
summary(mod_btc_nvdia)
ggAcf(resid(mod_btc_nvdia))

