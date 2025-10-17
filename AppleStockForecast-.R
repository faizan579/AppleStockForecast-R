# ============================
# 🚀 Forecasting Apple Stock Prices Using Holt-Winters & ARIMA
# ============================

# 🛠 Install required packages (if not already installed)
install.packages(c("quantmod", "forecast", "tseries"))  # Run only if packages are not installed

# 📚 Load necessary libraries
library(quantmod)   # For downloading stock data
library(forecast)   # For time series forecasting
library(tseries)    # For statistical tests

# ============================
# 📌 Step 1: Download & Prepare Stock Data
# ============================

# Define stock symbol and time range
stock_symbol <- "AAPL"         # Apple stock ticker
start_date <- "2019-01-01"     # Start date for data extraction
end_date <- "2024-12-31"       # End date for data extraction

# Fetch stock data from Yahoo Finance
getSymbols(stock_symbol, src = "yahoo", from = start_date, to = end_date)

# Convert daily data into monthly data (last trading day of each month)
aapl_monthly <- to.monthly(AAPL, indexAt = "lastof", OHLC = FALSE)

# Extract the Adjusted Closing Price (4th column of the matrix)
aapl_close <- aapl_monthly[, 4]

# Convert the adjusted closing price into a time series object
aapl_ts <- ts(aapl_close, start = c(2019, 1), frequency = 12)

# 🖼️ Plot the time series data
plot(aapl_ts, main = "Apple Monthly Stock Prices (2019-2024)",
     col = "blue", ylab = "Price (USD)", xlab = "Year")

# ============================
# 📌 Step 2: Apply Holt-Winters (Exponential Smoothing)
# ============================

# Fit Holt-Winters model with additive trend and seasonality
hw_model <- HoltWinters(aapl_ts, seasonal = "additive")

# Forecast the next 12 months (2025)
hw_forecast <- forecast(hw_model, h = 12)

# ============================
# 📌 Step 3: Apply ARIMA Model
# ============================

# Check if the time series is stationary using the Augmented Dickey-Fuller test
adf_test <- adf.test(aapl_ts)  # If p-value > 0.05, differencing is needed

# Perform first-order differencing if needed
diff_ts <- diff(aapl_ts, differences = 1)

# Fit an ARIMA model (automatically selects the best parameters)
auto_arima_model <- auto.arima(aapl_ts)

# Forecast the next 12 months using the ARIMA model
arima_forecast <- forecast(auto_arima_model, h = 12)

# ============================
# 📌 Step 4: Visualizing the Forecasts
# ============================

# Create side-by-side plots for easy comparison
par(mfrow = c(2, 1))  # Set up 2 rows, 1 column for plotting

# 📊 Plot Holt-Winters Forecast
plot(hw_forecast, main = "Holt-Winters Forecast for AAPL", col = "red")

# 📊 Plot ARIMA Forecast
plot(arima_forecast, main = "ARIMA Forecast for AAPL", col = "green")

# ============================
# 📌 Step 5: Forecast Evaluation (RMSE, MAE, MAPE) - With Fix
# ============================

# Extract actual values for the last 12 months (2024) to match forecasted period
actual_values <- window(aapl_ts, start = c(2024, 1))

# Ensure lengths match before computing accuracy
hw_accuracy <- accuracy(hw_forecast$fitted, actual_values)
arima_accuracy <- accuracy(arima_forecast$fitted, actual_values)

# Print model evaluation results
cat("📈 Holt-Winters Model Accuracy:\n")
print(hw_accuracy)

cat("\n📉 ARIMA Model Accuracy:\n")
print(arima_accuracy)

# ============================
# 📌 Step 6: Compare Models Using RMSE, MAE, and MAPE
# ============================

# Extract accuracy metrics for comparison
hw_rmse <- hw_accuracy["RMSE"]
arima_rmse <- arima_accuracy["RMSE"]

hw_mae <- hw_accuracy["MAE"]
arima_mae <- arima_accuracy["MAE"]

hw_mape <- hw_accuracy["MAPE"]
arima_mape <- arima_accuracy["MAPE"]

# Print the comparison results
cat("\n🔍 **Comparison of Forecasting Models:**\n")
cat(sprintf("RMSE: Holt-Winters = %.2f, ARIMA = %.2f\n", hw_rmse, arima_rmse))
cat(sprintf("MAE: Holt-Winters = %.2f, ARIMA = %.2f\n", hw_mae, arima_mae))
cat(sprintf("MAPE: Holt-Winters = %.2f%%, ARIMA = %.2f%%\n", hw_mape, arima_mape))

# ============================
# 📌 Interpretation of Results
# ============================

# If RMSE, MAE, and MAPE are lower for one model, it means that model performed better.
# - Lower RMSE → Less overall error in predictions
# - Lower MAE → Smaller average error in absolute terms
# - Lower MAPE → More accurate percentage-based predictions

# Based on the output, we can determine which model (Holt-Winters or ARIMA) is more effective.
