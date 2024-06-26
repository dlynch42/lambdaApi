## Finarima API

The Finarima API is a powerful tool for generating ARIMA (AutoRegressive Integrated Moving Average) models and associated plots for a given security and time period. This API is designed to provide users with access to predictive modeling capabilities and visualizations, enabling them to analyze historical stock data and make informed decisions.

Upon receiving the request, the API will process the data, generate the ARIMA model, and produce various plots, including forecasted price trends, time series analysis, differencing plots, residual plots, and autocorrelation plots. 

The API will then return a JSON response containing the following data:

* ticker: The ticker symbol of the security.
* forecast: URL of the plotted forecast.
* summary: Summary statistics of the ARIMA model.
* adf_fd: Augmented Dickey-Fuller test results for first difference.
* adf_secd: Augmented Dickey-Fuller test results for second difference.
* adf_sd: Augmented Dickey-Fuller test results for seasonal difference.
* adf_sfd: Augmented Dickey-Fuller test results for seasonal first difference.
* images: URLs of various plots generated by the API, including timeseries plot, differencing plots, residual plot, and autocorrelation plots.
* basics: Basic information about the security, including business summary, industry, website, daily high, daily low, opening price, closing price, year-to-date high, year-to-date low, and average volume.

## Technologies Used

* Python: The API is written in Python, utilizing libraries such as NumPy, Pandas, Statsmodels, Matplotlib, and others for data processing, modeling, and visualization.
* Docker: Docker was used to create the container and then deploy it to ECS, where it is used by Lambda. The Dockerfile shows the steps to create it.
* AWS: The API is deployed on Amazon Web Services (AWS) using Lambda functions, with data storage on Amazon S3.
* yfinance: The yfinance library is used to retrieve historical stock data for modeling.
