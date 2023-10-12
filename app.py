import json
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, DateOffset
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import re
import os
import io
import boto3
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)

def lambda_handler(event, context):
    # Extract the 'messages' parameter from the JSON
    ticker = event['messages'].upper()
    print('ticker = ' + ticker)
    
    # Run modeling
    stock = Arima(ticker)
    forecast = stock.plot_forecast()
    
    summary = stock.summ()
    
    adf_fd, adf_secd, adf_sd, adf_sfd = stock.adf()

    timeseries = stock.plot_timeseries()
    fd, secd, sd, sfd = stock.plot_diff()
    resid = stock.plot_resid()
    acf_fd, acf_sfd, acf_auto_sfd, pacf_sfd = stock.plot_autocorrelation()
    
    bus, ind, web, dayhigh, daylow, dayopen, dayclose, ytdhigh, ytdlow, vol = stock.basics()
    
    # Return the data
    data = {
        'ticker': ticker,
        
        # # Plotted forecast
        'forecast': forecast,  
        
        # Stats
        'summary': summary,
        'adf_fd': adf_fd,
        'adf_secd': adf_secd,
        'adf_sd': adf_sd,
        'adf_sfd': adf_sfd,
        
        'images': {
            # Timeseries Plot
            'timeseries': timeseries,
            # Differencing Plot
            'fd': fd,
            'secd': secd,
            'sd': sd,
            'sfd': sfd,
            # Residual Plot
            'resid': resid,
            # Autocorrelation Plots
            'acf_fd': acf_fd, 
            'acf_sfd': acf_sfd, 
            'acf_auto_sfd': acf_auto_sfd, 
            'pacf_sfd': pacf_sfd
        },
        
        'basics': {
            'bus': bus,
            'ind': ind,
            'web': web,
            'dayhigh': dayhigh,
            'daylow': daylow,
            'dayopen': dayopen,
            'dayclose': dayclose,
            'ytdhigh': ytdhigh,
            'ytdlow': ytdlow,
            'vol': vol
        }
    }
    
    return {
        'body': json.dumps(data),
        'statusCode': 200,
    }

class Arima():
    def __init__(self, ticker, forecast_timeframe=252, show=False, options=False):
        # Global Constants
        self.FD = 'First Difference'
        self.SECD ='Second Difference'
        self.SD = 'Seasonal Difference'
        self.SFD = 'Seasonal First Difference'
        self.now = datetime.datetime.now().strftime('%Y-%m-%d')
        self.public = 'public'
        
        # S3
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket('automodel')
        self.region = 'us-west-1'
        
        # Global Variables
        self.ticker = ticker.upper()
        self.show = show
        self.options = options 
        self.forecast_timeframe = forecast_timeframe
        
        # Global Dataframe
        self.df, self.time_series = self.setup(ticker)
        self.df, self.adf_fd, self.adf_secd, self.adf_sd, self.adf_sfd = self.differencing(self.df)
        self.df, self.results = self.model(self.df)
        self.forecasted_df = self.forecast(self.df, self.forecast_timeframe)
        
    # Setups
    def setup(self, ticker):
        # Get stock data
        stock = yf.Ticker(ticker).history(period="10y")
        
        # Reset index to index rather than date
        stock = stock.reset_index()
        
        # Create new df with only date and close
        df = stock[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.date
        df.set_index('Date', inplace=True)

        # Create time series
        time_series = df['Close']
            
        return df, time_series
            
    def differencing(self, df):
        # Differencing: first difference, change from one period to the next
        df[self.FD] = df['Close'] - df['Close'].shift(1)
        adf_fd = self.adf_check(self.df[self.FD].dropna(), self.FD)

        # Second
        df[self.SECD] = df[self.FD] - df[self.FD].shift(1)
        adf_secd = self.adf_check(self.df[self.SECD].dropna(), self.SECD)
        
        # Seasonal
        df[self.SD] = df['Close'] - df['Close'].shift(30)
        adf_sd = self.adf_check(self.df[self.SD].dropna(), self.SD)
        
        # Seasonal First Difference
        df[self.SFD] = df[self.FD] - df[self.FD].shift(30)
        adf_sfd = self.adf_check(self.df[self.SFD].dropna(), self.SFD)
        
        return df, adf_fd, adf_secd, adf_sd, adf_sfd
       
    def model(self, df):
        # Need frequency for SARIMA model
        df.index.freq = 'D' # may or may not need
        
        # seasonal arima model
        model = sm.tsa.arima.ARIMA(df['Close'], order=(0, 1, 0), seasonal_order=(0, 1, 0, 365))
        
        # Fit model to see results
        results = model.fit(method='innovations_mle', low_memory=True, cov_type='none')
            
        return df, results
    
    def forecast(self, df, forecast_timeframe):
        # Reset the index of df
        df.reset_index(drop=False, inplace=True)
        
        # Forecast Length
        start = len(df)
        end = round(len(df) + forecast_timeframe)
        
        # Forecasting Current
        df['Forecast'] = self.results.predict(start=start, end=end)
        
        # Forecast future values, create and add time periods to do so
        start_date = df['Date'].max()
        # Create a CustomBusinessDay object to exclude weekends and holidays
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        # Create future dates
        future_dates = pd.date_range(start_date, periods=forecast_timeframe, freq=us_bd).strftime('%Y-%m-%d')
        
        df.set_index('Date', inplace=True)

        # Create a future dates df
        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
        future_df.index.name = 'Date'
        
        # Concatenate the original df with the future_df
        forecasted_df = pd.concat([df, future_df])
        forecasted_df.index.name = 'Date'
        
        forecasted_df.reset_index(inplace=True)
        
        forecasted_df['Forecast'] = self.results.predict(start=start, end=end)
        
        return forecasted_df
       
    # Plots
    def plot_forecast(self):
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.forecasted_df.set_index('Date', inplace=True)
        self.forecasted_df[['Close', 'Forecast']].plot(figsize=(12, 8))

        # X limit
        plt.xlim((len(self.df) - 252), len(self.forecasted_df))

        # Add a grid with dotted lines
        plt.grid(linestyle='--')

        # Choose the number of evenly spaced ticks you want (e.g., 12)
        num_ticks = 12

        # Calculate tick positions
        tick_positions = np.linspace((len(self.df) - 252), (len(self.forecasted_df) - 1), num_ticks, dtype=int)

        # Get the corresponding dates for the tick positions
        tick_labels = self.forecasted_df.index[tick_positions]

        # Set the ticks and labels
        plt.xticks(tick_positions, tick_labels, rotation=45)  # Rotate labels for better visibility

        # Labels
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title(f'Forecasted Price of {self.ticker}')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning

        forecast = f'{self.public}/plots/forecast/{self.ticker}_{self.now}_arima_forecast.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=forecast)
        
        # Close the current figure
        plt.close()
        
        return f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{forecast}'
    
    def plot_timeseries(self): 
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        # Create rolling mean
        self.time_series.rolling(252).mean().plot(label='252 Day Rolling Mean')
        self.time_series.plot()
        
        # Create rolling std
        self.time_series.rolling(21).std().plot(label='21 Day Rolling STD')
        self.time_series.plot()
        
        plt.legend(['252 Day Rolling Mean', '21 Day Rolling STD'])
        plt.title('Time Series Analysis')
        
        # ETS Plot
        decomp = seasonal_decompose(self.time_series, period=12)
        decomp.plot()
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)
        
        timeseries = f'{self.public}/plots/timeseries/{self.ticker}_{self.now}_timeseries.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=timeseries)
        
        plt.close()  # Close the current figure
        
        return f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{timeseries}'
    
    def plot_diff(self):
        '''
        Plot the first difference, second difference, seasonal difference, and seasonal first difference.
        '''
        # First Difference
        fd = self.fd_plot()
        
        # Second Difference
        secd = self.secd_plot()
        
        # Seasonal
        sd = self.sd_plot()
        
        # Seasonal First Difference
        sfd = self.sfd_plot()
        
        return f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{fd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{secd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{sd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{sfd}'
    
    def plot_resid(self): 
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.results.resid.plot()
        self.results.resid.plot(kind='kde')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        resid = f'{self.public}/plots/resid/{self.ticker}_{self.now}_arima_resid.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=resid)
        
        plt.close()  # Close the current figure
        
        return f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{resid}'
        
    def plot_autocorrelation(self):
        acf_fd = self.acf_fd_plot()
        acf_sfd = self.acf_sfd_plot()
        acf_auto_sfd = self.acf_auto_sfd_plot()
        pacf_sfd = self.pacf_sfd_plot()
        
        return f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{acf_fd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{acf_sfd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{acf_auto_sfd}', f'https://{self.bucket}.s3.{self.region}.amazonaws.com/{pacf_sfd}'
    
    # Stats
    def summ(self):
        ''' 
        Create summary statistics for the model to load into DB.
        '''
        summary = str(self.results.summary())
        
        # Define regular expressions for the desired values
        patterns = {
            'Model': r'Model:\s+(.*?)\s+Log',
            'Log Likelihood': r'Log Likelihood\s+(\S+)',
            'Date': r'Date:\s+(.*?)\s+AIC',
            'AIC': r'AIC\s+(\S+)',
            'Time': r'Time:\s+(\S+)\s+BIC',
            'BIC': r'BIC\s+(\S+)',
            'Sample': r'Sample:\s+(\S+)\s+HQIC',
            'HQIC': r'HQIC\s+(\S+)',
            'sigma2': r'sigma2\s+(\S+)',
            'Ljung-Box (L1) (Q)': r'Ljung-Box \(L1\) \(Q\):\s+(\S+)',
            'Jarque-Bera (JB)': r'Jarque-Bera \(JB\):\s+(\S+)',
            'Prob(Q)': r'Prob\(Q\):\s+(\S+)',
            'Heteroskedasticity (H)': r'Heteroskedasticity \(H\):\s+(\S+)',
            'Skew': r'Skew:\s+(\S+)',
            'Prob(H) (two-sided)': r'Prob\(H\) \(two-sided\):\s+(\S+)',
            'Kurtosis': r'Kurtosis:\s+(\S+)',
        }
        
        # Initialize a dictionary to store extracted values
        extracted_data = {}
        
        # Iterate through the patterns and extract the data
        for key, pattern in patterns.items():
            match = re.search(pattern, summary)
            if match:
                extracted_data[key] = match.group(1)
                
        return extracted_data

    def adf(self):
        '''
        Just returns the adf results
        '''
        return self.adf_fd, self.adf_secd, self.adf_sd, self.adf_sfd
    
    def basics(self):
        basics = yf.Ticker(self.ticker)
        
        if basics.history_metadata['instrumentType'] == 'FUTURE':
            bus = None
            ind = None
            web = None
            dayhigh = basics.info['dayHigh']
            daylow = basics.info['dayLow']
            dayopen = basics.info['open']
            dayclose = basics.info['previousClose']
            ytdhigh = basics.info['fiftyTwoWeekHigh']
            ytdlow = basics.info['fiftyTwoWeekLow']
            vol = basics.info['averageVolume']
            
            return bus, ind, web, dayhigh, daylow, dayopen, dayclose, ytdhigh, ytdlow, vol
        
        else:
            bus = basics.info['longBusinessSummary']
            ind = basics.info['industry']
            web = basics.info['website']
            dayhigh = basics.info['dayHigh']
            daylow = basics.info['dayLow']
            dayopen = basics.info['open']
            dayclose = basics.info['previousClose']
            ytdhigh = basics.info['fiftyTwoWeekHigh']
            ytdlow = basics.info['fiftyTwoWeekLow']
            vol = basics.info['averageVolume']
            
            return bus, ind, web, dayhigh, daylow, dayopen, dayclose, ytdhigh, ytdlow, vol

    # Helpers    
    def adf_check(self, time_series, title):  
        # Stationarity
        result = adfuller(time_series)
        
        print(f"{title} Augmented Dicky-Fuller Test")
        
        # Initialize dictinoaries
        adf = {
            'symbol': self.ticker,
            'test': title,
            'date': self.now
        }
        
        stats = {
            'test_stat': '',
            'pvalue': '',
            'lags': '',
            'obs': ''
        }
        
        # Add values results
        labels = ['test_stat', 'pvalue', 'lags', 'obs']
        for label, value in zip(labels, result):
            # Check if the label exists in the stats dictionary
            if label in stats:
                # Assign the value to the corresponding key in the stats dictionary
                stats[label] = value
        
        # Return hypothesis       
        if result[1] <= 0.05:
            hyp = 'Strong evidence against null hypothesis. Reject null hypothesis. Data has no unit root and is stationary'
            res = {'hypothesis': hyp}
            
        else:
            hyp = 'Weak evidence against null hypothesis. Fail to reject null hypothesis. Data has a unit root and is non-stationary'
            res = {'hypothesis': hyp}
        
        # Add results to stats
        for k, v in res.items():
            stats[k] = v
        
        for k, v in stats.items():
            adf[k] = v
        
        return adf
    
    # Differencing Plot Helpers
    def fd_plot(self):
        ''' 
        First Difference
        '''
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.df[self.FD] = self.df['Close'].dropna() - self.df['Close'].dropna().shift(1)
        self.df[self.FD].dropna().plot()
        
        plt.title(self.FD)
        plt.xlabel('Date')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        fd = f'{self.public}/plots/diff/{self.ticker}_{self.now}_arima_{self.FD.lower().replace(" ", "_")}.png'

        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=fd)
        

        plt.close()  # Close the current figure
        
        return fd
          
    def secd_plot(self):
        ''' 
        Second Difference
        '''
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.df[self.SECD] = self.df[self.FD].dropna() - self.df[self.FD].dropna().shift(1)
        self.df[self.SECD].dropna().plot()
        
        plt.title(self.SECD)
        plt.xlabel('Date')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        secd = f'{self.public}/plots/diff/{self.ticker}_{self.now}_arima_{self.SECD.lower().replace(" ", "_")}.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=secd)
        
        plt.close()  # Close the current figure
        
        return secd
    
    def sd_plot(self):
        '''
        Seasonal Difference
        '''
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.df[self.SD] = self.df['Close'].dropna() - self.df['Close'].dropna().shift(30)
        self.df[self.SD].dropna().plot()
        
        plt.title(self.SD)
        plt.xlabel('Date')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        sd = f'{self.public}/plots/diff/{self.ticker}_{self.now}_arima_{self.SD.lower().replace(" ", "_")}.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=sd)
        
        plt.close()  # Close the current figure
        
        return sd
        
    def sfd_plot(self):
        '''
        Seasonal First Difference
        '''
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        self.df[self.SFD] = self.df[self.FD] - self.df[self.FD].shift(30)
        self.df[self.SFD].plot()
        
        plt.title(self.SFD)
        plt.xlabel('Date')
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        sfd = f'{self.public}/plots/diff/{self.ticker}_{self.now}_arima_{self.SFD.lower().replace(" ", "_")}.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=sfd)

        plt.close()  # Close the current figure
        
        return sfd
        
    # Autocorrelation Plot Helpers
    def acf_fd_plot(self):
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        plot_acf(self.df[self.SFD].dropna())
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        acf_fd = f'{self.public}/plots/autocorrelation/{self.ticker}_{self.now}_arima_{self.SFD.lower().replace(" ", "_")}_acf.png'

        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=acf_fd)
        
        plt.close()  # Close the current figure
        
        return acf_fd
        
    def acf_sfd_plot(self):
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        plot_acf(self.df[self.FD].dropna())
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        acf_sfd = f'{self.public}/plots/autocorrelation/{self.ticker}_{self.now}_arima_{self.FD.lower().replace(" ", "_")}_acf.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=acf_sfd)
        
        plt.close()  # Close the current figure
        
        return acf_sfd
    
    def acf_auto_sfd_plot(self):
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        autocorrelation_plot(self.df[self.SFD].dropna())
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        acf_auto_sfd = f'{self.public}/plots/autocorrelation/{self.ticker}_{self.now}_arima_{self.SFD.lower().replace(" ", "_")}_autocorrelation.png'
        
        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=acf_auto_sfd)
        
        plt.close()  # Close the current figure
        
        return acf_auto_sfd
    
    def pacf_sfd_plot(self):
        # Create a BytesIO stream to hold the plot data
        plot_stream = io.BytesIO()
        plt.figure()
        
        plot_pacf(self.df[self.SFD].dropna())
        
        plt.savefig(plot_stream, bbox_inches='tight')
        plot_stream.seek(0)  # Reset the stream position to the beginning
        
        pacf_sfd = f'{self.public}/plots/autocorrelation/{self.ticker}_{self.now}_arima_{self.FD.lower().replace(" ", "_")}_pacf.png'

        # Upload to S3
        self.bucket.put_object(Body=plot_stream, ContentType='image/png', Key=pacf_sfd)

        plt.close()  # Close the current figure
        
        return pacf_sfd
    
# Add more models below 
if __name__ == '__main__':
    lambda_handler({'messages': 'a'}, None)