import streamlit as st
import mysql.connector
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.arima_model import ARIMA

st.title("Build and Deploy Stock Market App Using Streamlit")
st.header("A Basic Data Science Web Application")

mydb = mysql.connector.connect(host="208.91.199.96", user="Optima_Internal", passwd="Optima$123",
                               database="Optima_FinData")
mycursor = mydb.cursor()

with st.sidebar:
    st.subheader("Configure the plot")
    mycursor = mydb.cursor()
    mycursor.execute("select name from company_malvi")
    data = mycursor.fetchall()
    data = pd.DataFrame(data)

    company_name = st.selectbox(label="Enter company name", options=data)
    mycursor.execute(
        "select * from security_price_malvi sp inner join security_malvi sm on sm.id = sp.security_id where "
        "sm.name = %s", (company_name,))

    start = st.date_input('Start', value=pd.to_datetime('2020-01-01'))
    end = st.date_input('End', value=pd.to_datetime('2022-02-05'))

mygraph = mycursor.fetchall()
mygraph = pd.DataFrame(mygraph)
mygraph.columns = ['id', 'date', 'high', 'low', 'open', 'close', 'volume', 'adj_close', 'security_id', 'id', 'ticker',
                   'name', 'company_id']

# ----To separate the trend and the seasonality from a time series----#

st.header('Seasonal Decompose of {}'.format(company_name))
df = mygraph[["close"]].copy()
result = seasonal_decompose(df, model='multiplicative', period=1)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)
st.pyplot(fig)

mygraph.volume = pd.to_numeric(mygraph.volume)
mygraph.adj_close = pd.to_numeric(mygraph.adj_close)

mygraph.date = pd.to_datetime(mygraph.date)
mygraph = mygraph.set_index(mygraph.date)

st.header('Candlestick Chart of {}'.format(company_name))
fig = go.Figure(data=[go.Candlestick(x=mygraph['date'],
                                     open=mygraph['open'],
                                     high=mygraph['high'],
                                     low=mygraph['low'],
                                     close=mygraph['close'])])
st.plotly_chart(fig)

st.header('Moving Average of {}'.format(company_name))
moving_average = mygraph.adj_close
moving_average = moving_average.rolling(window=50).mean()
print(moving_average)
st.line_chart(moving_average)


# ----Test for stationary----#

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    print("Results of dickey fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)

test_stationarity(df)

# ---split data into train and training set---#

train_data, test_data = moving_average[3:int(len(moving_average) * 0.9)], moving_average[
                                                                          int(len(moving_average) * 0.9):]
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(moving_average, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#---Auto_ARIMA Model----#
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                             test='adf',        # use adftest to find optimal 'd'
                             max_p=3, max_q=3,  # maximum p and q
                             m=1,               # frequency of series
                             d=None,            # let model determine 'd'
                             seasonal=False,    # No Seasonality
                             start_P=0,
                             D=0,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()
st.pyplot(model_autoARIMA)
