import streamlit as st
import mysql.connector
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas_datareader as data
from bs4 import BeautifulSoup
import streamlit as st
from urllib.request import urlopen
from urllib.request import Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


st.set_page_config(
   page_title="Real-Time Data Science Dashboard",
   page_icon="✅",
   layout="wide",
)

st.markdown("<h1 style='text-align: center; color:grey;'>Stock Price Analysis</h1>", unsafe_allow_html=True)


NASDAQ = si.tickers_nasdaq()
DOW_JONES = si.tickers_dow()
OTHER = si.tickers_other()
SnP = si.tickers_sp500()

BSE = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O'
                   '%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]

BSE = BSE.SYMBOL.to_list()

for count in range(len(BSE)):
    BSE[count] = BSE[count] + ".NS"

ex_data = ["NASDAQ", "DOW_JONES", "SnP", "BSE", "OTHER"]

exchange = st.sidebar.selectbox(label="Select exchange", options=ex_data)

if exchange == "NASDAQ":
    company_name = st.sidebar.selectbox(label="Enter company name", options=NASDAQ)

elif exchange == "DOW_JONES":
    company_name = st.sidebar.selectbox(label="Enter company name", options=DOW_JONES)

elif exchange == "SnP":
    company_name = st.sidebar.selectbox(label="Enter company name", options=SnP)

elif exchange == "BSE":
    company_name = st.sidebar.selectbox(label="Enter company name", options=BSE)

else:
    company_name = st.sidebar.selectbox(label="Enter company name", options=OTHER)


start_date = st.sidebar.date_input("enter start date:")
end_date = st.sidebar.date_input("enter end date:")

data = data.DataReader(company_name, 'yahoo',
                       start=start_date,
                       end=end_date)


information = yf.Ticker(company_name)

col1, col2, col3, col4 = st.columns([5,13,12,11])

information4 = information.info['logo_url']
col1.image(information4,width=80)

information2 = information.info['longName']
col2.header(information2)
col3.header("- " + company_name)

with col4:
    tickers = company_name

    finviz_url = 'http://finviz.com/quote.ashx?t='
    news_tables = {}
    if type(tickers) == str:
        try:
            url = finviz_url + tickers
            req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
            resp = urlopen(req)
            html = BeautifulSoup(resp, features="lxml")
            news_table = html.find(id='news-table')
            news_tables[tickers] = news_table
        except:
            st.write("Data Not Found")
            os._exit(1)

    df = news_tables[tickers]
    df_tr = df.findAll('tr')

    for i, table_row in enumerate(df_tr):
        a_text = table_row.a.text
        td_text = table_row.td.text
        td_text = td_text.strip()
        break

    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, time, text])
    print("pars new", parsed_news)

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)

    news = pd.DataFrame(news)
    print(news)

    analyzer = SentimentIntensityAnalyzer()
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()
    df_scores = pd.DataFrame(scores)

    news = news.join(df_scores, rsuffix='_right')
    print(news)

    news['Date'] = pd.to_datetime(news.Date).dt.date
    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    # Pre Mean
    pre_news = news.iloc[1:, :]
    print("pre news", pre_news)
    pre_news['Date'] = pd.to_datetime(pre_news.Date).dt.date
    unique_ticker = pre_news['Ticker'].unique().tolist()
    pre_news_dict = {name: pre_news.loc[pre_news['Ticker'] == name] for name in unique_ticker}

    values = []
    pre_values = []

    dataframe1 = news_dict[ticker]
    dataframe1 = dataframe1.set_index('Ticker')
    dataframe1 = dataframe1.drop(columns=['Headline'])

    dataframe = pre_news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns=['Headline'])

    mean = round(dataframe1['compound'].mean(), 3)
    values.append(mean)

    pre_mean = round(dataframe['compound'].mean(), 3)
    pre_values.append(pre_mean)

    print(values)
    print(pre_values)
    diff = round(pre_mean - mean, 3)

    st.metric(label="Mean Sentiment", value=mean, delta=diff,
              delta_color="off")

information1 = information.info['sector']
st.write("Sector :-", information1)

information3 = information.info['longBusinessSummary']
st.text_area("Summary", information3)

data = data.reset_index()
data.columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj_Close']

data.Date = pd.to_datetime(data.Date)
data = data.set_index(data.Date)

st.markdown("### Data History")
st.write(data)

st.markdown('### Price Chart')
fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])
st.plotly_chart(fig,use_container_width=True)

df_close = data.Close


col1,col2 = st.columns(2)


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
    output = pd.Series(adft[0:4],index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)
test_stationarity(df_close)

# ----To separate the trend and the seasonality from a time series----#

#col2.markdown('Seasonal Decompose of {}'.format(company_name))
df_close = data[["Close"]].copy()
result = seasonal_decompose(df_close, model='multiplicative', period=1)
fig = plt.figure(figsize=(8,6))
fig = result.plot()
#col2.pyplot(fig)


data.Close = pd.to_numeric(data.Close)
value = data.Close



#------Marlo carlo simulation-----#

containor1 = st.container()
coll1, coll2 = st.columns(2)
with containor1:
    with coll1:
        coll1.markdown('### Moving Average')
        moving_average = df_close.rolling(window=2).mean()
        print(moving_average)
        coll1.line_chart(moving_average)

    with coll2:
        train_data, test_data = value[3:int(len(value) * 0.9)], value[int(len(value) * 0.9):]
        coll2.markdown('### Auto ARIMA Model')
        model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                     test='adf',  # use adftest to find optimal 'd'
                                     max_p=3, max_q=3,  # maximum p and q
                                     m=1,  # frequency of series
                                     d=None,  # let model determine 'd'
                                     seasonal=False,  # No Seasonality
                                     start_P=0,
                                     D=0,
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)
        model_autoARIMA.plot_diagnostics()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        coll2.pyplot()

        mygraph = data[['Close']]

containor2 = st.container()
col1, col2 = st.columns(2)
with containor2:
    with col1:
        train_data, test_data = value[3:int(len(value) * 0.9)], value[int(len(value) * 0.9):]
        model = sm.tsa.statespace.SARIMAX(train_data, order=(1, 1, 2))
        model = model.fit()
        start_value = len(train_data)  # end of train data it is starting prediction
        end_value = len(train_data) + len(test_data) - 1
        pred = model.predict(start=start_value, end=end_value, type='levels')
        pred.index = data.index[start_value:end_value + 1]

        col1.markdown('### Final Stock Price')
        plt.plot(pred, label='Prediction', color='red')
        plt.plot(test_data, label='Test', color='yellow')
        plt.plot(train_data, label='Train', color='blue')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=9)
        col1.pyplot()

    with col2:
        adj_data = data["Close"][-1]

        T = 10  # no. of trading days
        mu = 0.2309  # Return
        vol = 0.4259  # Volatility
        for i in range(50):
            daily_returns = np.random.normal(mu / T, vol / math.sqrt(T),
                                             T) + 1  # list of daily returns using random normal distribution
            price_list = [adj_data]

            for x in daily_returns:
                price_list.append(price_list[-1] * x)
            plt.plot(price_list)
        col2.markdown('### Monte Carlo')
        col2.pyplot()

