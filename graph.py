import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
import dash
from dash import dcc, html
import plotly.graph_objects as go


mydb = mysql.connector.connect(host="208.91.199.96", user="Optima_Internal", passwd="Optima$123",
                                     database="Optima_FinData")
mycursor = mydb.cursor()
data = int(input("enter security_id :"))
a = mycursor.execute("select * from security_price_malvi sp inner join security_malvi sm on sm.id = sp.security_id where "
                 "sp.security_id = %s", (data,))

mygraph = mycursor.fetchall()
mygraph = pd.DataFrame(mygraph)
mygraph.columns = ['id','date','high','low','open','close','volume','adj_close','security_id','id','ticker', 'name','company_id']
print(mygraph)

mygraph.adj_close = pd.to_numeric(mygraph.adj_close)
mygraph.date=pd.to_datetime(mygraph.date)
mygraph = mygraph.set_index(mygraph.date)
mygraph['adj_close'].plot()
plt.xlabel("Date")
plt.ylabel("price values")

# candlestick Pattern

fig = go.Figure(data=[go.Candlestick(x=mygraph['date'],
                open=mygraph['open'],
                high=mygraph['high'],
                low=mygraph['low'],
                close=mygraph['close'])])

dcc.Graph(figure=fig)
fig.show()

# Moving Average

moving_average = mygraph.close
moving_average = moving_average.rolling(window=20).mean()
print(moving_average)

moving_average.plot()
mygraph["adj_close"].plot()
plt.show()

