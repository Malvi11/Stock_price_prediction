import streamlit as st
import mysql.connector
import plotly.graph_objects as go
import pandas as pd


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

    company_name = st.selectbox(label="Enter company name",options=data)
    mycursor.execute("select * from security_price_malvi sp inner join security_malvi sm on sm.id = sp.security_id where "
                     "sm.name = %s", (company_name,))

    start = st.date_input('Start', value=pd.to_datetime('2020-01-01'))
    end = st.date_input('End', value=pd.to_datetime('2022-02-05'))

mygraph = mycursor.fetchall()
mygraph = pd.DataFrame(mygraph)
mygraph.columns = ['id','date','high','low','open','close','volume','adj_close','security_id','id','ticker', 'name','company_id']

mygraph.volume = pd.to_numeric(mygraph.volume)
mygraph.adj_close = pd.to_numeric(mygraph.adj_close)

mygraph.date=pd.to_datetime(mygraph.date)
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



