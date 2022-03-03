import mysql.connector
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas_datareader.data as web
import sqlalchemy as sq


my_conn = sq.create_engine("mysql+pymysql://Optima_Internal:Optima$123@208.91.199.96/Optima_FinData")

data1 = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components')[1]
data1.rename(columns={'Company': 'name', 'Industry': 'industry', 'Symbol': 'ticker'}, inplace=True)

data1['ticker'] = data1['ticker'].apply(lambda x: x[::-1].partition(':')[0][::-1])
data1['ticker'] = data1['ticker'].str.strip()

data2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
data2.rename(columns={'Symbol': 'ticker', 'Security': 'name', 'Headquarters Location': 'hq_location',
                      'GICS Sector': 'sector', 'GICS Sub-Industry': 'industry'}, inplace=True)

company_table = pd.concat([data2[['name', 'industry', 'sector', 'hq_location']], data1[['name', 'industry']]])
company_table.drop_duplicates('name', inplace=True)
company_table.sort_values('name', inplace=True)
company_table.reset_index(inplace=True, drop=True)
company_table['id'] = company_table.index

sp_security_table = data2[['ticker', 'name']].copy()

dj_security_table = data1[['ticker', 'name']].copy()

security_table = pd.concat([sp_security_table, dj_security_table])
security_table.drop_duplicates(subset='ticker', inplace=True)
security_table.sort_values('ticker', inplace=True)
security_table.reset_index(inplace=True, drop=True)
security_table['id'] = security_table.index

company_id_mapper = pd.Series(company_table.id.values, index=company_table.name)

security_table['company_id'] = security_table['name'].map(company_id_mapper)

security_id_mapper = pd.Series(security_table.id.values, index=security_table.name).to_dict()

company_table['security_id'] = company_table['name'].map(security_id_mapper)


c = security_table[['id', 'ticker', 'name', 'company_id']]

#c.to_sql("security_malvi", my_conn, if_exists="append", index=False)


stock_pricing_dfs = []
for stock_id in tqdm(c['id']):
    try:
        stock_pricing_df = web.DataReader(c.iloc[stock_id]['ticker'],
                           start='2020-01-01',
                           end='2022-02-05',
                           data_source='yahoo')
        stock_pricing_df['security_id'] = stock_id
        stock_pricing_dfs.append(stock_pricing_df)
    except:
        pass
security_price_table = pd.concat(stock_pricing_dfs)
security_price_table.columns = ['high', 'low', 'open', 'close', 'volume', 'adj_close', 'security_id']
security_price_table.reset_index(inplace=True)
security_price_table['id'] = security_price_table.index
print(security_price_table)

security_price_table.to_sql("security_price_malvi", my_conn, if_exists="append", index=False)
