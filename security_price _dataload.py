import mysql.connector
from tqdm import tqdm_notebook as tqdm
import pandas_datareader
stock_pricing_dfs = []

mydb = mysql.connector.connect(host="208.91.199.96", user="Optima_Internal", passwd="Optima$123",
                               database="Optima_FinData")

mycursor = mydb.cursor()

stock_pricing_dfs = []
for stock_id in tqdm(Final_security_table['id']):
    try:
        stock_pricing_df = web.DataReader(Final_security_table.iloc[stock_id]['ticker'],
                           start='2020-1-1',
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
