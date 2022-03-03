import pandas as pd
import mysql.connector
from mysql import connector
import os
import sqlalchemy as sq

my_conn = sq.create_engine("mysql+pymysql://Optima_Internal:Optima$123@208.91.199.96/Optima_FinData")

exchange_data = pd.read_csv('C:/Users/malvi/Downloads/ISO10383_MIC.csv')

exchange_data = exchange_data[['ISO COUNTRY CODE (ISO 3166)', 'MIC', 'NAME-INSTITUTION DESCRIPTION', 'ACRONYM']]
exchange_data.rename(columns={'ISO COUNTRY CODE (ISO 3166)': 'country_code',
                              'MIC': 'code',
                              'NAME-INSTITUTION DESCRIPTION': 'name',
                              'ACRONYM': 'acronym'}, inplace=True)
exchange_data['id'] = exchange_data.index

#mapper = {'US': 'USD', 'GB': 'GBP', 'DE': 'EUR', 'CA': 'CAD', 'NO': 'EUR', 'ES': 'EUR', 'BG': 'EUR', 'AU': 'AUD' }
exchange_data['currency'] = exchange_data['country_code']
a = exchange_data[['id', 'name', 'currency', 'code']]
print(a)

a.to_sql("exchange_malvi", my_conn, if_exists="append", index=False)

