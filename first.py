import os
from aifc import Error

import mysql.connector
from mysql import connector

mydb = mysql.connector.connect(host="208.91.199.96",user="Optima_Intern",passwd="Optima@123")

def create_connection(mydb):
    """ create a database connection to the mysql database
            specified by mydb
        :param mydb: database file
        :return: Connection object or None
        """
    conn = None
    try:
        conn = connector.connect(mydb)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn,create_table_mysql):
    """ create a table from the create_table_mysql statement
       :param conn: Connection object
       :param create_table_mysql: a CREATE TABLE statement
       :return:
       """

    try:
        c = conn.cursor()
        c.execute(create_table_mysql)
    except Error as e:
        print(e)

db_name = 'Optima_FinData'




def main():
    database = os.path.join('..','data',db_name)

    mysql_create_exchange_table = """ CREATE TABLE IF NOT EXISTS exchange_malvi (
                                         integer ,
                                        name text NOT NULL,
                                        currency,
                                        code text NOT NULL UNIQUE,
                                        PRIMARY KEY(id)
                                    ); """
    mysql_create_company_table = """CREATE TABLE IF NOT EXISTS company_malvi (
                                    integer ,
                                    name text NOT NULL,
                                    industry text,
                                    sector text,
                                    hq_location text,
                                    security_id integer,
                                    PRIMARY KEY(id),
                                );"""
    mysql_create_security_table = """CREATE TABLE IF NOT EXISTS security_malvi (
                                   integer,
                                   ticker text NOT NULL UNIQUE,
                                   name text NOT NULL,
                                   company_id integer,
                                   exchange_id integer,
                                   PRIMARY KEY(id)
                                   FOREIGN KEY (company_id) REFERENCES company_malvi,
                                   FOREIGN KEY (exchange_id) REFERENCES exchange_malvi
                               );"""
    mysql_create_security_price_table = """CREATE TABLE IF NOT EXISTS security_price_malvi (
                            integer,
                            date text NOT NULL,
                            open decimal NOT NULL,
                            high decimal NOT NULL,
                            low decimal NOT NULL,
                            close decimal NOT NULL,
                            volume integer,
                            adj_close decimal NOT NULL,
                            security_malvi integer,
                            PRIMARY KEY(id),
                            FOREIGN KEY (security_id) REFERENCES security_malvi
                        );"""

    conn = create_connection(mydb)

    if conn is not None:

        create_table(conn,mysql_create_company_table)
        create_table(conn,mysql_create_exchange_table)
        create_table(conn,mysql_create_security_table)
        create_table(conn,mysql_create_security_price_table)

    else:

        print("Error! cannot create the database connection.")
