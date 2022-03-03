from mysql import connector


class dbHealper:
    def __init__(self):
        self.con = connector.connect(host="208.91.199.96", user="Optima_Intern", passwd="Optima@123",
                                     database="Optima_FinData")

        tc1 = "create table IF NOT EXISTS exchange_malvi(id int primary key,name varchar(15),currency varchar(15),code varchar(15))"
        cur = self.con.cursor()
        cur.execute(tc1)

        tc2 = """CREATE TABLE IF NOT EXISTS company_malvi(
                                            id integer ,
                                            name text NOT NULL,
                                            industry text,
                                            sector text,
                                            hq_location text,
                                            security_id integer,
                                            PRIMARY KEY(id))"""
        cur = self.con.cursor()
        cur.execute(tc2)

        tc3 = """CREATE TABLE IF NOT EXISTS security_malvi (
                                           id integer,
                                           Ticker text NOT NULL,
                                           name text NOT NULL,
                                           company_id integer,
                                           exchange_id integer,
                                           PRIMARY KEY(id),
                                           FOREIGN KEY (company_id) REFERENCES company_malvi (id)
                                       );"""
        cur = self.con.cursor()

        cur.execute(tc3)

        tc4 = """CREATE TABLE IF NOT EXISTS security_price_malvi (
                                    id integer,
                                    date text NOT NULL,
                                    open decimal NOT NULL,
                                    high decimal NOT NULL,
                                    low decimal NOT NULL,
                                    close decimal NOT NULL,
                                    volume integer,
                                    adj_close decimal NOT NULL,
                                    security_id integer,
                                    PRIMARY KEY(id),
                                    FOREIGN KEY (security_id) REFERENCES security_malvi (id)
                                );"""

        cur = self.con.cursor()
        cur.execute(tc4)


print("Done")

helper = dbHealper()