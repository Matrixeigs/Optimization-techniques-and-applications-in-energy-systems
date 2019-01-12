"""
Database management for the mess scheduling with active distribution systems
1) Table create and delete
2) Data storage and management
@date:12-Jan-2019

"""
import pymysql


class DataBaseManagement():

    def __init__(self, host="localhost", user="root", password="1", db="mess"):
        """
        Initialized the database connection string
        :param host: host ip
        :param user: user name
        :param password: password
        :param db: database name
        :return
        """
        self.db = pymysql.connect(host, user, password, db)

    def create_table(self, table_name, nl=32, nb=33, ng=6):
        """
        Creat table name
        :param table_name:
        :param nb:
        :param nb:
        :param ng:
        :return: no return value
        """
        cursor = self.db.cursor()
        sql = "DROP TABLE IF EXISTS "
        cursor.execute(sql + table_name)
        if table_name == "distribution_networks":
            sql_start = """CREATE TABLE distribution_networks ("""
            sql = 'SCENARIO  INT primary key,\n TIME INT NOT NULL,\n '
            for i in range(nl):
                sql += "PIJ{0} FLOAT,\n ".format(i)
            for i in range(nl):
                sql += "QIJ{0} FLOAT,\n ".format(i)
            for i in range(nl):
                sql += "IIJ{0} FLOAT,\n ".format(i)
            for i in range(nb):
                sql += "V{0} FLOAT,\n ".format(i)
            for i in range(ng):
                sql += "PG{0} FLOAT,\n ".format(i)
            for i in range(ng-1):
                sql += "QG{0} FLOAT,\n ".format(i)
            sql += "QG{0} FLOAT\n ".format(ng - 1)
            sql_start_end = """)"""
            cursor.execute(sql_start + sql + sql_start_end)

    def insert_data(self, table_name, nl=32, nb=33, ng=6, scenario=0, time=0, pij=0, qij=0, lij=0, vi=0, pg=0, qg=0):
        """
        Insert data into table_name
        :param table_name:
        :param nl:
        :param nb:
        :param ng:
        :param pij:
        :param qij:
        :param lij:
        :param vi:
        :param pg:
        :param qg:
        :return:
        """
        cursor = self.db.cursor()
        sql_start = "INSERT INTO "+ table_name +" ("
        sql = "SCENARIO,TIME,"
        value = "{0},{1},".format(scenario, time)
        for i in range(nl):
            sql += "PIJ{0},".format(i)
            value += "{0},".format(pij[i])
        for i in range(nl):
            sql += "QIJ{0},".format(i)
            value += "{0},".format(qij[i])
        for i in range(nl):
            sql += "IIJ{0},".format(i)
            value += "{0},".format(lij[i])
        for i in range(nb):
            sql += "V{0},".format(i)
            value += "{0},".format(vi[i])
        for i in range(ng):
            sql += "PG{0},".format(i)
            value += "{0},".format(pg[i])
        for i in range(ng - 1):
            sql += "QG{0},".format(i)
            value += "{0},".format(qg[i])
        sql += "QG{0}".format(ng - 1)
        value += "{0}".format(qg[ng - 1])

        sql += ") VALUES (" + value + ")"

        cursor.execute(sql_start + sql)
        self.db.commit()