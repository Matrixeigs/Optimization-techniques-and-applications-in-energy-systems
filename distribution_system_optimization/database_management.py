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
            sql = 'SCENARIO  INT,\n TIME INT NOT NULL,\n '
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
            for i in range(ng - 1):
                sql += "QG{0} FLOAT,\n ".format(i)
            sql += "QG{0} FLOAT\n ".format(ng - 1)
            sql_start_end = """)"""
        elif table_name == "micro_grids":
            sql_start = """CREATE TABLE micro_grids ("""
            sql = 'SCENARIO  INT,\n MG INT,\n TIME INT,\n '
            sql += 'PG FLOAT,\n QG FLOAT,\n PUG FLOAT,\n QUG FLOAT,\n '
            sql += 'PBIC_AC2DC FLOAT,\n PBIC_DC2AC FLOAT,\n QBIC FLOAT,\n PESS_CH FLOAT,\n '
            sql += 'PESS_DC  FLOAT,\n EESS FLOAT,\n PMESS FLOAT'
            sql_start_end = """)"""
        else:
            sql_start = """CREATE TABLE mobile_energy_storage system("""
            sql = 'SCENARIO  INT,\n TIME INT NOT NULL\n '
            sql_start_end = """)"""

        cursor.execute(sql_start + sql + sql_start_end)
        cursor.close()

    def insert_data_ds(self, table_name, nl=32, nb=33, ng=6, scenario=0, time=0, pij=0, qij=0, lij=0, vi=0, pg=0, qg=0):
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
        sql_start = "INSERT INTO " + table_name + " ("
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
        cursor.close()

    def insert_data_mg(self, table_name, scenario=0, time=0, mg=0, pg=0, qg=0, pug=0, qug=0, pbic_ac2dc=0, pbic_dc2ac=0,
                       qbic=0, pess_ch=0, pess_dc=0, eess=0, pmess=0):
        """
        insert microgrid data
        :param table_name:
        :param scenario:
        :param time:
        :param mg:
        :param pg:
        :param qg:
        :param pug:
        :param qug:
        :param pbic_ac2dc:
        :param pbic_dc2ac:
        :param qbic:
        :param pess_ch:
        :param pess_dc:
        :param eess:
        :param pmess:
        :return:
        """
        cursor = self.db.cursor()
        sql_start = "INSERT INTO " + table_name + " ("
        sql = "SCENARIO,MG,TIME,"
        value = "{0},{1},{2},".format(scenario, mg, time)
        sql += "PG,QG,PUG,QUG,PBIC_AC2DC,PBIC_DC2AC,QBIC,PESS_CH,PESS_DC,EESS,PMESS"
        value += "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}".format(pg, qg, pug, qug, pbic_ac2dc, pbic_dc2ac, qbic,
                                                                       pess_ch, pess_dc, eess, pmess)
        sql += ") VALUES (" + value + ")"
        cursor.execute(sql_start + sql)
        self.db.commit()
        cursor.close()

    # def insert_data_mess(self, table_name, nl=32, nb=33, ng=6, scenario=0, time=0, pij=0, qij=0, lij=0, vi=0, pg=0,
    #                      qg=0):


if __name__ == "__main__":
    db_management = DataBaseManagement()
    db_management.create_table(table_name="micro_grids")
    db_management.insert_data_mg(table_name="micro_grids")
