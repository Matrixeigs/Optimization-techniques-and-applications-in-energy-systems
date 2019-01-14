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

    def create_table(self, table_name, nl=32, nb=33, ng=6, nmg=3, nmes=3):
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
                sql += "PIJ{0} DECIMAL(8,6),\n ".format(i)
            for i in range(nl):
                sql += "QIJ{0} DECIMAL(8,6),\n ".format(i)
            for i in range(nl):
                sql += "IIJ{0} DECIMAL(8,6),\n ".format(i)
            for i in range(nb):
                sql += "V{0} DECIMAL(8,6),\n ".format(i)
            for i in range(ng):
                sql += "PG{0} DECIMAL(8,6),\n ".format(i)
            for i in range(ng - 1):
                sql += "QG{0} DECIMAL(8,6),\n ".format(i)
            sql += "QG{0} DECIMAL(8,6)\n ".format(ng - 1)
            sql_start_end = """)"""
        elif table_name == "micro_grids":
            sql_start = """CREATE TABLE micro_grids ("""
            sql = 'SCENARIO  INT,\n MG INT,\n TIME INT,\n '
            sql += 'PG DECIMAL(7,4),\n QG DECIMAL(7,4),\n PUG DECIMAL(7,4),\n QUG DECIMAL(7,4),\n '
            sql += 'PBIC_AC2DC DECIMAL(7,4),\n PBIC_DC2AC DECIMAL(7,4),\n QBIC DECIMAL(7,4),\n PESS_CH DECIMAL(7,4),\n '
            sql += 'PESS_DC  DECIMAL(7,4),\n EESS DECIMAL(7,4),\n PMESS DECIMAL(7,4)'
            sql_start_end = """)"""
        elif table_name == "mobile_energy_storage_systems":
            sql_start = """CREATE TABLE mobile_energy_storage_systems ("""
            sql = 'SCENARIO  INT,\n MESS INT,\n TIME INT,\n'
            for i in range(nmg):
                sql += "PDC_MG{0} DECIMAL(7,4),\n ".format(i)
            for i in range(nmg):
                sql += "PCH_MG{0} DECIMAL(7,4),\n ".format(i)
            sql += "EESS DECIMAL(7,4)\n "
            sql_start_end = """)"""
        elif table_name == "first_stage_solutions":  # First-stage solution table
            sql_start = """CREATE TABLE first_stage_solutions ("""
            sql = 'TIME  INT,\n'
            for i in range(ng):
                sql += "PG{0} DECIMAL(7,4),\n ".format(i)
                sql += "RG{0} DECIMAL(7,4),\n ".format(i)
            for i in range(nmg):
                sql += "PG_MG{0} DECIMAL(7,4),\n ".format(i)
                sql += "RG_MG{0} DECIMAL(7,4),\n ".format(i)
                sql += "IESS{0} INT,\n ".format(i)
                sql += "PESS_DC{0} DECIMAL(7,4),\n ".format(i)
                sql += "PESS_CH{0} DECIMAL(7,4),\n ".format(i)
                sql += "RESS{0} DECIMAL(7,4),\n ".format(i)
                sql += "ESS{0} DECIMAL(7,4),\n ".format(i)
            for i in range(nmes - 1):
                sql += "IMESS{0} INT,\n ".format(i)
                sql += "PCH_MESS{0} DECIMAL(7,4),\n ".format(i)
                sql += "PDC_MESS{0} DECIMAL(7,4),\n ".format(i)
                sql += "RMESS{0} DECIMAL(7,4),\n ".format(i)
                sql += "MESS_F_STOP{0} DECIMAL(7,4),\n ".format(i)
                sql += "MESS_T_STOP{0} DECIMAL(7,4),\n ".format(i)
            sql += "IMESS{0} INT,\n ".format(nmes - 1)
            sql += "PCH_MESS{0} DECIMAL(7,4),\n ".format(nmes - 1)
            sql += "PDC_MESS{0} DECIMAL(7,4),\n ".format(nmes - 1)
            sql += "RMESS{0} DECIMAL(7,4),\n ".format(nmes - 1)
            sql += "MESS_F_STOP{0} DECIMAL(7,4),\n ".format(nmes - 1)
            sql += "MESS_T_STOP{0} DECIMAL(7,4)\n ".format(nmes - 1)
            sql_start_end = """)"""
        else:
            sql_start = """CREATE TABLE scenarios ("""
            sql = 'SCENARIO  INT,\n WEIGHT DECIMAL(7,4),\n TIME INT,\n'
            for i in range(nb):
                sql += "PD{0} DECIMAL(7,4),\n ".format(i)
            for i in range(nmg):
                sql += "PD_AC{0} DECIMAL(7,4),\n ".format(i)
            for i in range(nmg - 1):
                sql += "PD_DC{0} DECIMAL(7,4),\n ".format(i)
            sql += "PD_DC{0} DECIMAL(7,4)\n".format(nmg - 1)
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

    def insert_data_mess(self, table_name, scenario=0, time=0, mess=0, pess_ch=[0, 0, 0], pess_dc=[0, 0, 0], eess=0,
                         nmg=3):
        """
        insert mobile energy storage systems data
        :param table_name:
        :param scenario:
        :param time:
        :param mess:
        :param pess_ch:
        :param pess_dc:
        :param eess:
        :param nmg:
        :return:
        """
        cursor = self.db.cursor()
        sql_start = "INSERT INTO " + table_name + " ("
        sql = "SCENARIO,MESS,TIME,"
        value = "{0},{1},{2},".format(scenario, mess, time)
        for i in range(nmg):
            sql += "PDC_MG{0},".format(i)
            value += "{0},".format(pess_dc[i])
        for i in range(nmg):
            sql += "PCH_MG{0},".format(i)
            value += "{0},".format(pess_ch[i])
        sql += "EESS"
        value += "{0}".format(eess)
        sql += ") VALUES (" + value + ")"
        cursor.execute(sql_start + sql)
        self.db.commit()
        cursor.close()

    def insert_data_first_stage(self, table_name, time=0, ng=2, nmg=2, nmes=2, pg=[0, 0], rg=[0, 0], pg_mg=[0, 0],
                                rg_mg=[0, 0], iess=[0, 0], pess_dc=[0, 0], pess_ch=[0, 0], ress=[0, 0], ess=[0, 0],
                                imess=[0, 0], pch_mess=[0, 0], pdc_mess=[0, 0], rmess=[0, 0], mess_f_stop=[0, 0],
                                mess_t_stop=[0, 0]):
        """
        insert scenario data
        :param table_name:
        :param scenario:
        :param weight:
        :param time:
        :param nb:
        :param nmg:
        :param pd:
        :param pd_ac:
        :param pd_dc:
        :return:
        """
        cursor = self.db.cursor()
        sql_start = "INSERT INTO " + table_name + " ("
        sql = "TIME,"
        value = "{0},".format(time)
        for i in range(ng):
            sql += "PG{0},".format(i)
            sql += "RG{0},".format(i)
            value += "{0},".format(pg[i])
            value += "{0},".format(rg[i])
        for i in range(nmg):
            sql += "PG_MG{0},".format(i)
            sql += "RG_MG{0},".format(i)
            sql += "IESS{0},".format(i)
            sql += "PESS_DC{0},".format(i)
            sql += "PESS_CH{0},".format(i)
            sql += "RESS{0},".format(i)
            sql += "ESS{0},".format(i)
            if nmg>1:
                value += "{0},".format(pg_mg[i])
                value += "{0},".format(rg_mg[i])
                value += "{0},".format(iess[i])
                value += "{0},".format(pess_dc[i])
                value += "{0},".format(pess_ch[i])
                value += "{0},".format(ress[i])
                value += "{0},".format(ess[i])
            else:
                value += "{0},".format(pg_mg)
                value += "{0},".format(rg_mg)
                value += "{0},".format(iess)
                value += "{0},".format(pess_dc)
                value += "{0},".format(pess_ch)
                value += "{0},".format(ress)
                value += "{0},".format(ess)
        if nmes > 1:
            for i in range(nmes - 1):
                sql += "IMESS{0},".format(i)
                sql += "PCH_MESS{0},".format(i)
                sql += "PDC_MESS{0}, ".format(i)
                sql += "RMESS{0},".format(i)
                sql += "MESS_F_STOP{0},".format(i)
                sql += "MESS_T_STOP{0},".format(i)
                value += "{0},".format(imess[i])
                value += "{0},".format(pch_mess[i])
                value += "{0},".format(pdc_mess[i])
                value += "{0},".format(rmess[i])
                value += "{0},".format(mess_f_stop[i])
                value += "{0},".format(mess_t_stop[i])
        sql += "IMESS{0},".format(nmes - 1)
        sql += "PCH_MESS{0},".format(nmes - 1)
        sql += "PDC_MESS{0}, ".format(nmes - 1)
        sql += "RMESS{0},".format(nmes - 1)
        sql += "MESS_F_STOP{0},".format(nmes - 1)
        sql += "MESS_T_STOP{0}".format(nmes - 1)
        value += "{0},".format(imess[nmes - 1])
        value += "{0},".format(pch_mess[nmes - 1])
        value += "{0},".format(pdc_mess[nmes - 1])
        value += "{0},".format(rmess[nmes - 1])
        value += "{0},".format(mess_f_stop[nmes - 1])
        value += "{0}".format(mess_t_stop[nmes - 1])

        sql += ") VALUES (" + value + ")"
        cursor.execute(sql_start + sql)
        self.db.commit()
        cursor.close()

    def inquery_data_scenario(self, table_name, scenario=0, time=0):
        cursor = self.db.cursor()
        # sql = "SELECT * FROM " + table_name + " ;"
        sql = "SELECT * FROM " + table_name + " WHERE SCENARIO={0} AND TIME={1};".format(scenario, time)
        cursor.execute(sql)
        data = cursor.fetchall()
        n_data = len(data[0])

        temp = []
        for i in range(n_data): temp.append(float(data[0][i]))

        cursor.close()
        return temp


if __name__ == "__main__":
    db_management = DataBaseManagement()
    db_management.create_table("first_stage_solutions")
    db_management.insert_data_first_stage("first_stage_solutions")

    data = db_management.inquery_data_scenario(table_name="scenarios")
    print(data)
