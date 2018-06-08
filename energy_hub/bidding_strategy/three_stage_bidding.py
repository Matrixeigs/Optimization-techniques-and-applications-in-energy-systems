"""
Three stage bidding based on the cplex
"""

from numpy import zeros, ones, array, eye, hstack, vstack, inf, transpose, where
import numpy as np
from gurobipy import *


def main(N_scenario_first_stage=5, N_scenario_second_stage=10):
    # 1) System level configuration
    T = 24
    weight_first_stage = ones((N_scenario_first_stage, 1)) / N_scenario_first_stage
    weight_second_stage = ones((N_scenario_second_stage, 1)) / N_scenario_second_stage

    forecasting_errors_ac = 0.03
    forecasting_errors_dc = 0.03
    forecasting_errors_pv = 0.05
    forecasting_errors_prices = 0.03
    Penalty = 0.0
    # For the HVAC system
    # 2) Thermal system configuration
    QHVAC_max = 100
    eff_HVAC = 4
    c_air = 1.85
    r_t = 1.3
    ambinent_temprature = array(
        [27, 27, 26, 26, 26, 26, 26, 25, 27, 28, 30, 31, 32, 32, 32, 32, 32, 32, 31, 30, 29, 28, 28, 27])
    temprature_in_min = 20
    temprature_in_max = 24

    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])

    # 3) Electricity system configuration
    PUG_MAX = 200
    PV_CAP = 50
    AC_PD_cap = 50
    DC_PD_cap = 50
    HD_cap = 100
    CD_cap = 100

    PESS_CH_MAX = 100
    PESS_DC_MAX = 100
    EFF_DC = 0.9
    EFF_CH = 0.9
    E0 = 50
    Emax = 100
    Emin = 20

    BIC_CAP = 100
    eff_BIC = 0.95

    electricity_price = array(
        [6.01, 73.91, 71.31, 69.24, 68.94, 70.56, 75.16, 73.19, 79.70, 85.76, 86.90, 88.60, 90.62, 91.26, 93.70, 90.94,
         91.26, 80.39, 76.25, 76.80, 81.22, 83.75, 76.16, 72.69])

    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])
    PV_PG = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])

    ELEC_PRICE = electricity_price / 300
    ELEC_PRICE = ELEC_PRICE.reshape(T, 1)
    Eess_cost = 0.01

    PV_PG = PV_PG * PV_CAP
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    ELEC_PRICE_first_stage = zeros((T, N_scenario_first_stage))
    for i in range(N_scenario_first_stage):
        ELEC_PRICE_first_stage[:, i] = ones((1, T)) + np.random.normal(0, forecasting_errors_prices,
                                                                       T)

    # Generate the second stage profiles using spline of scipy
    AC_PD_second_stage = zeros((T, N_scenario_second_stage))
    DC_PD_second_stage = zeros((T, N_scenario_second_stage))
    PV_second_stage = zeros((T, N_scenario_second_stage))
    ELEC_PRICE_second_stage = zeros((T, N_scenario_second_stage))

    for i in range(N_scenario_second_stage):
        AC_PD_second_stage[:, i] = ones((1, T)) + np.random.normal(0, forecasting_errors_ac, T)
        DC_PD_second_stage[:, i] = ones((1, T)) + np.random.normal(0, forecasting_errors_dc, T)
        PV_second_stage[:, i] = ones((1, T)) + np.random.normal(0, forecasting_errors_pv, T)
        ELEC_PRICE_second_stage[:, i] = ones((1, T)) + np.random.normal(0, forecasting_errors_prices, T)

    for i in range(N_scenario_second_stage):
        AC_PD_second_stage[:, i] = np.multiply(AC_PD, AC_PD_second_stage[:, i])
        DC_PD_second_stage[:, i] = np.multiply(DC_PD, DC_PD_second_stage[:, i])
        PV_second_stage[:, i] = np.multiply(PV_PG, PV_second_stage[:, i])
        ELEC_PRICE_second_stage[:, i] = np.multiply(transpose(ELEC_PRICE), ELEC_PRICE_second_stage[:, i])
        # Check the boundary information
        for j in range(T):
            if AC_PD_second_stage[j, i] < 0:
                AC_PD_second_stage[j, i] = 0
                if DC_PD_second_stage[j, i] < 0:
                    DC_PD_second_stage[j, i] = 0
            if PV_second_stage[j, i] < 0:
                PV_second_stage[j, i] = 0
            if ELEC_PRICE_second_stage[j, i] < 0:
                ELEC_PRICE_second_stage[j, i] = 0

    # CCHP system
    Gas_price = 0.1892
    Gmax = 200
    eff_CHP_e = 0.3
    eff_CHP_h = 0.4
    # Boiler information
    Boil_max = 100
    eff_boil = 0.9
    # Chiller information
    Chiller_max = 100
    eff_chiller = 1.2

    CCHP = {"MAX": Gmax,
            "EFF_E": eff_CHP_e,
            "EFF_C": eff_CHP_h,
            "EFF_H": eff_CHP_h,
            "COST": Gas_price}

    HVAC = {"CAP": QHVAC_max,
            "EFF": eff_HVAC,
            "C_AIR": c_air,
            "R_T": r_t,
            "TEMPERATURE": ambinent_temprature,
            "TEMP_MIN": temprature_in_min,
            "TEMP_MAX": temprature_in_max}

    THERMAL = {"HD": HD,
               "CD": CD, }

    ELEC = {"UG_MAX": PUG_MAX,
            "UG_MIN": -PUG_MAX,
            "UG_PRICE": ELEC_PRICE,
            "AC_PD": AC_PD,
            "DC_PD": DC_PD,
            "PV_PG": PV_PG
            }

    # The second stage scenarios
    ELEC_second_stage = {"AC_PD": AC_PD_second_stage,
                         "DC_PD": DC_PD_second_stage,
                         "PV_PG": PV_second_stage,
                         "UG_PRICE": ELEC_PRICE_second_stage, }

    BIC = {"CAP": BIC_CAP,
           "EFF": eff_BIC,
           }

    BESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "PC_MAX": PESS_CH_MAX,
            "PD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "COST": Eess_cost,
            }

    TESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            }

    CESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "TC_MAX": PESS_CH_MAX,
            "TD_MAX": PESS_DC_MAX,
            "EFF_CH": EFF_CH,
            "EFF_DC": EFF_DC,
            "EFF_SD": 0.98,  # The self discharging
            "COST": Eess_cost,
            "PMAX": PESS_CH_MAX * 10,
            "ICE": 3.5,
            }

    ESS = {"BESS": BESS,
           "TESS": TESS,
           "CESS": CESS}

    BOIL = {"CAP": Boil_max,
            "EFF": eff_boil}

    CHIL = {"CAP": Chiller_max,
            "EFF": eff_chiller}

    Price_DA = zeros((T, N_scenario_first_stage))
    for i in range(N_scenario_first_stage):
        for j in range(T):
            Price_DA[j, i] = ELEC_PRICE[j] * (1 + np.random.normal(0, forecasting_errors_prices))

    # Generate the order bidding curve, the constrain will be added from the highest order to the  lowest
    Order = zeros((T, N_scenario_first_stage))
    for i in range(T):
        Order[i, :] = np.argsort(Price_DA[i, :])

    model = Model("EnergyHub")
    PDA = {}  # Day-ahead bidding strategy
    pRT = {}  # Real-time prices
    pCHP = {}  # Real-time output of CHP units
    pAC2DC = {}  # Real-time power transfered from AC to DC
    pDC2AC = {}  # Real-time power transfered from DC to AC
    eESS = {}  # Real-time energy status
    pESS_DC = {}  # ESS discharging rate
    pESS_CH = {}  # ESS charging rate
    pIAC = {}  # HVAC consumption
    pPV = {}  # PV consumption
    pCS = {}
    ## Group 2: Heating ##
    qCHP = {}
    qGAS = {}
    eHSS = {}
    qHS_DC = {}
    qHS_CH = {}
    qAC = {}
    qTD = {}
    ## Group 3: Cooling ##
    qCE = {}
    qIAC = {}
    eCSS = {}
    qCS_DC = {}  # The output is cooling
    qCS_CH = {}  # The input is electricity
    qCD = {}
    ## Group 4: Gas ##
    vCHP = {}
    vGAS = {}

    # Define the day-ahead scheduling plan
    for i in range(T):
        for j in range(N_scenario_first_stage):
            PDA[i, j] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX, name="PDA{0}".format(i * N_scenario_first_stage + j))

    # Define the real-time scheduling plan
    for i in range(T):  # Dispatch at each time slot
        for j in range(N_scenario_second_stage):  # Scheduling plan under second stage plan
            for k in range(N_scenario_first_stage):  # Dispatch at each time slot
                pRT[i, j, k] = model.addVar(lb=-PUG_MAX, ub=PUG_MAX,
                                            name="pRT{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_E"],
                                             name="pCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pAC2DC[i, j, k] = model.addVar(lb=0, ub=BIC["CAP"],
                                               name="pAC2DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pDC2AC[i, j, k] = model.addVar(lb=0, ub=BIC["CAP"],
                                               name="pDC2AC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eESS[i, j, k] = model.addVar(lb=ESS["BESS"]["E_MIN"], ub=ESS["BESS"]["E_MAX"],
                                             name="eESS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pESS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["BESS"]["PD_MAX"],
                                                name="pESS_DC{0}".format(
                                                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pESS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["BESS"]["PC_MAX"],
                                                name="pESS_CH{0}".format(
                                                    i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pIAC[i, j, k] = model.addVar(lb=0, ub=HVAC["CAP"],
                                             name="pIAC{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pPV[i, j, k] = model.addVar(lb=0, ub=PV_second_stage[i, j],
                                            name="pPV{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                pCS[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["PMAX"],
                                            name="pCS{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 2: Heating ##
                qCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_H"],
                                             name="qCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qGAS[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"] * CCHP["EFF_H"],
                                             name="qGAS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eHSS[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["E_MAX"],
                                             name="eHSS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qHS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["TD_MAX"],
                                               name="qHS_DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qHS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["TESS"]["TC_MAX"],
                                               name="qHS_CH{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qAC[i, j, k] = model.addVar(lb=0, ub=CHIL["CAP"],
                                            name="qAC{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qTD[i, j, k] = model.addVar(lb=0, ub=THERMAL["HD"][i],
                                            name="qTD{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 3: Cooling ##
                qCE[i, j, k] = model.addVar(lb=0, ub=CHIL["CAP"],
                                            name="qCE{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qIAC[i, j, k] = model.addVar(lb=0, ub=HVAC["CAP"] * HVAC["EFF"],
                                             name="qIAC{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                eCSS[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["E_MAX"],
                                             name="eCSS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                qCS_DC[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["TD_MAX"],
                                               name="qCS_DC{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))  # The output is cooling
                qCS_CH[i, j, k] = model.addVar(lb=0, ub=ESS["CESS"]["TC_MAX"],
                                               name="qCS_CH{0}".format(
                                                   i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))  # The input is electricity
                qCD[i, j, k] = model.addVar(lb=0, ub=THERMAL["CD"][i],
                                            name="qCD{0}".format(
                                                i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                ## Group 4: Gas ##
                vCHP[i, j, k] = model.addVar(lb=0, ub=CCHP["MAX"],
                                             name="vCHP{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                vGAS[i, j, k] = model.addVar(lb=0, ub=BOIL["CAP"],
                                             name="vGAS{0}".format(
                                                 i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                if i == T - 1:
                    eESS[i, j, k] = model.addVar(lb=ESS["BESS"]["E0"], ub=ESS["BESS"]["E0"],
                                                 name="eESS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                    eHSS[i, j, k] = model.addVar(lb=ESS["TESS"]["E0"], ub=ESS["TESS"]["E0"],
                                                 name="eHSS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))
                    eCSS[i, j, k] = model.addVar(lb=ESS["CESS"]["E0"], ub=ESS["CESS"]["E0"],
                                                 name="eCSS{0}".format(
                                                     i * N_scenario_first_stage * N_scenario_second_stage + j * N_scenario_first_stage + k))

    ## Formulate the constraints set
    # 1） Energy storage systems
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                if i != 0:
                    # Battery energy constraint
                    model.addConstr(
                        eESS[i, j, k] == eESS[i - 1, j, k] + pESS_CH[i, j, k] * ESS["BESS"]["EFF_CH"] - pESS_DC[
                            i, j, k] / ESS["BESS"]["EFF_DC"])
                    # Heat energy storage constraint
                    model.addConstr(
                        eHSS[i, j, k] == ESS["TESS"]["EFF_SD"] * eHSS[i - 1, j, k] + qHS_CH[i, j, k] * ESS["TESS"][
                            "EFF_CH"] - qHS_DC[i, j, k] / ESS["TESS"]["EFF_DC"])
                    # Cooling energy storage constraint
                    model.addConstr(
                        eCSS[i, j, k] == ESS["CESS"]["EFF_SD"] * eCSS[i - 1, j, k] + qCS_CH[i, j, k] * ESS["CESS"][
                            "EFF_CH"] - qCS_DC[i, j, k] / ESS["CESS"]["EFF_DC"])
                else:
                    model.addConstr(
                        eESS[i, j, k] == ESS["BESS"]["E0"] + pESS_CH[i, j, k] * ESS["BESS"]["EFF_CH"] - pESS_DC[
                            i, j, k] / ESS["BESS"]["EFF_DC"])
                    model.addConstr(
                        eHSS[i, j, k] == ESS["TESS"]["EFF_SD"] * ESS["TESS"]["E0"] + qHS_CH[i, j, k] * ESS["TESS"][
                            "EFF_CH"] - qHS_DC[i, j, k] / ESS["TESS"]["EFF_DC"])
                    # Cooling energy storage constraint
                    model.addConstr(
                        eCSS[i, j, k] == ESS["CESS"]["EFF_SD"] * ESS["CESS"]["E0"] + qCS_CH[i, j, k] * ESS["CESS"][
                            "EFF_CH"] - qCS_DC[i, j, k] / ESS["CESS"]["EFF_DC"])
    # 2） Energy conversion relationship
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(pCHP[i, j, k] == vCHP[i, j, k] * CCHP["EFF_E"])
                model.addConstr(qCHP[i, j, k] == vCHP[i, j, k] * CCHP["EFF_H"])
                model.addConstr(qGAS[i, j, k] == vGAS[i, j, k] * BOIL["EFF"])
                model.addConstr(qCE[i, j, k] == qAC[i, j, k] * CHIL["EFF"])
                model.addConstr(qIAC[i, j, k] == pIAC[i, j, k] * HVAC["EFF"])
                model.addConstr(qCS_CH[i, j, k] == pCS[i, j, k] * ESS["CESS"]["ICE"])
    # 3) Energy balance equations
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                # AC bus power balance equation
                model.addConstr(
                    PDA[i, k] + pRT[i, j, k] + pCHP[i, j, k] - pAC2DC[i, j, k] + BIC["EFF"] * pDC2AC[i, j, k] ==
                    AC_PD_second_stage[i, j])

                # DC bus power balance equation
                model.addConstr(
                    BIC["EFF"] * pAC2DC[i, j, k] - pDC2AC[i, j, k] - pIAC[i, j, k] - pESS_CH[i, j, k] + pESS_DC[
                        i, j, k] + pPV[i, j, k] - pCS[i, j, k] ==
                    DC_PD_second_stage[i, j])

                # Heat energy balance
                model.addConstr(
                    qCHP[i, j, k] + qGAS[i, j, k] + qHS_DC[i, j, k] - qHS_CH[i, j, k] - qAC[i, j, k] - qTD[i, j, k] ==
                    HD[i])

                # Cooling energy balance
                model.addConstr(
                    qIAC[i, j, k] + qCE[i, j, k] + qCS_DC[i, j, k] - qCD[i, j, k] == CD[i])
    # 4) Constraints for the day-ahead and real-time energy trading
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                model.addConstr(pRT[i, j, k] + PDA[i, k] <= PUG_MAX)
                model.addConstr(pRT[i, j, k] + PDA[i, k] >= -PUG_MAX)

    # 5) Constraints for the bidding curves
    for i in range(T):
        Index = Order[i, :].tolist()
        for j in range(N_scenario_first_stage - 1):
            model.addConstr(PDA[i, Index.index(j)] >= PDA[i, Index.index(j + 1)])

    ## Formulate the objective functions
    # The first stage objective value
    obj_DA = 0
    for i in range(T):
        for j in range(N_scenario_first_stage):
            obj_DA += PDA[i, j] * Price_DA[i, j] * weight_first_stage[j]

    obj_RT = 0
    for k in range(N_scenario_first_stage):
        for j in range(N_scenario_second_stage):
            for i in range(T):
                obj_RT += pRT[i, j, k] * (ELEC_PRICE_second_stage[i, j] + Penalty) * weight_first_stage[k] * \
                          weight_second_stage[j]
                obj_RT += pESS_DC[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += pESS_CH[i, j, k] * ESS["BESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qHS_DC[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qHS_CH[i, j, k] * ESS["TESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qCS_DC[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += qCS_CH[i, j, k] * ESS["CESS"]["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += vGAS[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]
                obj_RT += vCHP[i, j, k] * CCHP["COST"] * weight_first_stage[k] * weight_second_stage[j]

    obj = obj_DA + obj_RT
    model.setObjective(obj)

    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.DisplayInterval = 1
    model.Params.LogFile = ""
    model.optimize()

    obj = obj.getValue()

    # Obtain the solutions
    pDA=zeros((T,N_scenario_first_stage))
    for i in range(T):
        for j in range(N_scenario_first_stage):
            pDA[i,j] = model.getVarByName("PDA{0}".format(i * N_scenario_first_stage + j)).X

    return model


if __name__ == "__main__":
    model = main(5, 10)
    print(model)
