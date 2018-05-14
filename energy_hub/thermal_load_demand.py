"""
Thermal load demand management
This function is to test the thermal load management method adopted in the following reference.
[1]Robustly Coordinated Operation of A Multi-Energy Microgrid with Flexible Electric and Thermal Loads

It should be noted that, the modelling in [1] is borrowed from Ref.[2].
[2]Optimal Smart Home Energy Management Considering Energy Saving and a Comfortable Lifestyle

Some useful references can be found in Ref.[3] and Ref.[4].
[3] Thermal Battery Modeling of Inverter Air Conditioning for Demand Response
[4]A Novel Thermal Energy Storage System in Smart Building Based on Phase Change Material

Three stages and two-stage decision making.



Data sources:
1) Weather information
https://www.timeanddate.com/weather/singapore/singapore/hourly
2) The test system is a commercial building

The calculation of CVaR is borrowed from
http://faculty.chicagobooth.edu/ruey.tsay/teaching/bs41202/sp2011/lec9-11.pdf

@author:Tianyang Zhao
@e-mail:zhaoty@ntu.edu.sg
"""
from numpy import array, arange
from scipy import interpolate
from matplotlib import pyplot


class EnergyHubManagement():
    def __init__(self):
        self.name = "hybrid AC/DC embedded energy hub"

    def problem_formulation(self, ELEC=None, BIC=None, ESS=None, CCHP=None, HVAC=None, THERMAL=None):
        """
        Problem formulation for energy hub management
        :param ELEC: Electrical system with the load and utility grid information
        :param BIC: Bi-directional converter information
        :param ESS: Energy storage system information (Battery ESS and Thermal ESS)
        :param CCHP: Combined heat and power units information
        :param HVAC: Heat, ventilation and air-conditioning information
        :param THERMAL: Thermal load information
        :return:
        """
        from energy_hub.data_format import CCHP, UG, PAC2DC, PDC2AC, PHVAC, PESS, PESSCH, PESSDC, TESS, TESSCH, TESSDC, \
            NX



        return ELEC


# def run(self,Delta_t,Profile,HVAC,):


if __name__ == "__main__":
    # A test system
    # 1) System level configuration
    T = 24
    Delta_t = 1
    delat_t = 1
    T_second_stage = int(T / delat_t)

    # For the HVAC system
    # 2) Thermal system configuration
    QHVAC_max = 100
    eff_HVAC = 0.9
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
    AC_PD_cap = 100
    DC_PD_cap = 100
    HD_cap = 50
    CD_cap = 50

    PESS_CH_MAX = 100
    PESS_DC_MAX = 100
    EFF_DC = 0.9
    EFF_CH = 0.9
    E0 = 100
    Emax = 200
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

    ELEC_PRICE = electricity_price / 1000

    Eess_cost = 0.01

    PV_PG = PV_PG * PV_CAP
    # Modify the first stage profiles
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    # Generate the second stage profiles using spline of scipy
    Time_first_stage = arange(0, T, Delta_t)
    Time_second_stage = arange(0, T, delat_t)

    # AC_PD_tck = interpolate.splrep(Time_first_stage, AC_PD, s=0)
    # DC_PD_tck = interpolate.splrep(Time_first_stage, DC_PD, s=0)
    # PV_PG_tck = interpolate.splrep(Time_first_stage, PV_PG, s=0)
    #
    # AC_PD_second_stage = interpolate.splev(Time_second_stage, AC_PD_tck, der=0)
    # DC_PD_second_stage = interpolate.splev(Time_second_stage, DC_PD_tck, der=0)
    # PV_PG_second_stage = interpolate.splev(Time_second_stage, PV_PG_tck, der=0)
    #
    # for i in range(T_second_stage):
    #     if AC_PD_second_stage[i] < 0:
    #         AC_PD_second_stage[i] = 0
    #     if DC_PD_second_stage[i] < 0:
    #         DC_PD_second_stage[i] = 0
    #     if PV_PG_second_stage[i] < 0:
    #         PV_PG_second_stage[i] = 0

    # CCHP system
    Gas_price = 0.1892
    Gmax = 200
    eff_CHP_e = 0.4
    eff_CHP_h = 0.35
    # pyplot.plot(Time_first_stage, AC_PD, 'x', Time_second_stage, AC_PD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, DC_PD, 'x', Time_second_stage, DC_PD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, HD, 'x', Time_second_stage, HD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, CD, 'x', Time_second_stage, CD_second_stage, 'b')
    # pyplot.plot(Time_first_stage, PV_PG, 'x', Time_second_stage, PV_PG_second_stage, 'b')
    # pyplot.show()

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
            "UG_PRICE": electricity_price,
            "AC_PD": AC_PD,
            "DC_PD": DC_PD,
            "PV_PG": PV_PG
            }

    BIC = {"CAP": BIC_CAP,
           "EFF": eff_BIC,
           }

    BESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "PC_MAX": PESS_CH_MAX,
            "PD_MAX": PESS_DC_MAX,
            "COST": Eess_cost,
            }

    TESS = {"E0": E0,
            "E_MAX": Emax,
            "E_MIN": Emin,
            "PC_MAX": PESS_CH_MAX,
            "PD_MAX": PESS_DC_MAX,
            "COST": Eess_cost,
            }
    ESS = {"BESS": BESS,
           "TESS": TESS}

    energy_hub_management = EnergyHubManagement()
    model = energy_hub_management.problem_formulation(ELEC=ELEC, CCHP=CCHP, THERMAL=THERMAL, BIC=BIC, ESS=ESS,
                                                      HVAC=HVAC)

    print(model)
