"""
Optimal power flow for hybrid AC/DC micro-grids
Two versions of optimal power flow models are proposed.
1) Single period
2) Multiple periods
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
"""
from pypower import case9
from numpy import power, array, zeros
from scipy import hstack, vstack

# import test cases
from Two_stage_stochastic_optimization.power_flow_modelling import case_converters
from Two_stage_stochastic_optimization.power_flow_modelling import case_converters


class MultipleMicrogridsDirect_CurrentNetworks():
    """
    Dynamic optimal power flow modelling for micro-grid power parks
    The power parks include
    """

    def __init__(self):
        self.logger = MultipleMicrogridsDirect_CurrentNetworks.run()

    def run(self):
        # 1) Optimal power flow modelling for MGs
        # 2) Optimal power flow modelling for DC networks
        # 3) Connnection matrix between MGs and DC networks
        # 3.1) Update the decision variables
        # 3.2) Update the constraint set
        # 3.3) Update the objective functions
        # 4) Results check
        # 4.1) Bi-directional power flows on ESSs
        # 4.2) Bi-directional power flows on BICs
        # 4.3) Relaxation of DC power flows
        # 4.4) Stochastic simulation

        sol = {"x": 0}
        return sol

    # def optimal_power_flow_microgrid(self):
    #
    # def optimal_power_flow_direct_current_networks(self):
    #
    # def optimal_power_flow_solving(self):
    #
    # def optimal_power_flow_solving_result_check(self):


# 1）


if __name__ == '__main__':
    T = 24
    NMG = 3
    # The test system for micro-grid systems
    # 1) Load profile within each MG
    # 1.1) Forecasting information
    Load_profile_MGs = array([
        [221, 219.7, 224.9, 221, 227.5, 240.5, 260, 315.9, 330.2, 338, 347.1, 336.7, 336.7, 331.5, 340.6, 344.5, 331.5,
         328.9, 325, 331.5, 338, 322.4, 296, 239.2],
        [114.15, 109.78, 107.6, 107.16, 110.07, 118.52, 132.79, 142.25, 144.29, 145.16, 145.6, 145.16, 144.44, 144,
         142.83, 141.38, 140.36, 139.19, 138.17, 137.59, 140.07, 138.32, 128.71, 119.68],
        [52, 50, 50, 51, 56, 63, 70, 75, 76, 80, 78, 74, 72, 72, 76, 80, 85, 88, 90, 87, 78, 71, 65, 56]
    ])
    # 1.2) Forecasting errors
    Delta_MGs = array([0.1, 0.1, 0.1]) * 2
    # 1.3) Information interval
    Load_profile_interval_MGs_min = zeros((NMG, T))
    Load_profile_interval_MGs_max = zeros((NMG, T))
    for i in range(NMG):
        Load_profile_interval_MGs_min[i, :] = Load_profile_MGs[i, :] * (1 - Delta_MGs[i])
        Load_profile_interval_MGs_max[i, :] = Load_profile_MGs[i, :] * (1 + Delta_MGs[i])
    # 2) PV information
    PV_profile_MGs_base = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_cap = array([100, 50, 50])
    Delta_PV = [0.1, 0.1, 0.1] * 3
    PV_profile_MGs = zeros((NMG, T))
    PV_profile_MGs_min = zeros((NMG, T))
    PV_profile_MGs_max = zeros((NMG, T))
    for i in range(NMG):
        for j in range(T):
            PV_profile_MGs[i, j] = PV_profile_MGs_base[j] * PV_cap[i]
            PV_profile_MGs_min[i, j] = PV_profile_MGs_base[j] * PV_cap[i] * (1 - Delta_PV[i])
            PV_profile_MGs_max[i, j] = PV_profile_MGs_base[j] * PV_cap[i] * (1 + Delta_PV[i])

    # 3) Market prices information
    Price_Wholesale = array(
        [0.0484, 0.0446, 0.0437, 0.0445, 0.0518, 0.069, 0.0824, 0.0801, 0.088, 0.09, 0.0926, 0.0887, 0.0906, 0.0905,
         0.086, 0.0791, 0.0746, 0.0655, 0.0624, 0.0658, 0.0727, 0.0609, 0.0525, 0.0449])
    # 4) Schedulable resources information
    Pdg_max = array([200, 100, 100])
    Pdg_min = array([40, 20, 20])
    Ru_dg = array([80, 40, 40])
    Rd_dg = array([80, 40, 40])
    Pbic_max = array([200, 100, 100])
    Pug_max = array([1000, 0, 0])
    Pug_min = array([0, 0, 0])
    Pmg_min = array([-50, -50, -50])
    Pmg_max = array([50, 50, 50])
    Pc_max = array([25, 25, 25])
    Capacity_ESS = array([50, 50, 50])
    Soc_max = array([1, 1, 1])
    Soc_min = array([0.1, 0.1, 0.1])
    eff_c = array([0.95, 0.95, 0.95])
    eff_dc = array([0.95, 0.95, 0.95])
    E0 = Capacity_ESS * 0.5
    eff_bic = array([0.95, 0.95, 0.95])
    C_dg = array([0.04335, 0.04554, 0.05154])
    Q_dg = array([0.01, 0.01, 0.01])
    C_ess = array([0.01, 0.01, 0.01])

    # 5) Generate information models for each MG
    # 5.1) MG queue for a cluster of MGs. Using list to store thees information.
    MG = []
    for i in range(NMG):
        DG_temp = {"PMAX": Pdg_max[i],
                   "PMIN": Pdg_min[i],
                   "RU": Ru_dg[i],
                   "RD": Rd_dg[i],
                   "C": C_dg[i],
                   "Q": Q_dg[i]}
        UG_temp = {"PMAX": Pug_max[i],
                   "PMIN": Pug_min[i],
                   "C": Price_Wholesale}
        BIC_temp = {"SMAX": Pbic_max[i],
                    "EFF_AC2DC": eff_bic[i],
                    "EFF_DC2AC": eff_bic[i], }
        ESS_temp = {"E0": E0[i],
                    "CAP": Capacity_ESS[i],
                    "SOC_MAX": 1,
                    "SOC_MIN": 0.1,
                    "EFF_DIS": eff_dc[i],
                    "EFF_CH": eff_c[i],
                    "COST_DIS": C_ess[i],
                    "COST_CH": C_ess[i], }
        Load_ac_temp = {"P": Load_profile_MGs[i, :] / 2,
                        "DELAT": (Load_profile_interval_MGs_max[i, :] - Load_profile_interval_MGs_min[i, :]) / 2}
        Load_dc_temp = {"P": Load_profile_MGs[i, :] / 2,
                        "DELAT": (Load_profile_interval_MGs_max[i, :] - Load_profile_interval_MGs_min[i, :]) / 2}
        PV_temp = {"P": PV_profile_MGs[i, :] / 2,
                   "DELAT": PV_profile_MGs_max[i, :] - PV_profile_MGs_min[i, :]}
        MG_temp = {"DG": DG_temp,
                   "UG": UG_temp,
                   "BIC": BIC_temp,
                   "ESS": ESS_temp,
                   "LOAD_AC": Load_ac_temp,
                   "LOAD_DC": Load_dc_temp,
                   "PV": PV_temp,
                   "AREA": i}
        MG.append(MG_temp)
        del DG_temp, UG_temp, BIC_temp, ESS_temp, Load_ac_temp, Load_dc_temp, PV_temp, MG_temp
    # 5.2) Connection matrix between each MG and the transmission networks

    # A test hybrid AC DC network is connected via BIC networks
    caseDC = case9.case9()
    mmDC = MultipleMicrogridsDirect_CurrentNetworks()

    sol = mmDC.run()
