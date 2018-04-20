"""
Optimal power flow for hybrid AC/DC micro-grids
Two versions of optimal power flow models are proposed.
1) Single period
2) Multiple periods
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
"""
from pypower import case9
from numpy import power, array
from scipy import hstack, vstack


class Multiple_Microgrids_Direct_Current_Networks():
    """
    Dynamic optimal power flow modelling for micro-grid power parks
    The power parks include
    """

    def __init__(self):
        self.logger = Multiple_Microgrids_Direct_Current_Networks.run()

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

    def optimal_power_flow_microgrid(self):

    def optimal_power_flow_direct_current_networks(self):

    def optimal_power_flow_solving(self):

    def optimal_power_flow_solving_result_check(self):


# 1）


if __name__ == '__main__':
    # A test hybrid AC DC network is connected via BIC networks
    caseDC = case9.case9()
    mmDC = Multiple_Microgrids_Direct_Current_Networks()
    sol = mmDC.run()
