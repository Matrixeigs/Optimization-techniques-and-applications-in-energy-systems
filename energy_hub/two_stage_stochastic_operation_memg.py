"""
Two stage stochastic optimization problem for multi energy microgrid
@author: Zhao Tianyang
@e-mail:matrixeigs@gmail.com
"""
from gurobipy import *
import numpy as np
import os
import pandas as pd


class TwoStageStochastic():
    def __init__(self):
        self.pwd = os.getcwd()

    def day_ahead_scheduling(self, SCENARIO_UPDATE=0, N_S=100, AC_LOAD_MEAN, AC_LOAD_STD):
        """
        Day-ahead scheduling problem for memg
        :return:
        """
    T = len(AC_LOAD_MEAN)
    self.T = T
    # 1) Generate scenarios
    if SCENARIO_UPDATE > 0:  # We need to update the scenarios
        ac_load = self.scenario_generation(AC_LOAD_MEAN, AC_LOAD_STD)

    # 2) Problem formulation

    # 3) Problem solving

    # 4) Save the results

    def scenario_generation(self, MEAN_VALUE, STD, N_S):
