"""
Two Stage Robust Optimization for Unit Commitment Problem
The test case is the IEEE-6ww system.

@date: 13 June 2018
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
"""

from pypower import case6ww


class TwoStageUnitCommitmentRobust():
    """"""

    def __init__(self):
        self.name = "Two stage robust optimization"

    def input_check(self, cases):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """
