"""
Interval unit commitment
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""
from numpy import zeros, shape, ones, diag, concatenate, r_, arange, divide
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
import scipy.linalg as linalg
from scipy.sparse import csr_matrix as sparse

class IntervalUnitCommitment():
    ""

    def __init__(self):
        self.name = "Interval Unit Commitment"

    def problem_formulation(self, case, Load_profile):
        # Hydro power unit
        IG = 0
        PG = 1
        RUG = 2
        RDG = 3
        VH = 4
        QH = 5
        # Load shedding part
        PLC = 6
        # Output of wind power unit
        PWG = 7

