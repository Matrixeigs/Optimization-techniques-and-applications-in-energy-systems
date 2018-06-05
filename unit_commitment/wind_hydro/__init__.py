"""
Jointed wind and hydro power dispatch under virtual power plan framework
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
# test the
"""

from unit_commitment.test_cases.case14 import case14
from pypower import runopf

sol = runopf.runopf(case14())