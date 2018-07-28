"""
Test cases for micro_grids under unit commitment
"""
from numpy import array

Price_UG = array(
    [6.01, 75.91, 73.31, 71.24, 70.94, 69.56, 74.16, 72.19, 80.70, 86.76, 85.90, 87.60, 91.62, 90.26, 95.70, 87.94,
     91.26, 82.39, 75.25, 76.80, 81.22, 83.75, 76.16, 72.69]) / 300

AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
               1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
               1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
               1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
               791.5814, 575.4085, 551.1441, 349.3568, 299.992])

DG = {"PMIN": 0,
      "PMAX": 5,
      "COST_A": 0.01,
      "COST_B": 0.5}

UG = {"PMIN": -5,
      "PMAX": 5,
      "COST": Price_UG, }  # The cost should be a profile

ESS = {"PDC_MAX": 5,
       "PCH_MAX": 5,
       "EFF_DC": 0.95,
       "EFF_CH": 0.95,
       "E0": 10,
       "EMIN": 5,
       "EMAX": 20, }

BIC = {"PMAX": 5,
       "EFF_AC2DC": 0.9,
       "EFF_DC2AC": 0.9, }

MG = {"PMAX": 5,
      "PMIN": -5}

PD = {"AC": AC_PD / max(AC_PD),
      "AC_MAX": 5,
      "DC": DC_PD / max(DC_PD),
      "DC_MAX": 5}

micro_grid = {"DG": DG,
              "UG": UG,
              "BIC": BIC,
              "ESS": ESS,
              "PD": PD,
              "MG": MG}
