"""
Gas flow package for gas network analysis
@e-mail: zhaoty@ntu.edu.sg
@author: Zhao Tianyang
"""

NODE_I = 1  # node number (1 to 29997)
NODE_TYPE = 2  # node type (1 - DN demand node, 2 - SN supply node, 3 - reference node, 4 - isolated node)
D = 3  # D, gas demand (Mm^3/day)
NODE_AREA = 4  # area number, 1-100
NP = 5  # P, pressure (bar)
BASE_PRESSURE = 6  # baseBAR, base pressure (bar)
ZONE = 7  # zone, loss zone (1-999)
PMAX = 8  # maxp, maximum pressure (bar)      (not in PTI format)
PMIN = 9  # minp, minimum pressure (bar)      (not in PTI format)
