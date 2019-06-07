"""
Adelaide (SA) 0
Melbourne (VIC) 1
Sydney (NSW) 2
Newcastle (NSW) 3
Brisbane (QLD) 4
"""

from numpy import array, arange, zeros
import os, platform
import pandas as pd


def transportation_network(delta=1):
    # The ports array
    ports = array([
        [0, 100],
        [1, 100],
        [2, 100],
        [3, 100],
        [4, 100]
    ])
    # The voyage array, from ports, to ports and distance
    voyage = array([
        [0, 1, 920],
        [0, 2, 1970],
        [0, 3, 2090],
        [0, 4, 2790],
        [1, 2, 1050],
        [1, 3, 1170],
        [1, 4, 1870],
        [2, 3, 120],
        [2, 4, 820],
        [3, 4, 700]
    ])

    voyage[:, 2] = voyage[:, 2] / 1.852

    initial_status = array([
        [1],
        [0],
        [0],
        [0],
        [0]
    ])

    end_status = array([
        [0],
        [0],
        [0],
        [0],
        [1]
    ])

    network = {"ports": ports,
               "voyage": voyage,
               "initial": initial_status,
               "end": end_status
               }

    return network


Vfull = 25
Vhalf = 12
Vin_out = 10
Vmin = 0
capacityEss = 100
socMax = 1
socMin = 0.2
pdcMax = 5
pchMax = 5
effCharing = 0.95
effDischaring = 1

PMIN = array([0, 0, 0])
PMAX = array([16.5, 16.5, 9.6])
a0 = array([3000, 3000, 210])
a1 = array([2185, 2185, 1623])
# a2 = array([30, 30, 10])
a2 = array([0, 0, 0])

b0 = array([8383, 8383, 360])
b1 = array([385, 385, 950])
# b2 = array([385, 385, 950])
b2 = array([0, 0, 0])

Price_port = pd.read_excel(os.getcwd() + '/Prices_modified.xlsx', index_col=0).as_matrix()

PL_FULL = 9.845  # Full speed service load
PL_CRUISE = 8.69  # Cruise speed service load
PL_IN_OUT = 9.075  # In/Out speed service load
PL_STOP = 3.5  # Stop speed service load

PUG_MAX = 10
PUG_MIN = -10

cp1 = 0.003
cp2 = 3
nV = 10
vBlock = arange(0, Vfull + Vfull / (nV - 1), Vfull / (nV - 1))
PproBlock = zeros(nV)
for i in range(nV): PproBlock[i] = cp1 * vBlock[i] ** cp2

mBlock = zeros(nV - 1)
for i in range(nV - 1):
    mBlock[i] = (PproBlock[i + 1] - PproBlock[i]) / (vBlock[i + 1] - vBlock[i])
