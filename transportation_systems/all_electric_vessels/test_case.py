"""
Test case for optimal voyage problem with 5 ports and 9 voyages
"""

from numpy import array, arange, zeros


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
        [0, 1, 100],
        [0, 2, 400],
        [0, 3, 200],
        [1, 2, 350],
        [1, 3, 300],
        [1, 4, 200],
        [2, 3, 500],
        [2, 4, 500],
        [3, 4, 300]
    ])

    initial_status = array([
        [0],
        [0],
        [0],
        [0],
        [1]
    ])

    end_status = array([
        [1],
        [0],
        [0],
        [0],
        [0]
    ])

    network = {"ports": ports,
               "voyage": voyage,
               "initial": initial_status,
               "end": end_status
               }

    return network


Vfull = 55.56
Vhalf = 33.336
Vin_out = 22.224
Vmin = 0
capacityEss = 100
socMax = 1
socMin = 0.2
pdcMax = 5
pchMax = 5
effCharing = 0.95
effDischaring = 1

PMIN = array([0, 0, 0, 0, 0])
PMAX = array([16.5, 16.5, 9.6, 9.6, 9.6])
a0 = array([3000, 3000, 210, 210, 210])
a1 = array([2185, 2185, 1623, 1623, 1623])
a2 = array([30, 30, 10, 10, 10])
b0 = array([8383, 8383, 360, 360, 360])
b1 = array([385, 385, 950, 950, 950])
b2 = array([385, 385, 950, 950, 950])

Price_port = array([
    [51.78, 52.54056, 52.70963],
    [55.71, 52.54, 18.44477],
    [59, 58, 10.5],
    [61.35, 52.52, 10.5],
    [62.66, 52.52, 10.5],
    [81.22, 52.52, 51.47604],
    [142.35, 103.69477, 107],
    [109.86, 208.96013, 217.70322],
    [80.28, 105.01, 113.51642],
    [64.74, 87.43991, 100.4981],
    [57.36, 67.40527, 77.8993],
    [51.03, 57.08583, 65],
    [54.01, 52.54, 60.17424],
    [60.64, 56.26158, 65],
    [60.47, 58, 66.99199],
    [63.46, 61, 71.51689],
    [81.2, 75.29024, 91.07793],
    [105.21, 105.01, 116.44817],
    [108.34, 299.6, 330.5956],
    [89.61, 97.53319, 107],
    [88.83, 100.13878, 85],
    [63.6, 76.81742, 71.36575],
    [83.5, 63.69508, 66.05],
    [68.84, 61.59095, 63.15723],
])

PL_FULL = 9.845  # Full speed service load
PL_CRUISE = 8.69  # Cruise speed service load
PL_IN_OUT = 9.075  # In/Out speed service load
PL_STOP = 3.5  # Stop speed service load

PUG_MAX = 10
PUG_MIN = -10

cp1 = 0.0012
cp2 = 3
nV = 10
vBlock = arange(0, Vfull + Vfull / (nV - 1), Vfull / (nV - 1))
PproBlock = zeros(nV)
for i in range(nV): PproBlock[i] = cp1 * (vBlock[i] / 1.852) ** cp2

mBlock = zeros(nV - 1)
for i in range(nV - 1):
    mBlock[i] = (PproBlock[i + 1] - PproBlock[i]) / (vBlock[i + 1] - vBlock[i])
