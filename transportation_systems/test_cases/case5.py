"""
Test cases for time space networks and jointed dispatch of time and space networks
"""

from numpy import array
F_BUS = 0
T_BUS = 1
TIME = 2

def transportation_network(delta=1):

    network=array([
        [0,1,1],
        [0,2,1],
        [0,3,2],
        [1,3,1],
        [1,4,2],
        [2,3,1],
        [3,4,3]
    ])

    network[:,TIME] = network[:,TIME]/delta

    return network
