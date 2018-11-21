"""
Test cases for time space networks and jointed dispatch of time and space networks
"""

from numpy import array

# from transportation_systems.test_cases import TIME
TIME = 2

def transportation_network(delta=1):
    branch = array([
        [0, 1, 1],
        [0, 2, 1],
        [1, 2, 2]
    ])

    branch[:, TIME] = branch[:, TIME] / delta

    bus = array([
        [0, 5],
        [1, 25],
        [2, 28]
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

    network = {"bus": bus,
               "branch": branch,
               "initial": initial_status,
               "end": end_status
               }

    return network
