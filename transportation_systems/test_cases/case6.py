"""
Test cases for time space networks and jointed dispatch of time and space networks
"""

from numpy import array

from transportation_systems.test_cases import TIME


def transportation_network(delta=1):
    branch = array([
        [0, 1, 1],
        [0, 2, 4],
        [0, 3, 2],
        [1, 3, 3],
        [1, 4, 2],
        [2, 3, 5],
        [3, 4, 3],
        [4, 5, 3],
        [2, 5, 2],
    ])

    branch[:, TIME] = branch[:, TIME] / delta

    bus = array([
        [0, 100, 10],
        [1, 100, 20],
        [2, 100, 30],
        [3, 100, 40],
        [4, 100, 50],
        [5, 100, 60]
    ])

    initial_status = array([
        [0],
        [0],
        [0],
        [0],
        [1],
        [0]
    ])

    end_status = array([
        [1],
        [0],
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
