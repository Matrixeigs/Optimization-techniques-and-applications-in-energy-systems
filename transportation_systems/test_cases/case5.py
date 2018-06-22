"""
Test cases for time space networks and jointed dispatch of time and space networks
"""

from numpy import array

from transportation_systems.test_cases import TIME


def transportation_network(delta=1):
    branch = array([
        [0, 1, 1],
        [0, 2, 1],
        [0, 3, 2],
        [1, 3, 1],
        [1, 4, 2],
        [2, 3, 1],
        [3, 4, 3]
    ])
    # branch = array([
    #     [0, 1, 1],
    #     [0, 2, 1],
    #     [1, 2, 2],
    # ])

    branch[:, TIME] = branch[:, TIME] / delta

    bus = array([
        [0, 100],
        [1, 100],
        [2, 100],
        [3, 100],
        [4, 100]
    ])
    # bus = array([
    #     [0, 100],
    #     [1, 100],
    #     [2, 100],
    # ])

    initial_status = array([
        [0],
        [1],
        [0],
        [0],
        [0]
    ])

    network = {"bus": bus,
               "branch": branch,
               "initial": initial_status,
               "end": initial_status
               }

    return network
