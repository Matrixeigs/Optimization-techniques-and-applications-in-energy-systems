"""
Bi-directional converters models for the connection between AC buses and DC buses
Data format:
AC bus, DC bus, efficiency from AC to DC, efficiency from DC to AC, maximal active power inject power to AC side, maximal reactive power inject power to AC side, maximal reactive power inject power to DC side
"""
from numpy import array


def con():
    ppc = {"version": '2'}
    ppc["baseMVA"] = 1000
    ppc["con"] = array([
        [1, 1, 0.9, 0.9, 1000, 1000, 1000],
        [5, 10, 0.9, 0.9, 1000, 1000, 1000],
        [10, 20, 0.9, 0.9, 1000, 1000, 1000],
    ])

    return ppc
