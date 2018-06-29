"""
Robust sample average sampling method for distributionally robust optimization

The second stage optimization problem is also based on the

sample average approximation mainly has two features
1) Asymptotic Convergence
2) Tractability
References on the robust sample average approximation
[1]Bertsimas, Dimitris, Vishal Gupta, and Nathan Kallus. "Robust sample average approximation." Mathematical Programming (2014): 1-66.
[2]

"""

from numpy import array

class RobustSampleAverageApproximation():

    def __init__(self):
        self.name = "Robust sample average approximation"

