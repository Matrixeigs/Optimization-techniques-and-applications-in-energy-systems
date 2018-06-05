"""
Interval unit commitment
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""


class IntervalUnitCommitment():
    ""

    def __init__(self):
        self.name = "Interval Unit Commitment"

    def problem_formulation(self):
        # Hydro power unit
        IG = 0
        PG = 1
        RUG = 2
        RDG = 3
        VH = 4
        QH = 5
        # Load shedding part
        PLC = 6
        # Output of wind power unit
        PWG = 7
