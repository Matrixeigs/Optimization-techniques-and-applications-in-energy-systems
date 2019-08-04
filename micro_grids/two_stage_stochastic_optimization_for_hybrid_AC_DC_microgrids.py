"""
Two-stage stochastic optimization for hybrid AC/DC microgrids under the uncertainty of loads
@author: Zhao Tianyang
@Date: 4 Aug 2019
@e-mail: matrixeigs@gmail.com

The hybrid AC/DC has multiple DGs, BICs, ESSs and renewable sources

"""
from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, lil_matrix

from micro_grids.idx_format_hybrid_AC_DC import PBIC_A2D,PBIC_D2A,PESS_CH0,PESS_DC0,PG0,PMESS,PPV0,PUG,NX_MG,QBIC,QG0,QUG

class TwoStageStochasticHybirdACDCMG():

    def __init__(self):
        self.name = "Two stage stochastic optimization for hybrid AC/DC microgrids"

    def main(self, microgrids):
        """
        Main function for hybrid AC DC microgrids
        :param microgrids:
        :return:
        """

    def problem_formulation_microgrid(self, mg, mess):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24
        nmes = self.nmes

        pmess_l = 0
        pmess_u = 0
        for i in range(nmes):
            pmess_l -= mess[i]["PCMAX"]
            pmess_u += mess[i]["PDMAX"]

        ## 1) boundary information and objective function
        nv = NX_MG * T
        lb = zeros(nv)
        ub = zeros(nv)
        c = zeros(nv)
        q = zeros(nv)
        vtypes = ["c"] * nv
        for t in range(T):
            ## 1.1) lower boundary
            lb[t * NX_MG + PG] = 0
            lb[t * NX_MG + QG] = mg["DG"]["QMIN"]
            lb[t * NX_MG + PUG] = 0
            lb[t * NX_MG + QUG] = mg["UG"]["QMIN"]
            lb[t * NX_MG + PBIC_DC2AC] = 0
            lb[t * NX_MG + PBIC_AC2DC] = 0
            lb[t * NX_MG + QBIC] = -mg["BIC"]["SMAX"]
            lb[t * NX_MG + PESS_CH] = 0
            lb[t * NX_MG + PESS_DC] = 0
            lb[t * NX_MG + EESS] = mg["ESS"]["EMIN"]
            lb[t * NX_MG + PPV] = 0
            lb[t * NX_MG + PMESS] = pmess_l
            ## 1.2) upper boundary
            ub[t * NX_MG + PG] = mg["DG"]["PMAX"]
            ub[t * NX_MG + QG] = mg["DG"]["QMAX"]
            ub[t * NX_MG + PUG] = mg["UG"]["PMAX"]
            ub[t * NX_MG + QUG] = mg["UG"]["QMAX"]
            ub[t * NX_MG + PBIC_DC2AC] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + PBIC_AC2DC] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + QBIC] = mg["BIC"]["SMAX"]
            ub[t * NX_MG + PESS_CH] = mg["ESS"]["PCH_MAX"]
            ub[t * NX_MG + PESS_DC] = mg["ESS"]["PDC_MAX"]
            ub[t * NX_MG + EESS] = mg["ESS"]["EMAX"]
            ub[t * NX_MG + PPV] = mg["PV"]["PROFILE"][t]
            ub[t * NX_MG + PMESS] = pmess_u
            ## 1.3) Objective functions
            c[t * NX_MG + PG] = mg["DG"]["COST_A"]
            c[t * NX_MG + PESS_CH] = mg["ESS"]["COST_OP"]
            c[t * NX_MG + PESS_DC] = mg["ESS"]["COST_OP"]
            c[t * NX_MG + PPV] = mg["PV"]["COST"]
            # c[t * NX_MG + PBIC_AC2DC] = mg["ESS"]["COST_OP"]
            # c[t * NX_MG + PBIC_DC2AC] = mg["ESS"]["COST_OP"]
            # c[t * NX_MG + PUG] = mg["DG"]["COST_A"]
            # c[t * NX_MG + PMESS] = 0.001
            ## 1.4) Upper and lower boundary information
            if t == T - 1:
                lb[t * NX_MG + EESS] = mg["ESS"]["E0"]
                ub[t * NX_MG + EESS] = mg["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = lil_matrix((T, nv))
        beq = zeros(T)
        for t in range(T):
            Aeq[t, t * NX_MG + PG] = 1
            Aeq[t, t * NX_MG + PUG] = 1
            Aeq[t, t * NX_MG + PBIC_AC2DC] = -1
            Aeq[t, t * NX_MG + PBIC_DC2AC] = mg["BIC"]["EFF_DC2AC"]
            beq[t] = mg["PD"]["AC"][t]
        # b) DC bus equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + PBIC_AC2DC] = mg["BIC"]["EFF_AC2DC"]
            Aeq_temp[t, t * NX_MG + PBIC_DC2AC] = -1
            Aeq_temp[t, t * NX_MG + PESS_CH] = -1
            Aeq_temp[t, t * NX_MG + PESS_DC] = 1
            Aeq_temp[t, t * NX_MG + PPV] = 1
            Aeq_temp[t, t * NX_MG + PMESS] = 1  # The power injection from mobile energy storage systems
            beq_temp[t] = mg["PD"]["DC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + QUG] = 1
            Aeq_temp[t, t * NX_MG + QBIC] = 1
            Aeq_temp[t, t * NX_MG + QG] = 1
            beq_temp[t] = mg["QD"]["AC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + EESS] = 1
            Aeq_temp[t, t * NX_MG + PESS_CH] = -mg["ESS"]["EFF_CH"]
            Aeq_temp[t, t * NX_MG + PESS_DC] = 1 / mg["ESS"]["EFF_DC"]
            if t == 0:
                beq_temp[t] = mg["ESS"]["E0"]
            else:
                Aeq_temp[t, (t - 1) * NX_MG + EESS] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequality constraints
        # There is no inequality constraint.

        # sol = milp(c, Aeq=Aeq, beq=beq, A=None, b=None, xmin=lb, xmax=ub)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lb,
                            "ub": ub,
                            "vtypes": vtypes,
                            "A": None,
                            "b": None,
                            "Aeq": Aeq,
                            "beq": beq
                            }

        return model_micro_grid