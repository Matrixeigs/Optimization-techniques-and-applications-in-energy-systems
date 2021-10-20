"""
Optimal ess planning with navigation routing
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@Date: 2021 Oct 19
"""

from numpy import zeros, concatenate, vstack, array, ones, reshape
import os, platform
import pandas as pd
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp
from transportation_systems.all_electric_vessels.Australia import a0, a1, a2, PMIN, PMAX
from transportation_systems.all_electric_vessels.Australia import Vfull, Vhalf, Vin_out, Vmin
from transportation_systems.all_electric_vessels.Australia import capacityEss, socMax, socMin, effCharing, \
    effDischaring, pchMax, pdcMax, PL_CRUISE, PL_FULL, PL_IN_OUT, PL_STOP, PUG_MAX, PUG_MIN, vBlock, PproBlock, mBlock, \
    nV, EcapacityEss, PcapacityEss, CostEssE, CostEssP,soc0, LBlock, nP, nD, PBlock, EBlock

from numpy import random
from transportation_systems.all_electric_vessels.Australia import transportation_network, Price_port


class OptimalPlanningESS():

    def __init__(self):
        self.pwd = os.getcwd()

    def problem_formulaiton(self, networks=transportation_network()):
        """
        Problem formulation for optimal voyage among multiple ports
        :param networks:
        :return:
        """
        lam = 0
        beta = 0.95

        DIS = networks["voyage"][:, 2]  # Distance matrix
        NPORTs = networks["ports"].shape[0]  # Number of ports
        NYs = networks["voyage"].shape[0]  # Number of voyage
        T = Price_port.shape[0]  # T time slots
        ## Generate the scenarios
        #
        ns = 100
        delta = 0.1
        scenario = random.random((T,3, ns))
        price_scenario = zeros((T,NPORTs,ns))
        for s in range(ns):
            for t in range(T):
                price_scenario[t, 0, s] = Price_port[t, 0] * (1 - delta + 2 * scenario[t, 0, s] * delta)
                price_scenario[t, 1, s] = Price_port[t, 1] * (1 - delta + 2 * scenario[t, 0, s] * delta)
                price_scenario[t, 2, s] = Price_port[t, 2] * (1 - delta + 2 * scenario[t, 1, s] * delta)
                price_scenario[t, 3, s] = Price_port[t, 3] * (1 - delta + 2 * scenario[t, 1, s] * delta)
                price_scenario[t, 4, s] = Price_port[t, 4] * (1 - delta + 2 * scenario[t, 2, s] * delta)
        ws = ones(ns)/ns
        d_dro = 0.1
        ng = len(PMIN)
        I_S0 = [0] * NPORTs
        I_S0[0] = 1
        I_Sn = [0] * NPORTs
        I_Sn[-1] = 1
        i_D0 = [0] * NYs
        i_C0 = [0] * NYs
        i_A0 = [0] * NYs
        # Optimization problem formulation
        # 1)
        # The variables are sorted by time
        ALPHA_A2S0 = 0
        ALPHA_A2Sn = NYs
        ALPHA_S2D0 = ALPHA_A2Sn + 1
        ALPHA_S2Dn = ALPHA_S2D0 + NYs
        ALPHA_D2A0 = ALPHA_S2Dn + 1
        ALPHA_D2An = ALPHA_D2A0 + NYs
        ALPHA_D2C0 = ALPHA_D2An + 1
        ALPHA_D2Cn = ALPHA_D2C0 + NYs
        ALPHA_C2A0 = ALPHA_D2Cn + 1
        ALPHA_C2An = ALPHA_C2A0 + NYs
        I_D0 = ALPHA_C2An + 1
        I_D0n = I_D0 + NYs
        I_C0 = I_D0n + 1
        I_Cn = I_C0 + NYs
        I_A0 = I_Cn + 1
        I_An = I_A0 + NYs
        I_C_F0 = I_An + 1
        I_C_Fn = I_C_F0 + NYs
        I_C_H0 = I_C_Fn + 1
        I_C_Hn = I_C_H0 + NYs
        I_S = I_C_Hn + 1  # vector of in ports status
        I_G0 = I_S + NPORTs + 1
        I_Gn = I_G0 + ng
        P_G0 = I_Gn + 1
        P_Gn = P_G0 + ng
        PESS_DC = P_Gn + 1
        PESS_CH = PESS_DC + 1
        IESS_DC = PESS_CH + 1
        EESS = IESS_DC + 1
        PUG0 = EESS + 1
        PUGn = PUG0 + NPORTs
        PL = PUGn + 1
        PPRO = PL + 1
        V0 = PPRO + 1
        Vn = PPRO + NYs
        # degradation
        Lc = Vn + 1
        Aux = Lc + 1
        # The number of decision within each time period
        NX = Aux + nP*nD*3

        ESSCAP = NX*T
        PESSCAP = ESSCAP + 1

        # The price information
        Z = PESSCAP + 1
        Z_p = Z + 1
        Z_n = Z_p + ns
        PI = Z_n + ns
        VI = PI + 1
        ETA = VI + ns

        nx = ETA + 1
        lb = zeros(nx)
        ub = zeros(nx)
        c = zeros(nx)
        vtypes = ['c'] * nx
        for i in range(T):
            # ALPHA_A2S
            for j in range(NYs):
                lb[i * NX + ALPHA_A2S0 + j] = 0
                ub[i * NX + ALPHA_A2S0 + j] = 1
                vtypes[i * NX + ALPHA_A2S0 + j] = 'b'
            # ALPHA_S2D
            for j in range(NYs):
                lb[i * NX + ALPHA_S2D0 + j] = 0
                ub[i * NX + ALPHA_S2D0 + j] = 1
                vtypes[i * NX + ALPHA_S2D0 + j] = 'b'
            # ALPHA_D2A
            for j in range(NYs):
                lb[i * NX + ALPHA_D2A0 + j] = 0
                ub[i * NX + ALPHA_D2A0 + j] = 1
                vtypes[i * NX + ALPHA_D2A0 + j] = 'b'
            # ALPHA_D2C
            for j in range(NYs):
                lb[i * NX + ALPHA_D2C0 + j] = 0
                ub[i * NX + ALPHA_D2C0 + j] = 1
                c[i * NX + ALPHA_D2C0 + j] = 0
                vtypes[i * NX + ALPHA_D2C0 + j] = 'b'
            # ALPHA_C2A
            for j in range(NYs):
                lb[i * NX + ALPHA_C2A0 + j] = 0
                ub[i * NX + ALPHA_C2A0 + j] = 1
                c[i * NX + ALPHA_C2A0 + j] = 0
                vtypes[i * NX + ALPHA_C2A0 + j] = 'b'
            # I_S
            for j in range(NPORTs):
                lb[i * NX + I_S + j] = 0
                ub[i * NX + I_S + j] = 0
                c[i * NX + I_S + j] = 0
                vtypes[i * NX + I_S + j] = 'b'
                # if i == 0:
                #     lb[i * NX + I_S + 0] = 1  # Should stop at the end of voyage
                #     ub[i * NX + I_S + 0] = 1  # Should stop at the end of voyage
                if i == T - 1:
                    lb[i * NX + I_S + j] = I_Sn[j]  # Should stop at the end of voyage
                    ub[i * NX + I_S + j] = I_Sn[j]  # Should stop at the end of voyage
            # I_D
            for j in range(NYs):
                lb[i * NX + I_D0 + j] = 0
                ub[i * NX + I_D0 + j] = 1
                c[i * NX + I_D0 + j] = 0
                vtypes[i * NX + I_D0 + j] = 'b'
            # I_C
            for j in range(NYs):
                lb[i * NX + I_C0 + j] = 0
                ub[i * NX + I_C0 + j] = 1
                c[i * NX + I_C0 + j] = 0
                vtypes[i * NX + I_C0 + j] = 'b'
            # I_A
            for j in range(NYs):
                lb[i * NX + I_A0 + j] = 0
                ub[i * NX + I_A0 + j] = 1
                c[i * NX + I_A0 + j] = 0
                vtypes[i * NX + I_A0 + j] = 'b'
            # I_C_F
            for j in range(NYs):
                lb[i * NX + I_C_F0 + j] = 0
                ub[i * NX + I_C_F0 + j] = 1
                c[i * NX + I_C_F0 + j] = 0
                vtypes[i * NX + I_C_F0 + j] = 'b'
            # I_C_H
            for j in range(NYs):
                lb[i * NX + I_C_H0 + j] = 0
                ub[i * NX + I_C_H0 + j] = 0  # Do not allow half cruise
                c[i * NX + I_C_H0 + j] = 0
                vtypes[i * NX + I_C_H0 + j] = 'b'
            # Ig
            for j in range(ng):
                lb[i * NX + I_G0 + j] = 0
                ub[i * NX + I_G0 + j] = 1
                c[i * NX + I_G0 + j] =  a0[j]
                vtypes[i * NX + I_G0 + j] = 'b'
            # Pg
            for j in range(ng):
                lb[i * NX + P_G0 + j] = 0
                ub[i * NX + P_G0 + j] = PMAX[j]
                c[i * NX + P_G0 + j] =  a1[j]
            # PESS_DC
            lb[i * NX + PESS_DC] = 0
            ub[i * NX + PESS_DC] = PcapacityEss
            c[i * NX + PESS_DC] = 0
            # PESS_CH
            lb[i * NX + PESS_CH] = 0
            ub[i * NX + PESS_CH] = PcapacityEss
            c[i * NX + PESS_CH] = 0
            # IESS_DC
            lb[i * NX + IESS_DC] = 0
            ub[i * NX + IESS_DC] = 1
            c[i * NX + IESS_DC] = 0
            vtypes[i * NX + IESS_DC] = "b"
            # EESS
            lb[i * NX + EESS] = 0
            ub[i * NX + EESS] = socMax * EcapacityEss
            c[i * NX + EESS] = 0
            # PUG
            for j in range(NPORTs):
                lb[i * NX + PUG0 + j] = PUG_MIN
                ub[i * NX + PUG0 + j] = PUG_MAX
                # c[i * NX + PUG0 + j] = Price_port[i, 1]
            # PL
            lb[i * NX + PL] = 0
            ub[i * NX + PL] = max([PL_STOP, PL_IN_OUT, PL_FULL, PL_CRUISE])
            c[i * NX + PL] = 0
            # PPRO
            lb[i * NX + PPRO] = 0
            ub[i * NX + PPRO] = sum(PMAX)
            c[i * NX + PPRO] = 0
            # V
            for j in range(NYs):
                lb[i * NX + V0 + j] = 0
                ub[i * NX + V0 + j] = Vfull
                c[i * NX + V0 + j] = 0
            # Lc
            lb[i * NX + Lc] = 0
            ub[i * NX + Lc] = 1e10
            # Aux
            lb[i * NX + Aux: i * NX + Aux + nP*nD*3] = 0
            ub[i * NX + Aux: i * NX + Aux + nP*nD*3] = 1
            vtypes[i * NX + Aux + nP*nD: i * NX + Aux + nP*nD*3] = ["b"]*nP*nD*2
            # print(i)

        lb[ESSCAP] = 0
        ub[ESSCAP] = EcapacityEss
        c[ESSCAP] = CostEssE
        vtypes[ESSCAP] = 'd'

        lb[PESSCAP] = 0
        ub[PESSCAP] = PcapacityEss
        c[PESSCAP] = CostEssP
        vtypes[PESSCAP] = 'd'

        lb[Z] = 0
        ub[Z] = 1e10
        c[Z] = d_dro

        for s in range(ns):
            lb[Z_p + s] = 0
            ub[Z_p + s] = 1e10
            c[Z_p + s] = ws[s]

        for s in range(ns):
            lb[Z_n + s] = 0
            ub[Z_n + s] = 1e10
            c[Z_n + s] = -ws[s]

        lb[PI] = -1e10
        ub[PI] = 1e10
        c[PI] = 1

        for j in range(ns):
            lb[VI + j] = 0
            ub[VI + j] = 1e10

        lb[ETA] = -1e10
        ub[ETA] = 1e10
        c[ETA] = lam


        # Arrival
        # lb[41 * NX + I_S + 1] = 1
        # lb[103 * NX + I_S + 2] = 1
        # lb[127 * NX + I_S + 3] = 1
        # # Departure
        # lb[0 * NX + I_D0 + 0] = 1
        # lb[55 * NX + I_D0 + 1] = 1
        # lb[121 * NX + I_D0 + 2] = 1
        # lb[135 * NX + I_D0 + 3] = 1
        # Constraints set
        # 1) Status change constraint
        # equation 5
        Aeq = zeros((T * NPORTs, nx))
        beq = zeros(T * NPORTs)
        for i in range(T):
            for j in range(NPORTs):
                Aeq[i * NPORTs + j, i * NX + I_S + j] = 1
                for k in range(NYs):
                    if networks["voyage"][k, 0] == j:  # From
                        Aeq[i * NPORTs + j, i * NX + ALPHA_S2D0 + k] = 1
                    if networks["voyage"][k, 1] == j:  # To
                        Aeq[i * NPORTs + j, i * NX + ALPHA_A2S0 + k] = -1
                if i == 0:
                    beq[i * NPORTs + j] = I_S0[j]
                else:
                    Aeq[i * NPORTs + j, (i - 1) * NX + I_S + j] = -1

        # equation 6
        Aeq_temp = zeros((T * NYs, nx))
        beq_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                Aeq_temp[i * NYs + j, i * NX + I_D0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_S2D0 + j] = -1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_D2A0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_D2C0 + j] = 1
                if i == 0:
                    beq_temp[i * NYs + j] = i_D0[j]
                else:
                    Aeq_temp[i * NYs + j, (i - 1) * NX + I_D0 + j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # # equation 7
        Aeq_temp = zeros((T * NYs, nx))
        beq_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                Aeq_temp[i * NYs + j, i * NX + I_C0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_D2C0 + j] = -1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_C2A0 + j] = 1
                if i == 0:
                    beq_temp[i * NYs + j] = i_C0[j]
                else:
                    Aeq_temp[i * NYs + j, (i - 1) * NX + I_C0 + j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # # equation 8
        Aeq_temp = zeros((T * NYs, nx))
        beq_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                Aeq_temp[i * NYs + j, i * NX + I_A0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_D2A0 + j] = -1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_C2A0 + j] = -1
                Aeq_temp[i * NYs + j, i * NX + ALPHA_A2S0 + j] = 1
                if i == 0:
                    beq_temp[i * NYs + j] = i_A0[j]
                else:
                    Aeq_temp[i * NYs + j, (i - 1) * NX + I_A0 + j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 9
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            for j in range(NYs):
                A[i, i * NX + ALPHA_A2S0 + j] = 1
                A[i, i * NX + ALPHA_D2A0 + j] = 1
                A[i, i * NX + ALPHA_S2D0 + j] = 1
                A[i, i * NX + ALPHA_C2A0 + j] = 1
                A[i, i * NX + ALPHA_D2C0 + j] = 1
            b[i] = 1
        # equation 10
        Aeq_temp = zeros((NYs, nx))
        beq_temp = zeros(NYs)
        for i in range(NYs):
            for j in range(T):
                Aeq_temp[i, j * NX + V0 + i] = 1
                Aeq_temp[i, j * NX + ALPHA_S2D0 + i] = -DIS[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        A_temp = zeros((NYs, nx))
        b_temp = zeros(NYs)
        for i in range(NYs):
            for j in range(T):
                A_temp[i, j * NX + ALPHA_S2D0 + i] = 1
            b_temp[i] = 1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        # equation 11
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            for j in range(NPORTs):
                Aeq_temp[i, i * NX + I_S + j] = 1
            for j in range(NYs):
                Aeq_temp[i, i * NX + I_A0 + j] = 1
                Aeq_temp[i, i * NX + I_D0 + j] = 1
                Aeq_temp[i, i * NX + I_C0 + j] = 1
            beq_temp[i] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 12
        Aeq_temp = zeros((T * NYs, nx))
        beq_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                Aeq_temp[i * NYs + j, i * NX + I_C_F0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + I_C_H0 + j] = 1
                Aeq_temp[i * NYs + j, i * NX + I_C0 + j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 13
        A_temp = zeros((T * NYs, nx))
        b_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                A_temp[i * NYs + j, i * NX + V0 + j] = 1
                A_temp[i * NYs + j, i * NX + I_C_F0 + j] = -Vfull
                A_temp[i * NYs + j, i * NX + I_C_H0 + j] = -Vhalf
                A_temp[i * NYs + j, i * NX + I_D0 + j] = -Vin_out
                A_temp[i * NYs + j, i * NX + I_A0 + j] = -Vin_out
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 14
        A_temp = zeros((T * NYs, nx))
        b_temp = zeros(T * NYs)
        for i in range(T):
            for j in range(NYs):
                A_temp[i * NYs + j, i * NX + V0 + j] = -1
                A_temp[i * NYs + j, i * NX + I_C_F0 + j] = Vhalf
                A_temp[i * NYs + j, i * NX + I_C_H0 + j] = Vin_out
                A_temp[i * NYs + j, i * NX + I_D0 + j] = Vmin
                A_temp[i * NYs + j, i * NX + I_A0 + j] = Vmin
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation z >= z_p + z_n
        A_temp = zeros((ns, nx))
        b_temp = zeros(ns)
        for i in range(ns):
            A_temp[i, Z_p + i] = 1
            A_temp[i, Z_n + i] = 1
            A_temp[i, Z] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        # equation 13(d)
        A_temp = zeros((ns, nx))
        A_price = zeros((ns, nx))
        b_temp = zeros(ns)
        for s in range(ns):
            for j in range(NPORTs):
                for i in range(T):
                    A_temp[s, i * NX + PUG0 + j] = (1-lam) * price_scenario[i,j,s]
                    A_price[s, i * NX + PUG0 + j] = (1-lam) * price_scenario[i,j,s]
            A_temp[s, VI + s] = lam/(1-beta)
            A_temp[s, Z_n + s] = 1
            A_temp[s, Z_p + s] = -1
            A_temp[s, PI] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation (13e)
        A_temp = zeros((ns, nx))
        b_temp = zeros(ns)
        for i in range(ns):
            for j in range(NPORTs):
                for t in range(T):
                    A_temp[i, t * NX + PUG0 + j] = price_scenario[t,j,i]
            A_temp[i, ETA] = -1
            A_temp[i, VI + i] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        # equation 15
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            for j in range(NPORTs):
                Aeq_temp[i, i * NX + PUG0 + j] = 1
            for j in range(ng):
                Aeq_temp[i, i * NX + P_G0 + j] = 1
            Aeq_temp[i, i * NX + PESS_DC] = 1
            Aeq_temp[i, i * NX + PESS_CH] = -1
            Aeq_temp[i, i * NX + PL] = -1
            Aeq_temp[i, i * NX + PPRO] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 16
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + PL] = 1
            for j in range(NYs):
                Aeq_temp[i, i * NX + I_C_F0 + j] = -PL_FULL
                Aeq_temp[i, i * NX + I_C_H0 + j] = -PL_CRUISE
                Aeq_temp[i, i * NX + I_A0 + j] = -PL_IN_OUT
                Aeq_temp[i, i * NX + I_D0 + j] = -PL_IN_OUT
            for j in range(NPORTs):
                Aeq_temp[i, i * NX + I_S + j] = -PL_STOP
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 17
        A_temp = zeros((T * ng, nx))
        b_temp = zeros(T * ng)
        for i in range(T):
            for j in range(ng):
                A_temp[i * ng + j, i * NX + P_G0 + j] = 1
                A_temp[i * ng + j, i * NX + I_G0 + j] = -PMAX[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T * ng, nx))
        b_temp = zeros(T * ng)
        for i in range(T):
            for j in range(ng):
                A_temp[i * ng + j, i * NX + P_G0 + j] = -1
                A_temp[i * ng + j, i * NX + I_G0 + j] = PMIN[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 19
        A_temp = zeros((T * ng, nx))
        b_temp = zeros(T * ng)
        for i in range(T):
            for j in range(ng):
                A_temp[i * ng + j, i * NX + I_G0 + j] = 1
                for k in range(NPORTs):
                    A_temp[i * ng + j, i * NX + I_S + k] = 1
                b_temp[i * ng + j] = 1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 20
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_DC] = 1
            A_temp[i, i * NX + IESS_DC] = -PcapacityEss
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 21
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_CH] = 1
            A_temp[i, i * NX + IESS_DC] = PcapacityEss
            b_temp[i] = PcapacityEss
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 22
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + EESS] = 1
            Aeq_temp[i, i * NX + PESS_DC] = 1 / effDischaring
            Aeq_temp[i, i * NX + PESS_CH] = -effCharing
            if i == 0:
                Aeq_temp[i, ESSCAP] = -soc0
            else:
                Aeq_temp[i, (i - 1) * NX + EESS] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 23
        A_temp = zeros((T * NPORTs, nx))
        b_temp = zeros(T * NPORTs)
        for i in range(T):
            for j in range(NPORTs):
                A_temp[i * NPORTs + j, i * NX + PUG0 + j] = 1
                A_temp[i * NPORTs + j, i * NX + I_S + j] = -PUG_MAX
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T * NPORTs, nx))
        b_temp = zeros(T * NPORTs)
        for i in range(T):
            for j in range(NPORTs):
                A_temp[i * NPORTs + j, i * NX + PUG0 + j] = -1
                A_temp[i * NPORTs + j, i * NX + I_S + j] = PUG_MIN
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 24, the energy capacity constraint
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + EESS] = 1
            A_temp[i, ESSCAP] = -socMax
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + EESS] = -1
            A_temp[i, ESSCAP] = socMin
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_CH] = 1
            A_temp[i, PESSCAP] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_DC] = 1
            A_temp[i, PESSCAP] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        #
        # piece-wise linear approximation
        A_temp = zeros((T * (nV - 1), nx))
        b_temp = zeros(T * (nV - 1))
        for i in range(T):
            for j in range(nV - 1):
                A_temp[i * (nV - 1) + j, i * NX + PPRO] = -1
                for k in range(NYs):
                    A_temp[i * (nV - 1) + j, i * NX + V0 + k] = mBlock[j]
                b_temp[i * (nV - 1) + j] = -PproBlock[j] + mBlock[j] * vBlock[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # Pproload
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PPRO] = 1
            for j in range(NYs):
                A_temp[i, i * NX + I_C0 + j] = -PproBlock[-1]
                A_temp[i, i * NX + I_D0 + j] = -PproBlock[-1]
                A_temp[i, i * NX + I_A0 + j] = -PproBlock[-1]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # The relation among auxiliary variables
        A_temp = zeros((T*nP*nD, nx))
        b_temp = zeros(T*nP*nD)
        for t in range(T):
            for i in range(nP):
                for j in range(nD):
                    A_temp[t * nP * nD + i * nD + j, t * NX + Aux + i * nD + j] = 1
                    A_temp[t * nP * nD + i * nD + j, t * NX + Aux + nP*nD + i * nD + j] = -1
                    A_temp[t * nP * nD + i * nD + j, t * NX + Aux + 2*nP*nD + i * nD + j] = -1
                    if i>0:
                        if j>0:
                            A_temp[t * nP * nD + i * nD + j, t * NX + Aux + nP * nD + i * nD + j] = -1
                            A_temp[t * nP * nD + i * nD + j, t * NX + Aux + nP * nD + (i-1) * nD + j] = -1
                            A_temp[t * nP * nD + i * nD + j, t * NX + Aux + nP * nD * 2 + i * nD + j] = -1
                            A_temp[t * nP * nD + i * nD + j, t * NX + Aux + nP * nD * 2 + (i-1) * nD + j] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # The degradation cost
        Aeq_temp = zeros((T, nx))
        beq_temp = ones(T)
        for t in range(T):
            Aeq_temp[t,t*NX + Aux : t*NX + Aux + nP*nD] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # power range
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for t in range(T):
            for i in range(nP):
                for j in range(nD):
                    Aeq_temp[t, t*NX + Aux + i*nD+j] = PBlock[i]
            Aeq_temp[t, t * NX + PESS_CH] = -1
            Aeq_temp[t, t * NX + PESS_DC] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # energy capacity
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for t in range(T):
            for i in range(nP):
                for j in range(nD):
                    Aeq_temp[t, t*NX + Aux + i*nD+j] = EBlock[j]
            Aeq_temp[t, ESSCAP] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # lift loss
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for t in range(T):
            for i in range(nP):
                for j in range(nD):
                    Aeq_temp[t, t*NX + Aux + i*nD+j] = LBlock[j,i]
            Aeq_temp[t, t*NX + Lc] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        Aeq_temp = zeros((1, nx))
        beq_temp = zeros(1)
        Aeq_temp[0, ESSCAP] = -soc0
        Aeq_temp[0, (T-1) * NX + EESS] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        ## Problem solving
        # (x, obj, success) = milp(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)
        (x, obj, success) = milp(c, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)
        # Obtain the solution
        alpha_A2S = zeros((T, NYs))
        alpha_S2D = zeros((T, NYs))
        alpha_D2A = zeros((T, NYs))
        alpha_D2C = zeros((T, NYs))
        alpha_C2A = zeros((T, NYs))
        i_S = zeros((T, NPORTs))
        i_D = zeros((T, NYs))
        i_C = zeros((T, NYs))
        i_A = zeros((T, NYs))
        i_C_F = zeros((T, NYs))
        i_C_H = zeros((T, NYs))
        i_G = zeros((T, ng))
        p_G = zeros((T, ng))
        Pess_DC = zeros(T)
        Pess_CH = zeros(T)
        Iess_DC = zeros(T)
        Eess = zeros(T)
        Pl = zeros(T)
        Pug = zeros((T, NPORTs))
        Ppro = zeros(T)
        v = zeros((T, NYs))
        BLc = zeros(T)


        for i in range(T):
            for j in range(NYs):
                alpha_A2S[i, j] = x[i * NX + ALPHA_A2S0 + j]
                alpha_S2D[i, j] = x[i * NX + ALPHA_S2D0 + j]
                alpha_D2A[i, j] = x[i * NX + ALPHA_D2A0 + j]
                alpha_D2C[i, j] = x[i * NX + ALPHA_D2C0 + j]
                alpha_C2A[i, j] = x[i * NX + ALPHA_C2A0 + j]
                i_D[i, j] = x[i * NX + I_D0 + j]
                i_C[i, j] = x[i * NX + I_C0 + j]
                i_A[i, j] = x[i * NX + I_A0 + j]
                i_C_F[i, j] = x[i * NX + I_C_F0 + j]
                i_C_H[i, j] = x[i * NX + I_C_H0 + j]
                v[i, j] = x[i * NX + V0 + j]
            for j in range(NPORTs):
                i_S[i, j] = x[i * NX + I_S + j]
                Pug[i, j] = x[i * NX + PUG0 + j]
            for j in range(ng):
                i_G[i, j] = x[i * NX + I_G0 + j]
                p_G[i, j] = x[i * NX + P_G0 + j]
            Pess_DC[i] = x[i * NX + PESS_DC]
            Pess_CH[i] = x[i * NX + PESS_CH]
            Iess_DC[i] = x[i * NX + IESS_DC]
            Eess[i] = x[i * NX + EESS]
            Pl[i] = x[i * NX + PL]
            Ppro[i] = x[i * NX + PPRO]
            BLc[i] = x[i * NX + Lc]


        essCAP = x[ESSCAP]
        pessCAP = x[PESSCAP]
        z = x[Z]
        eta = x[ETA]
        pi = x[PI]
        vi = zeros(ns)
        z_n = zeros(ns)
        z_p = zeros(ns)
        for i in range(ns):
            vi[i] = x[VI+i]
            z_n[i] = x[Z_n+i]
            z_p[i] = x[Z_p+i]



        solution = {"obj": obj,
                    "success": success,
                    "i_S": i_S,
                    "i_D": i_D,
                    "i_C": i_C,
                    "i_A": i_A,
                    "i_C_F": i_C_F,
                    "i_C_H": i_C_H,
                    "i_G": i_G,
                    "p_G": p_G,
                    "Pess_DC": Pess_DC,
                    "Pess_CH": Pess_CH,
                    "Eess": Eess,
                    "v": v,
                    "Pug": Pug,
                    "Pl": Pl,
                    "Ppro": Ppro,
                    "essCAP": essCAP,
                    "pessCAP": pessCAP,
                    }

        # save the results into excel file
        if platform.system() == "Windows":
            writer = pd.ExcelWriter(self.pwd + r"\result.xlsx", float_format="10.4%f", index=True)
        else:
            writer = pd.ExcelWriter(self.pwd + "/result.xlsx", float_format="10.4%f", index=True)

        df = pd.DataFrame(array([solution["obj"]]))
        df.to_excel(writer, sheet_name='obj')
        df = pd.DataFrame(array([solution["essCAP"]]))
        df.to_excel(writer, sheet_name='EESS')
        df = pd.DataFrame(array([solution["pessCAP"]]))
        df.to_excel(writer, sheet_name='PESS')
        port_text = []
        for i in range(NPORTs): port_text.append("#{0} P".format(i + 1))
        df = pd.DataFrame(solution["i_S"])
        df.to_excel(writer, sheet_name='Berth_status', header=port_text)
        voyage_text = []
        for i in range(NYs): voyage_text.append("#{0} R".format(i))
        df = pd.DataFrame(solution["i_D"])
        df.to_excel(writer, sheet_name='Departure_status', header=voyage_text)
        df = pd.DataFrame(solution["i_C"])
        df.to_excel(writer, sheet_name='Cruise_status', header=voyage_text)
        df = pd.DataFrame(solution["i_A"])
        df.to_excel(writer, sheet_name='Arrival_status', header=voyage_text)
        df = pd.DataFrame(solution["i_C_H"])
        df.to_excel(writer, sheet_name='Cruise_half_status', header=voyage_text)
        df = pd.DataFrame(solution["i_C_F"])
        df.to_excel(writer, sheet_name='Cruise_full_status', header=voyage_text)
        gene_text = []
        for i in range(ng): gene_text.append("#{0} G".format(i))
        df = pd.DataFrame(solution["i_G"])
        df.to_excel(writer, sheet_name='Generator_status', header=gene_text)
        df = pd.DataFrame(solution["p_G"])
        df.to_excel(writer, sheet_name='Generator_output', header=gene_text)
        df = pd.DataFrame(solution["Pug"])
        df.to_excel(writer, sheet_name='Port_exchange', header=port_text)
        df = pd.DataFrame(solution["v"])
        df.to_excel(writer, sheet_name='vessel_speed', header=voyage_text)
        df = pd.DataFrame(solution["Ppro"])
        df.to_excel(writer, sheet_name='Pro_load')
        df = pd.DataFrame(solution["Pl"])
        df.to_excel(writer, sheet_name='service_load')
        df = pd.DataFrame(solution["Pess_DC"])
        df.to_excel(writer, sheet_name='ESS_discharging')
        df = pd.DataFrame(solution["Pess_CH"])
        df.to_excel(writer, sheet_name='ESS_charging')
        df = pd.DataFrame(solution["Eess"])
        df.to_excel(writer, sheet_name='ESS_energy_status')
        writer.save()

        return solution


if __name__ == "__main__":
    optimal_voyage = OptimalPlanningESS()
    sol = optimal_voyage.problem_formulaiton(networks=transportation_network())
