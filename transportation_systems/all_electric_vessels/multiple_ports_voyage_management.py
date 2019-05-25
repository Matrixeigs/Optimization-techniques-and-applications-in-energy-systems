"""
Optimal voyage management among multiple ports
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg

"""

from numpy import zeros, concatenate, vstack, array
import os, platform
import pandas as pd
from solvers.mixed_integer_quadratic_solver_cplex import mixed_integer_quadratic_programming as miqp
from transportation_systems.all_electric_vessels.test_case import a0, a1, a2, PMIN, PMAX, b0, b1, b2
from transportation_systems.all_electric_vessels.test_case import Vfull, Vhalf, Vin_out, Vmin
from transportation_systems.all_electric_vessels.test_case import capacityEss, socMax, socMin, effCharing, \
    effDischaring, pchMax, pdcMax, PL_CRUISE, PL_FULL, PL_IN_OUT, PL_STOP, PUG_MAX, PUG_MIN, vBlock, PproBlock, mBlock, \
    nV

from transportation_systems.all_electric_vessels.test_case import transportation_network, Price_port


class OptimalVoyage():

    def __init__(self):
        self.pwd = os.getcwd()

    def problem_formulaiton(self, networks=transportation_network()):
        """
        Problem formulation for optimal voyage among multiple ports
        :param networks:
        :return:
        """
        DIS = networks["voyage"][:, 2]  # Distance matrix
        NPORTs = networks["ports"].shape[0]  # Number of ports
        NYs = networks["voyage"].shape[0]  # Number of voyage
        T = Price_port.shape[0]  # T time slots
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

        NX = Vn + 1
        nx = NX * T
        lb = zeros(nx)
        ub = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ['c'] * nx
        for i in range(T):
            # ALPHA_A2S
            for j in range(NYs):
                lb[i * NX + ALPHA_A2S0 + j] = 0
                ub[i * NX + ALPHA_A2S0 + j] = 1
                c[i * NX + ALPHA_A2S0 + j] = 0
                q[i * NX + ALPHA_A2S0 + j] = 0
                vtypes[i * NX + ALPHA_A2S0 + j] = 'b'
            # ALPHA_S2D
            for j in range(NYs):
                lb[i * NX + ALPHA_S2D0 + j] = 0
                ub[i * NX + ALPHA_S2D0 + j] = 1
                c[i * NX + ALPHA_S2D0 + j] = 0
                q[i * NX + ALPHA_S2D0 + j] = 0
                vtypes[i * NX + ALPHA_S2D0 + j] = 'b'
            # ALPHA_D2A
            for j in range(NYs):
                lb[i * NX + ALPHA_D2A0 + j] = 0
                ub[i * NX + ALPHA_D2A0 + j] = 1
                c[i * NX + ALPHA_D2A0 + j] = 0
                q[i * NX + ALPHA_D2A0 + j] = 0
                vtypes[i * NX + ALPHA_D2A0 + j] = 'b'
            # ALPHA_D2C
            for j in range(NYs):
                lb[i * NX + ALPHA_D2C0 + j] = 0
                ub[i * NX + ALPHA_D2C0 + j] = 1
                c[i * NX + ALPHA_D2C0 + j] = 0
                q[i * NX + ALPHA_D2C0 + j] = 0
                vtypes[i * NX + ALPHA_D2C0 + j] = 'b'
            # ALPHA_C2A
            for j in range(NYs):
                lb[i * NX + ALPHA_C2A0 + j] = 0
                ub[i * NX + ALPHA_C2A0 + j] = 1
                c[i * NX + ALPHA_C2A0 + j] = 0
                q[i * NX + ALPHA_C2A0 + j] = 0
                vtypes[i * NX + ALPHA_C2A0 + j] = 'b'
            # I_S
            for j in range(NPORTs):
                lb[i * NX + I_S + j] = 0
                ub[i * NX + I_S + j] = 1
                c[i * NX + I_S + j] = 0
                q[i * NX + I_S + j] = 0
                vtypes[i * NX + I_S + j] = 'b'
                if i == T - 1:
                    lb[i * NX + I_S + j] = I_Sn[j]  # Should stop at the end of voyage
            # I_D
            for j in range(NYs):
                lb[i * NX + I_D0 + j] = 0
                ub[i * NX + I_D0 + j] = 1
                c[i * NX + I_D0 + j] = 0
                q[i * NX + I_D0 + j] = 0
                vtypes[i * NX + I_D0 + j] = 'b'
            # I_C
            for j in range(NYs):
                lb[i * NX + I_C0 + j] = 0
                ub[i * NX + I_C0 + j] = 1
                c[i * NX + I_C0 + j] = 0
                q[i * NX + I_C0 + j] = 0
                vtypes[i * NX + I_C0 + j] = 'b'
            # I_A
            for j in range(NYs):
                lb[i * NX + I_A0 + j] = 0
                ub[i * NX + I_A0 + j] = 1
                c[i * NX + I_A0 + j] = 0
                q[i * NX + I_A0 + j] = 0
                vtypes[i * NX + I_A0 + j] = 'b'
            # I_C_F
            for j in range(NYs):
                lb[i * NX + I_C_F0 + j] = 0
                ub[i * NX + I_C_F0 + j] = 1
                c[i * NX + I_C_F0 + j] = 0
                q[i * NX + I_C_F0 + j] = 0
                vtypes[i * NX + I_C_F0 + j] = 'b'
            # I_C_H
            for j in range(NYs):
                lb[i * NX + I_C_H0 + j] = 0
                ub[i * NX + I_C_H0 + j] = 1
                c[i * NX + I_C_H0 + j] = 0
                q[i * NX + I_C_H0 + j] = 0
                vtypes[i * NX + I_C_H0 + j] = 'b'
            # Ig
            for j in range(ng):
                lb[i * NX + I_G0 + j] = 0
                ub[i * NX + I_G0 + j] = 1
                c[i * NX + I_G0 + j] = a0[j]
                q[i * NX + I_G0 + j] = 0
                vtypes[i * NX + I_G0 + j] = 'b'
            # Pg
            for j in range(ng):
                lb[i * NX + P_G0 + j] = 0
                ub[i * NX + P_G0 + j] = PMAX[j]
                c[i * NX + P_G0 + j] = a1[j]
                q[i * NX + P_G0 + j] = a2[j]
            # PESS_DC
            lb[i * NX + PESS_DC] = 0
            ub[i * NX + PESS_DC] = pdcMax
            c[i * NX + PESS_DC] = 0
            q[i * NX + PESS_DC] = 0
            # PESS_CH
            lb[i * NX + PESS_CH] = 0
            ub[i * NX + PESS_CH] = pchMax
            c[i * NX + PESS_CH] = 0
            q[i * NX + PESS_CH] = 0
            # IESS_DC
            lb[i * NX + IESS_DC] = 0
            ub[i * NX + IESS_DC] = 1
            c[i * NX + IESS_DC] = 0
            q[i * NX + IESS_DC] = 0
            vtypes[i * NX + IESS_DC] = "b"
            # EESS
            lb[i * NX + EESS] = capacityEss * socMin
            ub[i * NX + EESS] = capacityEss * socMax
            c[i * NX + EESS] = 0
            q[i * NX + EESS] = 0
            # PUG
            for j in range(NPORTs):
                lb[i * NX + PUG0 + j] = PUG_MIN
                ub[i * NX + PUG0 + j] = PUG_MAX
                c[i * NX + PUG0 + j] = Price_port[i, 1]
                q[i * NX + PUG0 + j] = 0
            # PL
            lb[i * NX + PL] = 0
            ub[i * NX + PL] = max([PL_STOP, PL_IN_OUT, PL_FULL, PL_CRUISE])
            c[i * NX + PL] = 0
            q[i * NX + PL] = 0
            # PPRO
            lb[i * NX + PPRO] = 0
            ub[i * NX + PPRO] = sum(PMAX)
            c[i * NX + PPRO] = 0
            q[i * NX + PPRO] = 0
            # V
            for j in range(NYs):
                lb[i * NX + V0 + j] = 0
                ub[i * NX + V0 + j] = Vfull
                c[i * NX + V0 + j] = 0
                q[i * NX + V0 + j] = 0

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
            A_temp[i, i * NX + IESS_DC] = -pdcMax
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 21
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_CH] = 1
            A_temp[i, i * NX + IESS_DC] = pchMax
            b_temp[i] = pchMax
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 22
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + EESS] = 1
            Aeq_temp[i, i * NX + PESS_DC] = 1 / effDischaring
            Aeq_temp[i, i * NX + PESS_CH] = effCharing
            if i == 0:
                beq_temp[i] = capacityEss * 0.5
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

        ## Problem solving
        (x, obj, success) = miqp(c, zeros(nx), Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)
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
                    "Ppro": Ppro
                    }

        # save the results into excel file
        if platform.system() == "Windows":
            writer = pd.ExcelWriter(self.pwd + r"\result.xlsx", float_format="10.4%f", index=True)
        else:
            writer = pd.ExcelWriter(self.pwd + "/result.xlsx", float_format="10.4%f", index=True)

        df = pd.DataFrame(array([solution["obj"]]))
        df.to_excel(writer, sheet_name='obj')
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
    optimal_voyage = OptimalVoyage()
    sol = optimal_voyage.problem_formulaiton(networks=transportation_network())
