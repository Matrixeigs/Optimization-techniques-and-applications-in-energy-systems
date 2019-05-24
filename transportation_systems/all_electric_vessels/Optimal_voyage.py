"""
Optimal voyage among multiple ports

For paper:
﻿Mathematical Models for All Electric Ship Scheduling: A Mathematical Problem Approach
The input is a time space network
"""

from numpy import zeros, array, concatenate, vstack, diag
import os

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
        DIS = networks["voyage"][1, 2]
        T = Price_port.shape[0]  # T time slots
        ng = len(PMIN)
        I_S0 = 1
        I_D0 = 0
        I_C0 = 0
        I_A0 = 0
        # Optimization problem formulation
        # 1)
        # The variables are sorted by time
        ALPHA_A2S = 0
        ALPHA_S2D = 1
        ALPHA_D2A = 2
        ALPHA_D2C = 3
        ALPHA_C2A = 4
        I_S = 5  # equals to I_ANC
        I_D = 6
        I_C = 7
        I_A = 8
        I_C_F = 9
        I_C_H = 10
        I_G0 = I_C_H + 1
        I_Gn = I_G0 + ng
        P_G0 = I_Gn + 1
        P_Gn = P_G0 + ng
        PESS_DC = P_Gn + 1
        PESS_CH = PESS_DC + 1
        IESS_DC = PESS_CH + 1
        EESS = IESS_DC + 1
        PUG = EESS + 1
        PL = PUG + 1
        PPRO = PL + 1
        V = PPRO + 1

        NX = V + 1
        nx = NX * T
        lb = zeros(nx)
        ub = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ['c'] * nx
        for i in range(T):
            # 0
            lb[i * NX + ALPHA_A2S] = 0
            ub[i * NX + ALPHA_A2S] = 1
            c[i * NX + ALPHA_A2S] = 0
            q[i * NX + ALPHA_A2S] = 0
            vtypes[i * NX + ALPHA_A2S] = 'b'
            # 0
            lb[i * NX + ALPHA_S2D] = 0
            ub[i * NX + ALPHA_S2D] = 1
            c[i * NX + ALPHA_S2D] = 0
            q[i * NX + ALPHA_S2D] = 0
            vtypes[i * NX + ALPHA_S2D] = 'b'
            # 1
            lb[i * NX + ALPHA_D2A] = 0
            ub[i * NX + ALPHA_D2A] = 1
            c[i * NX + ALPHA_D2A] = 0
            q[i * NX + ALPHA_D2A] = 0
            vtypes[i * NX + ALPHA_D2A] = 'b'
            # 2
            lb[i * NX + ALPHA_D2C] = 0
            ub[i * NX + ALPHA_D2C] = 1
            c[i * NX + ALPHA_D2C] = 0
            q[i * NX + ALPHA_D2C] = 0
            vtypes[i * NX + ALPHA_D2C] = 'b'
            # 3
            lb[i * NX + ALPHA_C2A] = 0
            ub[i * NX + ALPHA_C2A] = 1
            c[i * NX + ALPHA_C2A] = 0
            q[i * NX + ALPHA_C2A] = 0
            vtypes[i * NX + ALPHA_C2A] = 'b'
            # 4
            lb[i * NX + I_S] = 0
            ub[i * NX + I_S] = 1
            c[i * NX + I_S] = 0
            q[i * NX + I_S] = 0
            vtypes[i * NX + I_S] = 'b'
            if i == T - 1:
                lb[i * NX + I_S] = 1  # Should stop at the end of voyage
            # 5
            lb[i * NX + I_D] = 0
            ub[i * NX + I_D] = 1
            c[i * NX + I_D] = 0
            q[i * NX + I_D] = 0
            vtypes[i * NX + I_D] = 'b'
            # 6
            lb[i * NX + I_C] = 0
            ub[i * NX + I_C] = 1
            c[i * NX + I_C] = 0
            q[i * NX + I_C] = 0
            vtypes[i * NX + I_C] = 'b'
            # 7
            lb[i * NX + I_A] = 0
            ub[i * NX + I_A] = 1
            c[i * NX + I_A] = 0
            q[i * NX + I_A] = 0
            vtypes[i * NX + I_A] = 'b'
            # 9
            lb[i * NX + I_C_F] = 0
            ub[i * NX + I_C_F] = 1
            c[i * NX + I_C_F] = 0
            q[i * NX + I_C_F] = 0
            vtypes[i * NX + I_C_F] = 'b'
            # 10
            lb[i * NX + I_C_H] = 0
            ub[i * NX + I_C_H] = 1
            c[i * NX + I_C_H] = 0
            q[i * NX + I_C_H] = 0
            vtypes[i * NX + I_C_H] = 'b'
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
            lb[i * NX + PUG] = PUG_MIN
            ub[i * NX + PUG] = PUG_MAX
            c[i * NX + PUG] = Price_port[i, 0]
            q[i * NX + PUG] = 0
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
            lb[i * NX + V] = 0
            ub[i * NX + V] = Vfull
            c[i * NX + V] = 0
            q[i * NX + V] = 0

        # Constraints set
        # 1) Status change constraint
        # equation 5
        Aeq = zeros((T, nx))
        beq = zeros(T)
        for i in range(T):
            Aeq[i, i * NX + I_S] = 1
            Aeq[i, i * NX + ALPHA_A2S] = -1
            Aeq[i, i * NX + ALPHA_S2D] = 1
            if i == 0:
                beq[i] = I_S0
            else:
                Aeq[i, (i - 1) * NX + I_S] = -1

        # equation 6
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_D] = 1
            Aeq_temp[i, i * NX + ALPHA_S2D] = -1
            Aeq_temp[i, i * NX + ALPHA_D2A] = 1
            Aeq_temp[i, i * NX + ALPHA_D2C] = 1
            if i == 0:
                beq_temp[i] = I_D0
            else:
                Aeq_temp[i, (i - 1) * NX + I_D] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # # equation 7
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_C] = 1
            Aeq_temp[i, i * NX + ALPHA_D2C] = -1
            Aeq_temp[i, i * NX + ALPHA_C2A] = 1
            if i == 0:
                beq_temp[i] = I_C0
            else:
                Aeq_temp[i, (i - 1) * NX + I_C] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # # equation 8
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_A] = 1
            Aeq_temp[i, i * NX + ALPHA_D2A] = -1
            Aeq_temp[i, i * NX + ALPHA_C2A] = -1
            Aeq_temp[i, i * NX + ALPHA_A2S] = 1
            if i == 0:
                beq_temp[i] = I_A0
            else:
                Aeq_temp[i, (i - 1) * NX + I_A] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 9
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            A[i, i * NX + ALPHA_A2S] = 1
            A[i, i * NX + ALPHA_D2A] = 1
            A[i, i * NX + ALPHA_S2D] = 1
            A[i, i * NX + ALPHA_C2A] = 1
            A[i, i * NX + ALPHA_D2C] = 1
            b[i] = 1
        # equation 14
        Aeq_temp = zeros((1, nx))
        beq_temp = zeros(1)
        for i in range(T):
            Aeq_temp[0, i * NX + V] = 1
        beq_temp[0] = DIS
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 15
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_S] = 1
            Aeq_temp[i, i * NX + I_A] = 1
            Aeq_temp[i, i * NX + I_D] = 1
            Aeq_temp[i, i * NX + I_C] = 1
            beq_temp[i] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 16
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_C_F] = 1
            Aeq_temp[i, i * NX + I_C_H] = 1
            Aeq_temp[i, i * NX + I_C] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 17
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + V] = 1
            A_temp[i, i * NX + I_C_F] = -Vfull
            A_temp[i, i * NX + I_C_H] = -Vhalf
            A_temp[i, i * NX + I_D] = -Vin_out
            A_temp[i, i * NX + I_A] = -Vin_out
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 18
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + V] = -1
            A_temp[i, i * NX + I_C_F] = Vhalf
            A_temp[i, i * NX + I_C_H] = Vin_out
            A_temp[i, i * NX + I_D] = Vmin
            A_temp[i, i * NX + I_A] = Vmin
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 19
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + PUG] = 1
            for j in range(ng):
                Aeq_temp[i, i * NX + P_G0 + j] = 1
            Aeq_temp[i, i * NX + PESS_DC] = 1
            Aeq_temp[i, i * NX + PESS_CH] = -1
            Aeq_temp[i, i * NX + PL] = -1
            Aeq_temp[i, i * NX + PPRO] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 20
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + PL] = 1
            Aeq_temp[i, i * NX + I_C_F] = -PL_FULL
            Aeq_temp[i, i * NX + I_C_H] = -PL_CRUISE
            Aeq_temp[i, i * NX + I_A] = -PL_IN_OUT
            Aeq_temp[i, i * NX + I_D] = -PL_IN_OUT
            Aeq_temp[i, i * NX + I_S] = -PL_STOP
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 21
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
        # equation 23
        A_temp = zeros((T * ng, nx))
        b_temp = zeros(T * ng)
        for i in range(T):
            for j in range(ng):
                A_temp[i * ng + j, i * NX + I_G0 + j] = 1
                A_temp[i * ng + j, i * NX + I_S] = 1
                b_temp[i * ng + j] = 1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 25
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_DC] = 1
            A_temp[i, i * NX + IESS_DC] = -pdcMax
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 26
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PESS_CH] = 1
            A_temp[i, i * NX + IESS_DC] = pchMax
            b_temp[i] = pchMax
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # equation 27
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
        # equation 28
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PUG] = 1
            A_temp[i, i * NX + I_S] = -PUG_MAX
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX + PUG] = -1
            A_temp[i, i * NX + I_S] = PUG_MIN
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # piece-wise linear approximation
        A_temp = zeros((T * (nV - 1), nx))
        b_temp = zeros(T * (nV - 1))
        for i in range(T):
            for j in range(nV - 1):
                A_temp[i * (nV - 1) + j, i * NX + PPRO] = -1
                A_temp[i * (nV - 1) + j, i * NX + V] = mBlock[j]
                b_temp[i * (nV - 1) + j] = -PproBlock[j] + mBlock[j] * vBlock[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        ## Problem solving
        (x, obj, success) = miqp(c, diag(q), Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)
        # Obtain the solution
        alpha_A2S = zeros(T)
        alpha_S2D = zeros(T)
        alpha_D2A = zeros(T)
        alpha_D2C = zeros(T)
        alpha_C2A = zeros(T)
        i_S = zeros(T)
        i_D = zeros(T)
        i_C = zeros(T)
        i_A = zeros(T)
        i_C_F = zeros(T)
        i_C_H = zeros(T)
        i_G = zeros((T, ng))
        p_G = zeros((T, ng))
        Pess_DC = zeros(T)
        Pess_CH = zeros(T)
        Iess_DC = zeros(T)
        Eess = zeros(T)
        Pl = zeros(T)
        Pug = zeros(T)
        Ppro = zeros(T)

        v = zeros(T)

        for i in range(T):
            alpha_A2S[i] = x[i * NX + ALPHA_A2S]
            alpha_S2D[i] = x[i * NX + ALPHA_S2D]
            alpha_D2A[i] = x[i * NX + ALPHA_D2A]
            alpha_D2C[i] = x[i * NX + ALPHA_D2C]
            alpha_C2A[i] = x[i * NX + ALPHA_C2A]
            i_S[i] = x[i * NX + I_S]
            i_D[i] = x[i * NX + I_D]
            i_C[i] = x[i * NX + I_C]
            i_A[i] = x[i * NX + I_A]
            i_C_F[i] = x[i * NX + I_C_F]
            i_C_H[i] = x[i * NX + I_C_H]
            for j in range(ng):
                i_G[i, j] = x[i * NX + I_G0 + j]
                p_G[i, j] = x[i * NX + P_G0 + j]
            Pess_DC[i] = x[i * NX + PESS_DC]
            Pess_CH[i] = x[i * NX + PESS_CH]
            Iess_DC[i] = x[i * NX + IESS_DC]
            Eess[i] = x[i * NX + EESS]
            v[i] = x[i * NX + V]
            Pug[i] = x[i * NX + PUG]
            Pl[i] = x[i * NX + PL]
            Ppro[i] = x[i * NX + PPRO]


        return sol


if __name__ == "__main__":
    optimal_voyage = OptimalVoyage()
    sol = optimal_voyage.problem_formulaiton(networks=transportation_network())
