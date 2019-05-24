"""
Optimal voyage among multiple ports

For paper:
﻿Mathematical Models for All Electric Ship Scheduling: A Mathematical Problem Approach

The input is a time space network

"""

from numpy import zeros, array, concatenate, vstack
import os

from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
from transportation_systems.all_electric_vessels.test_case import a0, a1, a2, PMIN, PMAX, b0, b1, b2
from transportation_systems.all_electric_vessels.test_case import Vfull, Vhalf, Vin_out, Vmin
from transportation_systems.all_electric_vessels.test_case import capacityEss, socMax, socMin, effCharing, \
    effDischaring, pchMax, pdcMax

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

        # Optimization problem formulation
        # 1)
        # The variables are sorted by time
        ALPHA_S2D = 0
        ALPHA_D2A = 1
        ALPHA_D2C = 2
        ALPHA_C2A = 3
        I_S = 4
        I_D = 5
        I_C = 6
        I_A = 7
        I_ANC = 8
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
        V = EESS + 1

        NX = V
        nx = NX * T
        lb = zeros(nx)
        ub = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ['c'] * nx
        for i in range(T):
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
            # 8
            lb[i * NX + I_ANC] = 0
            ub[i * NX + I_ANC] = 1
            c[i * NX + I_ANC] = 0
            q[i * NX + I_ANC] = 0
            vtypes[i * NX + I_ANC] = 'b'
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
            Aeq[i, i * NX + ALPHA_D2A] = 1
            Aeq[i, i * NX + I_A] = -1
            Aeq[i, i * NX + I_D] = 1
        # equation 6
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + ALPHA_D2C] = 1
            Aeq_temp[i, i * NX + I_C] = -1
            Aeq_temp[i, i * NX + I_D] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 7
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + ALPHA_C2A] = 1
            Aeq_temp[i, i * NX + I_A] = -1
            Aeq_temp[i, i * NX + I_C] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 8
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + ALPHA_S2D] = 1
            Aeq_temp[i, i * NX + I_D] = -1
            Aeq_temp[i, i * NX + I_A] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 9
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(1, T):
            A[i, i * NX + ALPHA_D2A] = 1
            A[i, i * NX + ALPHA_D2C] = 1
            A[i, (i - 1) * NX + I_D] = 1
        # equation 10
        A_temp = zeros((T, nx))
        b = zeros(T)
        for i in range(1, T):
            A[i, i * NX + ALPHA_C2A] = 1
            A[i, i * NX + I_C] = -1
        # equation 11
        A_temp = zeros((T, nx))
        b = zeros(T)
        for i in range(1, T):
            A[i, i * NX + ALPHA_S2D] = 1
            A[i, i * NX + I_A] = -1
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
            Aeq_temp[i, i * NX + I_ANC] = 1
            Aeq_temp[i, i * NX + I_A] = 1
            Aeq_temp[i, i * NX + I_D] = 1
            Aeq_temp[i, i * NX + I_C] = 1
            beq_temp[i] = 1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # equation 15
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX + I_C_F] = 1
            Aeq_temp[i, i * NX + I_C_H] = 1
            Aeq_temp[i, i * NX + I_C] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        ## Problem solving
        sol = miqp(c, q, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lb, xmax=ub, vtypes=vtypes)



if __name__ == "__main__":
    optimal_voyage = OptimalVoyage()
    optimal_voyage.problem_formulaiton(networks=transportation_network())
