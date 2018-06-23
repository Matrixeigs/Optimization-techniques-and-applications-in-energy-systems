"""
Optimal deposit problem with discrete time steps
"""
from numpy import zeros, ones, concatenate, array
from transportation_systems.test_cases import case5, F_BUS, T_BUS, TIME, BUS_ID, D
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from transportation_systems.transportation_network_models import TransportationNetworkModel
from numpy import flatnonzero as find


class OptimalDepositProblem():
    def __init__(self):
        self.name = "Optimal deposit"

    def run(self, network, T):

        nb = network["bus"].shape[0]
        nl = network["branch"].shape[0]

        # Develop the connection matrix
        connection_matrix = zeros(((2 * nl + nb) * T, 3))
        weight = zeros(((2 * nl + nb) * T, 1))
        for i in range(T):
            for j in range(nl):
                # Add from matrix
                connection_matrix[i * (2 * nl + nb) + j, F_BUS] = network["branch"][j, F_BUS] + i * nb
                connection_matrix[i * (2 * nl + nb) + j, T_BUS] = network["branch"][j, T_BUS] + network["branch"][
                    j, TIME] * nb + i * nb
                weight[i * (2 * nl + nb) + j, 0] = 1
                connection_matrix[i * (2 * nl + nb) + j, TIME] = network["branch"][j, TIME]

            for j in range(nl):
                # Add to matrix
                connection_matrix[i * (2 * nl + nb) + j + nl, F_BUS] = network["branch"][j, T_BUS] + i * nb
                connection_matrix[i * (2 * nl + nb) + j + nl, T_BUS] = network["branch"][j, F_BUS] + network["branch"][
                    j, TIME] * nb + i * nb
                weight[i * (2 * nl + nb) + j + nl, 0] = 1

                connection_matrix[i * (2 * nl + nb) + j + nl, TIME] = network["branch"][j, TIME]

            for j in range(nb):
                connection_matrix[i * (2 * nl + nb) + 2 * nl + j, F_BUS] = j + i * nb  # This time slot
                connection_matrix[i * (2 * nl + nb) + 2 * nl + j, T_BUS] = j + (i + 1) * nb  # The next time slot

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb)
        connection_matrix = connection_matrix[index, :]
        weight = weight[index]

        # add two virtual nodes to represent the initial and end status of vehicles
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        for i in range(nb):
            temp = zeros((1, 3))
            temp[0, 1] = i + 1
            connection_matrix = concatenate([connection_matrix, temp])

        # Delete the out of range lines
        connection_matrix = connection_matrix[find(connection_matrix[:, T_BUS] < T * nb + 2), :]
        for i in range(nb):
            temp = zeros((1, 3))
            temp[0, 0] = nb * (T - 1) + i + 1
            temp[0, 1] = nb * T + 1
            connection_matrix = concatenate([connection_matrix, temp])

        # Status transition matrix
        nl = connection_matrix.shape[0]
        status_matrix = zeros((T, nl))
        for i in range(T):
            for j in range(nl):
                if connection_matrix[j, F_BUS] >= i * nb + 1 and connection_matrix[j, F_BUS] < (i + 1) * nb + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb + 1 and connection_matrix[j, T_BUS] > (i + 1) * nb + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros((T * nb + 2, nl))
        connection_matrix_t = zeros((T * nb + 2, nl))

        for i in range(T * nb + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        nx = status_matrix.shape[1]
        lb = zeros((nx, 1))
        ub = ones((nx, 1))
        c = zeros((nx, 1))
        c[0:weight.shape[0]] = -weight
        lb[find(connection_matrix[:, F_BUS] == 0)] = network["initial"]
        ub[find(connection_matrix[:, F_BUS] == 0)] = network["initial"]

        lb[find(connection_matrix[:, T_BUS] == T * nb + 1)] = network["end"]
        ub[find(connection_matrix[:, T_BUS] == T * nb + 1)] = network["end"]

        vtypes = ["b"] * nx
        # From and to constraints
        Aeq = connection_matrix_f - connection_matrix_t
        beq = zeros((Aeq.shape[0], 1))
        beq[0] = 1
        beq[-1] = -1

        # statue constraints
        Aeq_temp = status_matrix
        beq_temp = ones((status_matrix.shape[0], 1))

        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        (xx, obj, status) = lp(c, Aeq=Aeq, beq=beq, xmin=lb, xmax=ub, vtypes=vtypes)
        # Return the routine
        routine = []
        for i in range(nx - 2 * nb):
            if xx[i] > 0:
                if (connection_matrix[i, T_BUS] - connection_matrix[i, F_BUS]) % nb:
                    print(i)
                    space = zeros((1, 2))
                    space[0, 0] = (connection_matrix[i, F_BUS] - 1) % nb
                    space[0, 1] = (connection_matrix[i, T_BUS] - connection_matrix[i, TIME] * nb - 1) % nb
                    routine.append(space)

        xx = array(xx).reshape((nx, 1))
        return routine


if __name__ == "__main__":
    optimal_deposit_problem = OptimalDepositProblem()
    transportation_network_model = TransportationNetworkModel()
    test_case = case5.transportation_network()
    routine = optimal_deposit_problem.run(test_case, 24)

    print(routine)
