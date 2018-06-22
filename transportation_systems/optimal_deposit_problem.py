"""
Optimal deposit problem with discrete time steps
"""
from numpy import zeros, ones, concatenate
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
        connection_matrix = zeros(((nl + nb) * T, 2))
        for i in range(T):
            for j in range(nl):
                connection_matrix[i * (nl + nb) + j, F_BUS] = network["branch"][j, F_BUS] + i * nb
                connection_matrix[i * (nl + nb) + j, T_BUS] = network["branch"][j, T_BUS] + network["branch"][
                    j, TIME] * nb + i * nb

            for j in range(nb):
                connection_matrix[i * (nl + nb) + nl + j, F_BUS] = j + i * nb  # This time slot
                connection_matrix[i * (nl + nb) + nl + j, T_BUS] = j + (i + 1) * nb  # The next time slot
        # add two virtual nodes to represent the initial and end status of vehicles
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        for i in range(nb):
            temp = zeros((1, 2))
            temp[0, 1] = i + 1
            connection_matrix = concatenate([connection_matrix, temp])

        for i in range(nb):
            temp = zeros((1, 2))
            temp[0, 0] = nb * T + i + 1
            temp[0, 1] = nb * (T + 1) + 1
            connection_matrix = concatenate([connection_matrix, temp])

        # Delete the out of range lines
        connection_matrix = connection_matrix[find(connection_matrix[:, T_BUS] < (T + 1) * nb + 2), :]

        # Status transition matrix
        nl = connection_matrix.shape[0]
        status_matrix = zeros((T - 1, nl))
        for i in range(T - 1):
            for j in range(nl):
                if connection_matrix[j, F_BUS] >= i * nb + 1 and connection_matrix[j, F_BUS] < (i + 1) * nb + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb + 1 and connection_matrix[j, T_BUS] > (i + 1) * nb + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros(((T + 1) * nb + 2, nl))
        connection_matrix_t = zeros(((T + 1) * nb + 2, nl))

        for i in range((T + 1) * nb + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        nx = status_matrix.shape[1]
        lb = zeros((nx, 1))
        ub = ones((nx, 1))
        c = zeros((nx, 1))
        lb[find(connection_matrix[:, F_BUS] == 0)] = network["initial"]
        ub[find(connection_matrix[:, F_BUS] == 0)] = network["initial"]

        lb[find(connection_matrix[:, T_BUS] == (T + 1) * nb + 1)] = network["end"]
        ub[find(connection_matrix[:, T_BUS] == (T + 1) * nb + 1)] = network["end"]

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

        return xx, obj, status


if __name__ == "__main__":
    optimal_deposit_problem = OptimalDepositProblem()
    transportation_network_model = TransportationNetworkModel()
    test_case = case5.transportation_network()
    (xx, obj, status) = optimal_deposit_problem.run(test_case, 10)

    print(xx)
