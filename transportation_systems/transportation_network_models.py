"""
The transportation networks models for discrete time step dispatch
"""

from numpy import zeros
from transportation_systems.test_cases import case5, F_BUS, T_BUS, TIME, BUS_ID, D
from numpy import flatnonzero as find


class TransportationNetworkModel():

    def __init__(self):
        self.name = "Transportation networks"

    def run(self, network, T):
        """
        Transportation network with discrete time step
        All buses are aggregated as one state
        :param network: The traffic network model
        :param T: The total time slots
        :return:
        """
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
        # Delete the out of range lines
        connection_matrix = connection_matrix[find(connection_matrix[:, T_BUS] < T * nb), :]
        # Status transition matrix
        nl = connection_matrix.shape[0]
        status_matrix = zeros((T - 1, nl))
        for i in range(T - 1):
            for j in range(nl):
                if connection_matrix[j, F_BUS] >= i * nb and connection_matrix[j, F_BUS] < (i + 1) * nb:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb and connection_matrix[j, T_BUS] > (i + 1) * nb:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros(((T - 2) * nb, nl))
        connection_matrix_t = zeros(((T - 2) * nb, nl))

        for i in range(nb, (T - 1) * nb):
            connection_matrix_f[i - nb, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i - nb, find(connection_matrix[:, T_BUS] == i)] = 1

        return connection_matrix_f, connection_matrix_t, status_matrix


if __name__ == "__main__":
    transportation_network_model = TransportationNetworkModel()
    test_case = case5.transportation_network()
    (connection_matrix_f, connection_matrix_t, status_matrix) = transportation_network_model.run(test_case, 3)
    print(connection_matrix_f)
