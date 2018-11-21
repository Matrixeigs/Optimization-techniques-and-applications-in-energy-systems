
"""Sioux Falls transportation network.
"""

from numpy import array

def siouxfalls():
    '''
    https: // github.com / bstabler / TransportationNetworks
    :return:
    '''
    tsc = {"version": '2'}

    # Transportaton system case
    # node data (??? The coordinates of x, y)
    # node x y
    NODE_I = 0
    tsc['node_columns'] = ['node_i', 'x', 'y']
    tsc['node'] = array([
        [1, 50000, 510000],
        [2, 320000, 510000],
        [3, 50000, 440000],
        [4, 130000, 440000],
        [5, 220000, 440000],
        [6, 320000, 440000],
        [7, 420000, 380000],
        [8, 320000, 380000],
        [9, 220000, 380000],
        [10, 220000, 320000],
        [11, 130000, 320000],
        [12, 50000, 320000],
        [13, 50000, 50000],
        [14, 130000, 190000],
        [15, 220000, 190000],
        [16, 320000, 320000],
        [17, 320000, 260000],
        [18, 420000, 320000],
        [19, 320000, 190000],
        [20, 320000, 50000],
        [21, 220000, 50000],
        [22, 220000, 130000],
        [23, 130000, 130000],
        [24, 130000, 50000],
    ])
   

    # Init node Term node 	Capacity 	Length 	Free Flow Time 	B	Power
    # Speed limit 	Toll 	Type	;
    INIT_NODE = 0;
    TERM_NODE = 1;
    tsc['edge_columns'] = ['init_node', 'term_node', 'capacity', 'length',
                           'free_flow_time', 'b', 'power', 'speed_limit',
                           'toll', 'type']
    tsc['edge'] = array([
        [1, 2, 25900.20064, 6, 6, 0.15, 4, 0, 0, 1],
        [1, 3, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [2, 1, 25900.20064, 6, 6, 0.15, 4, 0, 0, 1],
        [2, 6, 4958.180928, 5, 5, 0.15, 4, 0, 0, 1],
        [3, 1, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [3, 4, 17110.52372, 4, 4, 0.15, 4, 0, 0, 1],
        [3, 12, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [4, 3, 17110.52372, 4, 4, 0.15, 4, 0, 0, 1],
        [4, 5, 17782.7941, 2, 2, 0.15, 4, 0, 0, 1],
        [4, 11, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1],
        [5, 4, 17782.7941, 2, 2, 0.15, 4, 0, 0, 1],
        [5, 6, 4947.995469, 4, 4, 0.15, 4, 0, 0, 1],
        [5, 9, 10000, 5, 5, 0.15, 4, 0, 0, 1],
        [6, 2, 4958.180928, 5, 5, 0.15, 4, 0, 0, 1],
        [6, 5, 4947.995469, 4, 4, 0.15, 4, 0, 0, 1],
        [6, 8, 4898.587646, 2, 2, 0.15, 4, 0, 0, 1],
        [7, 8, 7841.81131, 3, 3, 0.15, 4, 0, 0, 1],
        [7, 18, 23403.47319, 2, 2, 0.15, 4, 0, 0, 1],
        [8, 6, 4898.587646, 2, 2, 0.15, 4, 0, 0, 1],
        [8, 7, 7841.81131, 3, 3, 0.15, 4, 0, 0, 1],
        [8, 9, 5050.193156, 10, 10, 0.15, 4, 0, 0, 1],
        [8, 16, 5045.822583, 5, 5, 0.15, 4, 0, 0, 1],
        [9, 5, 10000, 5, 5, 0.15, 4, 0, 0, 1],
        [9, 8, 5050.193156, 10, 10, 0.15, 4, 0, 0, 1],
        [9, 10, 13915.78842, 3, 3, 0.15, 4, 0, 0, 1],
        [10, 9, 13915.78842, 3, 3, 0.15, 4, 0, 0, 1],
        [10, 11, 10000, 5, 5, 0.15, 4, 0, 0, 1],
        [10, 15, 13512.00155, 6, 6, 0.15, 4, 0, 0, 1],
        [10, 16, 4854.917717, 4, 4, 0.15, 4, 0, 0, 1],
        [10, 17, 4993.510694, 8, 8, 0.15, 4, 0, 0, 1],
        [11, 4, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1],
        [11, 10, 10000, 5, 5, 0.15, 4, 0, 0, 1],
        [11, 12, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1],
        [11, 14, 4876.508287, 4, 4, 0.15, 4, 0, 0, 1],
        [12, 3, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [12, 11, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1],
        [12, 13, 25900.20064, 3, 3, 0.15, 4, 0, 0, 1],
        [13, 12, 25900.20064, 3, 3, 0.15, 4, 0, 0, 1],
        [13, 24, 5091.256152, 4, 4, 0.15, 4, 0, 0, 1],
        [14, 11, 4876.508287, 4, 4, 0.15, 4, 0, 0, 1],
        [14, 15, 5127.526119, 5, 5, 0.15, 4, 0, 0, 1],
        [14, 23, 4924.790605, 4, 4, 0.15, 4, 0, 0, 1],
        [15, 10, 13512.00155, 6, 6, 0.15, 4, 0, 0, 1],
        [15, 14, 5127.526119, 5, 5, 0.15, 4, 0, 0, 1],
        [15, 19, 14564.75315, 3, 3, 0.15, 4, 0, 0, 1],
        [15, 22, 9599.180565, 3, 3, 0.15, 4, 0, 0, 1],
        [16, 8, 5045.822583, 5, 5, 0.15, 4, 0, 0, 1],
        [16, 10, 4854.917717, 4, 4, 0.15, 4, 0, 0, 1],
        [16, 17, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1],
        [16, 18, 19679.89671, 3, 3, 0.15, 4, 0, 0, 1],
        [17, 10, 4993.510694, 8, 8, 0.15, 4, 0, 0, 1],
        [17, 16, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1],
        [17, 19, 4823.950831, 2, 2, 0.15, 4, 0, 0, 1],
        [18, 7, 23403.47319, 2, 2, 0.15, 4, 0, 0, 1],
        [18, 16, 19679.89671, 3, 3, 0.15, 4, 0, 0, 1],
        [18, 20, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [19, 15, 14564.75315, 3, 3, 0.15, 4, 0, 0, 1],
        [19, 17, 4823.950831, 2, 2, 0.15, 4, 0, 0, 1],
        [19, 20, 5002.607563, 4, 4, 0.15, 4, 0, 0, 1],
        [20, 18, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1],
        [20, 19, 5002.607563, 4, 4, 0.15, 4, 0, 0, 1],
        [20, 21, 5059.91234, 6, 6, 0.15, 4, 0, 0, 1],
        [20, 22, 5075.697193, 5, 5, 0.15, 4, 0, 0, 1],
        [21, 20, 5059.91234, 6, 6, 0.15, 4, 0, 0, 1],
        [21, 22, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1],
        [21, 24, 4885.357564, 3, 3, 0.15, 4, 0, 0, 1],
        [22, 15, 9599.180565, 3, 3, 0.15, 4, 0, 0, 1],
        [22, 20, 5075.697193, 5, 5, 0.15, 4, 0, 0, 1],
        [22, 21, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1],
        [22, 23, 5000, 4, 4, 0.15, 4, 0, 0, 1],
        [23, 14, 4924.790605, 4, 4, 0.15, 4, 0, 0, 1],
        [23, 22, 5000, 4, 4, 0.15, 4, 0, 0, 1],
        [23, 24, 5078.508436, 2, 2, 0.15, 4, 0, 0, 1],
        [24, 13, 5091.256152, 4, 4, 0.15, 4, 0, 0, 1],
        [24, 21, 4885.357564, 3, 3, 0.15, 4, 0, 0, 1],
        [24, 23, 5078.508436, 2, 2, 0.15, 4, 0, 0, 1],
    ])

    # Index conversion to zero-base, if input of bus ID is one-based
    # For bus
    tsc['node'][:, NODE_I] -= 1
    # For branch
    tsc['edge'][:, [INIT_NODE, TERM_NODE]] -= 1


    return tsc