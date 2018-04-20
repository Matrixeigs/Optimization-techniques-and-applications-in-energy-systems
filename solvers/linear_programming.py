"""

A interior point method based linear programming method is provided.
The algorithm is obtained from the following reference
[1]Interior Point Methods and Linear Programming
https://www.cs.toronto.edu/~robere/paper/interiorpoint.pdf
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg

The standard format is shown as follows:

min c'*x
x

s.t. Ax=b
     x>=0

"""
def linear_programming(c,A,b):
    """
    The linear programming solver
    :param c: The objective value
    :param A: Linear matrix
    :param b: Constraint vector
    :return: solution status, solution, objective value
    """
    # Step 1: input check of the

