"""
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'*x + sum(p_s*Q_s(x))
			x
			s.t. A*x<=b,
			     Aeq*x==beq,   x \in [lb,ub]
			where Q_s(x)=min q_s'*y
			             y
			             s.t. W_s*y = h_s-T_s*x
			             y \in R^+
References:
    [1]Benders Decomposition for Solving Two-stage Stochastic Optimization Models
    https://www.ima.umn.edu/materials/2015-2016/ND8.1-12.16/25378/Luedtke-spalgs.pdf
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 8 Feb 2018
@version: 0.1

notes:
1) The data structure is based on the numpy and scipy
2) This algorithm should be extended for further version to solve the jointed chance constrained stochastic programming
3) In this test algorithm, Mosek is adpoted. https://www.mosek.com/
4) In the second stage optimization, the dual problem is solved, so that only the gurobi problem is needed to solve the problem.
5) The multi-cuts version Benders decomposition is adopted.
"""


def main(c, A, b, Aeq, beq, vtype, ps, qs, Ws, hs, Ts):
    """
    The standard input format for Benders decomposition problem
    :param c: Cost parameter for the first stage optimization
    :param A: Inequality constraint matrix for the first stage optimization
    :param b: Inequality constraint parameters for the first stage optimization
    :param Aeq: Equality constraint matrix for the first stage optimization
    :param beq: Equality constraint parameters for the first stage optimization
    :param vtype: The type for the first stage optimization problems
    :param ps: Probability for the second stage optimization problem under scenario s
    :param qs: Cost parameters for the second stage optimization problem
    :param Ws: Equality constraint parameters for the second stage optimization
    :param hs: Equality constraint parameters for the second stage optimization
    :param Ts: Equality constraint matrix between the first stage and the second stage optimization
    :return: The obtained solution for the first stage optimization
    """
    # 1) Input check for the first stage optimization
    
    # 2) Reformulate the first stage optimization problem
    # 2.1) Estimate the boundary of the first stage optimization problem.
    # 2.2) Add additional variables to the first stage optimization problem

    # 3) Reformulate the second stage optimization problem
    # 3.1) Formulate the dual problem for each problem under dual problems

    # 4) Begin the iteration

    # 5) Check the output

    # 6) Return the solution
