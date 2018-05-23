"""
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'*x + sum(p_s*Q_s(x))
			x
			s.t. A*x<=b,
			     Aeq*x==beq,   x \in [lb,ub]
			where Q_s(x)=min q_s'*ys
			             y
			             s.t. W_s*y = h_s-T_s*x
			             ys \in R^+
References:
    [1]Benders Decomposition for Solving Two-stage Stochastic Optimization Models
    https://www.ima.umn.edu/materials/2015-2016/ND8.1-12.16/25378/Luedtke-spalgs.pdf
    [2]http://www.iems.ucf.edu/qzheng/grpmbr/seminar/Yuping_Intro_to_BendersDecomp.pdf
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 8 Feb 2018
@version: 0.1

notes:
1) The data structure is based on the numpy and scipy
2) This algorithm should be extended for further version to solve the jointed chance constrained stochastic programming
3) In this test algorithm, Mosek is adpoted. https://www.mosek.com/
4) In the second stage optimization, the dual problem is solved, so that only the cplex problem is needed to solve the problem.
5) The multi-cuts version Benders decomposition is adopted.
"""
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from numpy import zeros, hstack, vstack, transpose, ones, inf, array
from copy import deepcopy
from solvers.benders_solvers import linear_programming as lp_dual


class BendersDecomposition():
    def __init__(self):
        self.name = "Benders decomposition"

    def main(self, c=None, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, vtype=None, ps=None, qs=None, Ws=None,
             hs=None, Ts=None):
        """
        The standard input format for Benders decomposition problem
        :param c: Cost parameter for the first stage optimization
        :param A: Inequality constraint matrix for the first stage optimization
        :param b: Inequality constraint parameters for the first stage optimization
        :param Aeq: Equality constraint matrix for the first stage optimization
        :param beq: Equality constraint parameters for the first stage optimization
        :param vtype: The type for the first stage optimization problems
        :param ps: Probability for the second stage optimization problem under scenario s
        :param qs: Cost parameters for the second stage optimization problem, a list of arrays
        :param Ws: Equality constraint parameters for the second stage optimization, a list of arrays
        :param hs: Equality constraint parameters for the second stage optimization
        :param Ts: Equality constraint matrix between the first stage and the second stage optimization
        :return: The obtained solution for the first stage optimization
        """
        # 1) Try to solve the first stage optimization problem
        model_first_stage = {"c": c,
                             "Aeq": Aeq,
                             "beq": beq,
                             "A": A,
                             "b": b,
                             "lb": lb,
                             "ub": ub,
                             "vtypes": vtype}

        sol_first_stage = BendersDecomposition.master_problem(self, model_first_stage)

        if sol_first_stage["status"] == 0:
            print("The master problem is infeasible!")
            return
        else:
            print("The master problem is feasible, the process continutes!")

        self.N = len(ps)  # The number of second stage decision variables

        self.nx_second_stage = Ws[0].shape[1]
        self.nx_first_stage = lb.shape[0]
        M = 10 ** 8
        model_second_stage = [0] * self.N

        for i in range(self.N):
            model_second_stage[i] = {"c": qs[i],
                                     "Aeq": Ws[i],
                                     "hs": hs[i],
                                     "Ts": Ts[i],
                                     "lb": zeros((self.nx_second_stage, 1))}
        # 2) Reformulate the first stage optimization problem
        # 2.1) Estimate the boundary of the first stage optimization problem.
        # 2.2) Add additional variables to the first stage optimization problem
        # Using the multiple cuts version
        model_master = deepcopy(model_first_stage)
        model_master["c"] = vstack([model_first_stage["c"], ps])
        if model_master["Aeq"] is not None:
            model_master["Aeq"] = hstack([model_first_stage["Aeq"], zeros((model_first_stage["Aeq"].shape[0], self.N))])
        if model_master["A"] is not None:
            model_master["A"] = hstack([model_first_stage["A"], zeros((model_first_stage["A"].shape[0], self.N))])

        if model_master["lb"] is not None:
            model_master["lb"] = vstack([model_first_stage["lb"], -ones((self.N, 1)) * M])
        else:
            model_master["lb"] = -ones((self.N + self.nx_first_stage, 1)) * M

        if model_master["ub"] is not None:
            model_master["ub"] = vstack([model_first_stage["ub"], ones((self.N, 1)) * M])
        else:
            model_master["ub"] = ones((self.N + self.nx_first_stage, 1)) * M

        model_master["vtypes"] = ["c"] * model_master["c"].shape[0]

        # 3) Reformulate the second stage optimization problem
        # 3.1) Formulate the dual problem for each problem under dual problems
        # The dual problem is solved
        x_first_stage = array(sol_first_stage["x"][0:self.nx_first_stage]).reshape(self.nx_first_stage, 1)
        model_second_stage = BendersDecomposition.sub_problems_update(self, model_second_stage, x_first_stage)

        sol_second_stage = [0] * self.N
        sol_second_stage_primal = [0] * self.N
        A_cuts = zeros((self.N, self.nx_first_stage + self.N))
        b_cuts = zeros((self.N, 1))

        for i in range(self.N):
            # Solve the dual problem
            sol_second_stage[i] = BendersDecomposition.sub_problem_dual(self, model_second_stage[i])
            # sol_second_stage_primal[i] = BendersDecomposition.sub_problem(self, model_second_stage[i])
            A_cuts[i, 0:self.nx_first_stage] = -transpose(
                transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
            b_cuts[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])
            if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                A_cuts[i, self.nx_first_stage + i] = -1
            # else add infeasible cuts
        Upper = [inf]
        Lower = sol_first_stage["objvalue"]
        Gap = [BendersDecomposition.gap_calculaiton(self, Upper[0], Lower)]
        eps = 10 ** -3
        iter_max = 1000
        iter = 0

        # 4) Begin the iteration
        while iter < iter_max:
            # Update the master problem
            if model_master["A"] is not None:
                model_master["A"] = vstack([model_master["A"], A_cuts])
            else:
                model_master["A"] = A_cuts
            if model_master["b"] is not None:
                model_master["b"] = vstack([model_master["b"], b_cuts])
            else:
                model_master["b"] = b_cuts
            # solve the master problem
            sol_first_stage = BendersDecomposition.master_problem(self, model_master)
            Lower = sol_first_stage["objvalue"]

            # update the second stage solution
            x_first_stage = array(sol_first_stage["x"][0:self.nx_first_stage]).reshape(self.nx_first_stage, 1)
            model_second_stage = BendersDecomposition.sub_problems_update(self, model_second_stage, x_first_stage)

            objvalue_second_stage = zeros((self.N, 1))

            sol_second_stage = [0] * self.N

            A_cuts = zeros((self.N, self.nx_first_stage + self.N))
            b_cuts = zeros((self.N, 1))

            for i in range(self.N):
                # Solve the dual problem
                sol_second_stage[i] = BendersDecomposition.sub_problem_dual(self, model_second_stage[i])

                A_cuts[i, 0:self.nx_first_stage] = -transpose(
                    transpose(model_second_stage[i]["Ts"]).dot(sol_second_stage[i]["x"]))
                b_cuts[i, 0] = -transpose(sol_second_stage[i]["x"]).dot(model_second_stage[i]["hs"])

                if sol_second_stage[i]["status"] == 1:  # if the primal problem is feasible, add feasible cuts
                    A_cuts[i, self.nx_first_stage + i] = -1
                    objvalue_second_stage[i, 0] = -sol_second_stage[i]["objvalue"]
                else:
                    objvalue_second_stage[i, 0] = inf

            Upper = transpose(x_first_stage).dot(model_first_stage["c"]) + transpose(objvalue_second_stage).dot(ps)

            Gap.append(BendersDecomposition.gap_calculaiton(self, Upper[0], Lower))
            print(Gap[-1][0])
            print(Lower)
            iter += 1

            if Gap[-1][0] < eps:
                break

        # x_first_stage = sol_first_stage["x"][0:self.nx_first_stage]
        #
        # x_second_stage = zeros((self.N, self.nx_second_stage))

        # for i in range(self.N):
        #     x_second_stage[i, :] = sol_second_stage[i]["x"]
        sol = {"objvalue": Upper,
               "x_first_stage": x_first_stage, }

        return sol

    def master_problem(self, model):
        """
        Solve the master problem
        :param model:
        :return:
        """
        (x, objvalue, status) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"], A=model["A"], b=model["b"],
                                   xmin=model["lb"], xmax=model["ub"], vtypes=model["vtypes"])

        sol = {"x": x,
               "objvalue": objvalue,
               "status": status}

        return sol

    def sub_problem(self, model):
        """
        Solve each slave problems
        :param model:
        :return:
        """
        (x, objvalue, status) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"], xmin=model["lb"])

        sol = {"x": x,
               "objvalue": objvalue,
               "status": status}

        return sol

    def sub_problem_dual(self, model):
        """
        Solve each slave problems
        :param model:
        :return:
        """
        (x, objvalue, status) = lp_dual(model["beq"], A=transpose(model["Aeq"]), b=model["c"])

        sol = {"x": x,
               "objvalue": -objvalue,
               "status": status}

        return sol

    def sub_problems_update(self, model, x):
        """

        :param model: The second stage models
        :param hs: The equality constraints under each stage
        :param Ts: The coupling constraints between the first stage and second stage constraints
        :return: hs-Ts*x
        """
        for i in range(self.N):
            model[i]["beq"] = model[i]["hs"] - model[i]["Ts"].dot(x)

        return model

    def gap_calculaiton(self, upper, lower):

        if lower != 0:
            gap = abs((upper - lower) / lower * 100)
        else:
            gap = inf

        if gap == inf:
            gap = [inf]

        return gap


if __name__ == "__main__":
    # c = array([2, 3, 0, 0]).reshape(4, 1)
    # Ts = array([[1, 2, -1, 0], [2, -1, 0, -1]])
    # hs = array([3, 4]).reshape(2, 1)
    # Ws = array([1, 3]).reshape(2, 1)
    # lb = zeros((4, 1))
    # ub = ones((4, 1)) * inf
    # qs = array([2]).reshape(1, 1)
    # benders_decomposition = BendersDecomposition()
    # sol = benders_decomposition.main(c=c, lb=lb, ub=ub, ps=[1], qs=[qs], hs=[hs], Ts=[Ts], Ws=[Ws])
    # print(sol)

    # The second test case
    c = array([1, 1]).reshape(2, 1)
    lb = zeros((2, 1))

    ps = array([1 / 3, 1 / 3, 1 / 3]).reshape(3, 1)
    hs = [0] * 3
    hs[0] = array([7, 4]).reshape(2, 1)
    hs[1] = array([7, 4]).reshape(2, 1)
    hs[2] = array([7, 4]).reshape(2, 1)

    Ws = [0] * 3
    Ws[0] = array([[1, 0, -1, 0], [1, 0, 0, -1]])
    Ws[1] = array([[1, 0, -1, 0], [1, 0, 0, -1]])
    Ws[2] = array([[1, 0, -1, 0], [1, 0, 0, -1]])

    Ts = [0] * 3
    Ts[0] = array([[1, 1], [1 / 3, 1]])
    Ts[1] = array([[5 / 2, 1], [2 / 3, 1]])
    Ts[2] = array([[4, 1], [1, 1]])

    qs = [0] * 3
    qs[0] = array([1, 1, 0, 0])
    qs[1] = array([1, 1, 0, 0])
    qs[2] = array([1, 1, 0, 0])

    benders_decomposition = BendersDecomposition()
    sol = benders_decomposition.main(c=c, lb=lb, ps=ps, qs=qs, hs=hs, Ts=Ts, Ws=Ws)
    print(sol)
