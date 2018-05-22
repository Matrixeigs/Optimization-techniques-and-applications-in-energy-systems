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
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from numpy import zeros, hstack, vstack, multiply, transpose, ones, inf
from copy import deepcopy


class BendersDecomposition():
    def __init__(self):
        self.name = "Benders decomposition"

    def main(self, c, A, b, Aeq, beq, lb, ub, vtype, ps, qs, Ws, hs, Ts):
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
        self.nx_first_stage = Aeq.shape[1]
        M = 10 ^ 8

        model_second_stage = [0] * self.N
        for i in range(self.N):
            model_second_stage[i] = {"c": qs[i],
                                     "Aeq": Ws[i],
                                     "hs": hs[i],
                                     "Ts": Ts[i],
                                     "lb": zeros((self.nx_second_stage, 1))}

            # Solve the primal problem
            # sol_second_stage[i] = BendersDecomposition.sub_problem(self, model_second_stage[i])

        # 2) Reformulate the first stage optimization problem
        # 2.1) Estimate the boundary of the first stage optimization problem.
        # 2.2) Add additional variables to the first stage optimization problem
        # Using the multiple cuts version
        model_master = deepcopy(model_first_stage)
        model_master["c"] = hstack([model_first_stage["c"], ps])
        model_master["Aeq"] = hstack([model_first_stage["Aeq"], zeros((model_first_stage["Aeq"].shape[0], self.N))])
        model_master["A"] = hstack([model_first_stage["A"], zeros((model_first_stage["A"].shape[0], self.N))])
        model_master["lb"] = hstack([model_first_stage["lb"], -ones((self.N, 1)) * M])
        model_master["ub"] = hstack([model_first_stage["ub"], ones((self.N, 1)) * M])

        # 3) Reformulate the second stage optimization problem
        # 3.1) Formulate the dual problem for each problem under dual problems
        # The dual problem is solved
        model_second_stage = BendersDecomposition.sub_problems_update(self, model_second_stage,
                                                                      sol_first_stage["x"][0:self.nx_first_stage])

        sol_second_stage = [0] * self.N

        A_cuts = zeros((self.N, self.nx_first_stage + self.N))
        b_cuts = zeros((self.N, 1))

        for i in range(self.N):
            # Solve the dual problem
            sol_second_stage[i] = BendersDecomposition.sub_problem_dual(self, model_second_stage[i])

            A_cuts[i, 0:self.nx_first_stage] = transpose(
                multiply(sol_second_stage[i]["x"], model_second_stage[i]["Ts"]))
            b_cuts[i, 0] = - multiply(sol_second_stage[i]["x"], model_second_stage[i]["hs"])

            if sol_second_stage[i]["status"] == "optimal":  # if the primal problem is feasible, add feasible cuts
                A_cuts[i, 0:self.nx_first_stage + i] = -1
            # else add infeasible cuts
        Upper = inf
        Lower = sol_first_stage["objvalue"]
        Gap = [BendersDecomposition.gap_calculaiton(self, Upper, Lower)]
        eps = 10 ^ -3
        iter_max = 1000
        iter = 0

        # 4) Begin the iteration
        while Gap[-1] > eps and iter <= iter_max:
            # Update the master problem
            model_master["A"] = vstack([model_master["A"], A_cuts])
            model_master["b"] = vstack([model_master["b"], b_cuts])
            # solve the master problem
            sol_first_stage = BendersDecomposition.master_problem(self, model_master)
            Lower = sol_first_stage["objvalue"]

            # update the second stage solution
            model_second_stage = BendersDecomposition.sub_problems_update(self, model_second_stage,
                                                                          sol_first_stage["x"][0:self.nx_first_stage])

            objvalue_second_stage = zeros((self.N, 1))

            sol_second_stage = [0] * self.N

            A_cuts = zeros((self.N, self.nx_first_stage + self.N))
            b_cuts = zeros((self.N, 1))

            for i in range(self.N):
                # Solve the dual problem
                sol_second_stage[i] = BendersDecomposition.sub_problem_dual(self, model_second_stage[i])

                A_cuts[i, 0:self.nx_first_stage] = transpose(
                    multiply(sol_second_stage[i]["x"], model_second_stage[i]["Ts"]))
                b_cuts[i, 0] = - multiply(sol_second_stage[i]["x"], model_second_stage[i]["hs"])

                if sol_second_stage[i]["status"] == "optimal":  # if the primal problem is feasible, add feasible cuts
                    A_cuts[i, 0:self.nx_first_stage + i] = -1
                    objvalue_second_stage[i, 0] = sol_second_stage[i]["objvalue"]
                else:
                    objvalue_second_stage[i, 0] = inf

            Upper = multiply(sol_first_stage["x"][0:self.nx_first_stage], model_first_stage["c"]) + multiply(
                objvalue_second_stage, ps)

            Gap.append(BendersDecomposition.gap_calculaiton(self, Upper, Lower))

        x_first_stage = sol_first_stage[0:self.nx_first_stage]

        x_second_stage = zeros((self.N, self.nx_second_stage))

        for i in range(self.N):
            x_second_stage[i, :] = sol_second_stage[i]["x"]
        sol = {"objvalue": Upper,
               "x_first_stage": x_first_stage,
               "x_second_stage": x_second_stage}
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

        return model

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

        return model

    def sub_problem_dual(self, model):
        """
        Solve each slave problems
        :param model:
        :return:
        """
        (x, objvalue, status) = lp(-model["beq"], A=transpose(model["Aeq"]), b=model["c"])

        sol = {"x": x,
               "objvalue": objvalue,
               "status": status}

        return model

    def sub_problems_update(self, model, x):
        """

        :param model: The second stage models
        :param hs: The equality constraints under each stage
        :param Ts: The coupling constraints between the first stage and second stage constraints
        :return: hs-Ts*x
        """
        for i in range(self.N):
            model[i]["beq"] = model[i]["hs"] - multiply(model[i]["Ts"], x)

        return model

    def gap_calculaiton(self, upper, lower):

        if lower != 0:
            gap = (upper - lower) / lower * 100
        else:
            gap = inf

        return gap
