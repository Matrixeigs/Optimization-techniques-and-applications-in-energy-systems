"""
Two-stage unit commitment for
"""
from pypower import loadcase, ext2int, makeBdc
from scipy.sparse import csr_matrix as sparse
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, concatenate, array, inf, eye, diag, subtract
from numpy import flatnonzero as find
from scipy.sparse.linalg import inv
from scipy.sparse import vstack
from numpy.linalg import pinv
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp


def problem_formulation(case):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    CAP_WIND = 1  # The capacity of wind farm
    BETA = 0.1  # The disturbance range of wind farm
    BETA_HYDRO = 0.05  # The disturbance range of wind farm
    CAPVALUE = 10  # The capacity value
    Price_energy = r_[ones(8), 3 * ones(8), ones(8)]

    from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN

    mpc = ext2int.ext2int(case)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  ## number of buses
    nl = shape(mpc['branch'])[0]  ## number of branches
    ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

    Bbus = makeBdc.makeBdc(baseMVA, bus, branch)
    Distribution_factor = Bbus[1] * inv(Bbus[0])

    # Distribution_factor = array([
    #     [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [-0.005, -0.005, -0.005, -1.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005,
    #      -0.005, ],
    #     [0.47, 0.47, 0.47, 0.47, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
    #     [0.47, 0.47, 0.47, 0.47, -0.03, - 0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
    #     [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, -0.68, -0.68, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32],
    #     [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, -0.68, -0.68, 0.32, 0.32, 0.32, 0.32],
    #     [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, -0.84, 0.16, 0.16, 0.16, 0.16],
    #     [-0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -1.16, -0.16, -1.16, -0.16],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    #     [-0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -0.16, -1.16, -0.16, -0.16],
    #     [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -1.08],
    # ])
    # Distribution_factor = sparse(Distribution_factor)
    # Formulate connection matrix for wind farms
    i = []
    PWMAX = []
    PWMIN = []
    for index in range(ng):
        if gen[index, PMIN] == 0:
            i.append(index)
            PWMAX.append(gen[index, PMAX])
            PWMIN.append(gen[index, PMIN])
    i = array(i)
    nw = i.shape[0]
    Cw = sparse((ones(nw), (gen[i, GEN_BUS], arange(nw))), shape=(nb, nw))
    PWMAX = array(PWMAX).reshape((len(PWMAX), 1))
    PWMIN = array(PWMIN).reshape((len(PWMIN), 1))
    # Formulate the connection matrix for hydro power plants
    i = []
    PHMAX = []
    PHMIN = []
    for index in range(ng):
        if gen[index, PMIN] > 0:
            i.append(index)
            PHMAX.append(gen[index, PMAX])
            PHMIN.append(gen[index, PMIN])
    i = array(i)
    nh = i.shape[0]
    Ch = sparse((ones(nh), (gen[i, GEN_BUS], arange(nh))), shape=(nb, nh))
    PHMAX = array(PHMAX).reshape((len(PHMAX), 1))
    PHMIN = array(PHMIN).reshape((len(PHMIN), 1))

    # Formulate the external power systems
    i = []
    PEXMAX = []
    PEXMIN = []
    for index in range(ng):
        if gen[index, PMIN] < 0:
            i.append(index)
            PEXMAX.append(gen[index, PMAX])
            PEXMIN.append(gen[index, PMIN])
    i = array(i)
    nex = i.shape[0]
    Cex = sparse((ones(nex), (gen[i, GEN_BUS], arange(nex))), shape=(nb, nex))
    PEXMAX = array(PEXMAX).reshape((len(PEXMAX), 1))
    PEXMIN = array(PEXMIN).reshape((len(PEXMIN), 1))
    PLMAX = branch[:, RATE_A].reshape((nl, 1))  # The power flow limitation

    T = 24
    ## Profiles
    # Wind profile
    WIND_PROFILE = array(
        [591.35, 714.50, 1074.49, 505.06, 692.78, 881.88, 858.48, 609.11, 559.95, 426.86, 394.54, 164.47, 27.15, 4.47,
         54.08, 109.90, 111.50, 130.44, 111.59, 162.38, 188.16, 216.98, 102.94, 229.53]).reshape((T, 1))
    WIND_PROFILE = WIND_PROFILE / WIND_PROFILE.max()
    WIND_PROFILE_FORECAST = zeros((T * nw, 1))
    Delta_wind = zeros((T * nw, 1))
    for i in range(T):
        WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX
        Delta_wind[i * nw:(i + 1) * nw, :] = WIND_PROFILE[i] * PWMAX * BETA

    # Load profile
    LOAD_PROFILE = array([0.632596195634005, 0.598783973523217, 0.580981513054525, 0.574328051348912, 0.584214221241601,
                          0.631074282084712, 0.708620833751212, 0.797665730618795, 0.877125330124026, 0.926981579915087,
                          0.947428654208872, 0.921588439808779, 0.884707317888543, 0.877717046100358, 0.880387289807107,
                          0.892056129442049, 0.909233443653261, 0.926748403704075, 0.968646575067696, 0.999358974358974,
                          0.979169591816267, 0.913517534182463, 0.806453715775750, 0.699930632166617]).reshape((T, 1))
    # Hydro information
    HYDRO_INJECT = array([6, 2, 4, 3]).reshape((nh, 1))
    HYDRO_INJECT_FORECAST = zeros((T * nh, 1))
    Delta_hydro = zeros((T * nh, 1))
    for i in range(T):
        HYDRO_INJECT_FORECAST[i * nh:(i + 1) * nh, :] = HYDRO_INJECT
        Delta_hydro[i * nh:(i + 1) * nh, :] = HYDRO_INJECT * BETA_HYDRO

    MIN_DOWN = ones((nh, 1))
    MIN_UP = ones((nh, 1))

    QMIN = array([1.5, 1, 1, 1]).reshape((nh, 1))
    QMAX = array([20, 10, 10, 10]).reshape((nh, 1))
    VMIN = array([70, 50, 70, 40]).reshape((nh, 1))
    VMAX = array([160, 140, 150, 130]).reshape((nh, 1))
    V0 = array([110, 90, 100, 80]).reshape((nh, 1))
    M = diag(array([8.8649, 6.4444, 6.778, 7.3333]))
    C_TEMP = array([30, 2, 9, 4]).reshape((4, 1))
    Q_TEMP = array([1.5, 1, 1, 1]).reshape((4, 1))
    # Define the first stage decision variables
    ON = 0
    OFF = 1
    IHG = 2
    PHG = 3
    RUHG = 4
    RDHG = 5
    QHG = 6
    QUHG = 7
    QDHG = 8
    V = 9
    S = 10
    PWC = 11
    PLC = 12
    PEX = 13
    CEX = 14
    nx = PWC * nh * T + nw * T + nb * T + nex * T + 1
    lb = zeros((nx, 1))
    ub = zeros((nx, 1))
    c = zeros((nx, 1))
    vtypes = ["c"] * nx
    for i in range(T):
        for j in range(nh):
            # lower boundary information
            lb[ON * nh * T + i * nh + j] = 0
            lb[OFF * nh * T + i * nh + j] = 0
            lb[IHG * nh * T + i * nh + j] = 0
            lb[PHG * nh * T + i * nh + j] = 0
            lb[RUHG * nh * T + i * nh + j] = 0
            lb[RDHG * nh * T + i * nh + j] = 0
            lb[QHG * nh * T + i * nh + j] = 0
            lb[QUHG * nh * T + i * nh + j] = 0
            lb[QDHG * nh * T + i * nh + j] = 0
            lb[V * nh * T + i * nh + j] = VMIN[j]
            lb[S * nh * T + i * nh + j] = 0
            # upper boundary information
            ub[ON * nh * T + i * nh + j] = 1
            ub[OFF * nh * T + i * nh + j] = 1
            ub[IHG * nh * T + i * nh + j] = 1
            ub[PHG * nh * T + i * nh + j] = PHMAX[j]
            ub[RUHG * nh * T + i * nh + j] = PHMAX[j]
            ub[RDHG * nh * T + i * nh + j] = PHMAX[j]
            ub[QHG * nh * T + i * nh + j] = QMAX[j]
            ub[QUHG * nh * T + i * nh + j] = QMAX[j]
            ub[QDHG * nh * T + i * nh + j] = QMAX[j]
            ub[V * nh * T + i * nh + j] = VMAX[j]
            ub[S * nh * T + i * nh + j] = 10 ** 8
            # objective value
            c[S * nh * T + i * nh + j] = 1
            # variables types
            vtypes[ON * nh * T + i * nh + j] = "D"
            vtypes[OFF * nh * T + i * nh + j] = "D"
            vtypes[IHG * nh * T + i * nh + j] = "D"
            if i == T:
                lb[V * nh * T + i * nh + j] = V0[j]
                ub[V * nh * T + i * nh + j] = V0[j]

        for j in range(nw):
            # lower boundary information
            lb[PWC * nh * T + i * nw + j] = 0
            # upper boundary information
            ub[PWC * nh * T + i * nw + j] = WIND_PROFILE_FORECAST[i * nw + j]
            # objective value
            c[PWC * nh * T + i * nw + j] = 1
        for j in range(nb):
            # lower boundary information
            lb[PWC * nh * T + nw * T + i * nb + j] = 0
            # upper boundary information
            ub[PWC * nh * T + nw * T + i * nb + j] = bus[j, PD] * LOAD_PROFILE[i]
            # objective value
            c[PWC * nh * T + nw * T + i * nb + j] = 10 ** 8
        for j in range(nex):
            # lower boundary information
            lb[PWC * nh * T + nw * T + nb * T + i * nex + j] = PEXMIN[j]
            # upper boundary information
            ub[PWC * nh * T + nw * T + nb * T + i * nex + j] = PEXMAX[j]
            # objective value
            c[PWC * nh * T + nw * T + nb * T + i * nex + j] = -Price_energy[i]
    # lower boundary information
    lb[PWC * nh * T + nw * T + nb * T + nex * T] = PEXMIN[0]
    # upper boundary information
    ub[PWC * nh * T + nw * T + nb * T + nex * T] = PEXMAX[0]
    # objective value
    c[PWC * nh * T + nw * T + nb * T + nex * T] = -CAPVALUE

    # 2) Constraint set
    # 2.1) Power balance equation
    Aeq = zeros((T, nx))
    beq = zeros((T, 1))
    for i in range(T):
        # For the hydro units
        for j in range(nh):
            Aeq[i, PHG * nh * T + i * nh + j] = 1
        # For the wind farms
        for j in range(nw):
            Aeq[i, PWC * nh * T + i * nw + j] = -1
        # For the loads
        for j in range(nb):
            Aeq[i, PWC * nh * T + nw * T + i * nb + j] = 1
        # For the power exchange
        for j in range(nex):
            Aeq[i, PWC * nh * T + nw * T + nb * T + i * nex + j] = -1

        beq[i] = sum(bus[:, PD]) * LOAD_PROFILE[i] - sum(WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])

    # 2.2) Status transformation of each unit
    # Aeq_temp = zeros((T * nh, nx))
    # beq_temp = zeros((T * nh, 1))
    # for i in range(T):
    #     for j in range(nh):
    #         Aeq_temp[i * nh + j, ON * nh * T + i * nh + j] = 1
    #         Aeq_temp[i * nh + j, OFF * nh * T + i * nh + j] = -1
    #         Aeq_temp[i * nh + j, IHG * nh * T + i * nh + j] = 1
    #         if i != 0:
    #             Aeq_temp[i * nh + j, IHG * nh * T + (i - 1) * nh + j] = -1
    #         else:
    #             beq_temp[i * T + j] = 0

    # Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    # beq = concatenate((beq, beq_temp), axis=0)
    # 2.3) water status change
    Aeq_temp = zeros((T * nh, nx))
    beq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aeq_temp[i * nh + j, V * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, S * nh * T + i * nh + j] = -1
            Aeq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            if i != 0:
                Aeq_temp[i * nh + j, V * nh * T + (i - 1) * nh + j] = -1
            else:
                beq_temp[i * T + j] = V0[j]

    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq = concatenate((beq, beq_temp), axis=0)

    # 2.4) Power water transfering
    Aeq_temp = zeros((T * nh, nx))
    beq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aeq_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            Aeq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -M[j, j]
            Aeq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -abs(-C_TEMP[j] + M[j, j] * Q_TEMP[j])
    Aeq = concatenate((Aeq, Aeq_temp), axis=0)
    beq = concatenate((beq, beq_temp), axis=0)

    # 2.5) Power range limitation
    Aineq = zeros((T * nh, nx))
    bineq = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq[i * nh + j, IHG * nh * T + i * nh + j] = PHMIN[j]
            Aineq[i * nh + j, PHG * nh * T + i * nh + j] = -1
            Aineq[i * nh + j, RDHG * nh * T + i * nh + j] = 1

    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -PHMAX[j]
            Aineq_temp[i * nh + j, PHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, RUHG * nh * T + i * nh + j] = 1

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)
    # (xx, obj, success) = lp(c, Aeq=Aeq, beq=beq, A=Aineq, b=bineq, xmin=lb, xmax=ub, vtypes=vtypes)
    # xx = array(xx).reshape((len(xx), 1))
    # 2.6) Water reserve constraints
    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        Aineq_temp[i * nh:(i + 1) * nh, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = eye(nh)
        Aineq_temp[i * nh:(i + 1) * nh, RUHG * nh * T + i * nh:RUHG * nh * T + (i + 1) * nh] = eye(nh)
        Aineq_temp[i * nh:(i + 1) * nh, IHG * nh * T + i * nh:IHG * nh * T + (i + 1) * nh] = -diag(
            C_TEMP - M.dot(Q_TEMP))
        Aineq_temp[i * nh:(i + 1) * nh, QHG * nh * T + i * nh:QHG * nh * T + (i + 1) * nh] = -M
        Aineq_temp[i * nh:(i + 1) * nh, QUHG * nh * T + i * nh:QUHG * nh * T + (i + 1) * nh] = -M

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        Aineq_temp[i * nh:(i + 1) * nh, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = -eye(nh)
        Aineq_temp[i * nh:(i + 1) * nh, RDHG * nh * T + i * nh:RDHG * nh * T + (i + 1) * nh] = eye(nh)
        Aineq_temp[i * nh:(i + 1) * nh, IHG * nh * T + i * nh:IHG * nh * T + (i + 1) * nh] = diag(
            C_TEMP - M.dot(Q_TEMP))
        Aineq_temp[i * nh:(i + 1) * nh, QHG * nh * T + i * nh:QHG * nh * T + (i + 1) * nh] = M
        Aineq_temp[i * nh:(i + 1) * nh, QDHG * nh * T + i * nh:QDHG * nh * T + (i + 1) * nh] = -M

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.7) water flow constraints
    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = QMIN[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QDHG * nh * T + i * nh + j] = 1
    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, IHG * nh * T + i * nh + j] = -QMAX[j]
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QUHG * nh * T + i * nh + j] = 1

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.8) Water reserve limitation
    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, V * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QDHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, S * nh * T + i * nh + j] = -1
            bineq_temp[i * nh + j] = VMAX[j] - HYDRO_INJECT_FORECAST[i * nh + j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # (xx, obj, success) = lp(c, Aeq=Aeq, beq=beq, A=Aineq, b=bineq, xmin=lb, xmax=ub, vtypes=vtypes)
    # xx = array(xx).reshape((len(xx), 1))

    Aineq_temp = zeros((T * nh, nx))
    bineq_temp = zeros((T * nh, 1))
    for i in range(T):
        for j in range(nh):
            Aineq_temp[i * nh + j, V * nh * T + i * nh + j] = -1
            Aineq_temp[i * nh + j, QHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, QUHG * nh * T + i * nh + j] = 1
            Aineq_temp[i * nh + j, S * nh * T + i * nh + j] = 1
            bineq_temp[i * nh + j] = -VMIN[j] + HYDRO_INJECT_FORECAST[i * nh + j]

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    # 2.9) Line flow limitation
    # Aineq_temp = zeros((T * nl, nx))
    # bineq_temp = zeros((T * nl, 1))
    # for i in range(T):
    #     Aineq_temp[i * nl:(i + 1) * nl, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = -(
    #             Distribution_factor * Ch).todense()
    #     Aineq_temp[i * nl:(i + 1) * nl, PWC * nh * T + i * nw:PWC * nh * T + (i + 1) * nw] = (
    #             Distribution_factor * Cw).todense()
    #     Aineq_temp[i * nl:(i + 1) * nl,
    #     PWC * nh * T + nw * T + i * nb:PWC * nh * T + nw * T + (i + 1) * nb] = -Distribution_factor.todense()
    #
    #     Aineq_temp[i * nl:(i + 1) * nl,
    #     PWC * nh * T + nw * T + nb * T + i * nex:PWC * nh * T + nw * T + nb * T + (i + 1) * nex] = (
    #             Distribution_factor * Cex).todense()
    #
    #     bineq_temp[i * nl:(i + 1) * nl, :] = PLMAX - Distribution_factor * (
    #             (bus[:, PD] * LOAD_PROFILE[i]).reshape(nb, 1) - Cw * WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])
    # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    # bineq = concatenate((bineq, bineq_temp), axis=0)
    #
    # Aineq_temp = zeros((T * nl, nx))
    # bineq_temp = zeros((T * nl, 1))
    # for i in range(T):
    #     Aineq_temp[i * nl:(i + 1) * nl, PHG * nh * T + i * nh:PHG * nh * T + (i + 1) * nh] = (
    #             Distribution_factor * Ch).todense()
    #
    #     Aineq_temp[i * nl:(i + 1) * nl, PWC * nh * T + i * nw:PWC * nh * T + (i + 1) * nw] = -(
    #             Distribution_factor * Cw).todense()
    #
    #     Aineq_temp[i * nl:(i + 1) * nl,
    #     PWC * nh * T + nw * T + i * nb:PWC * nh * T + nw * T + (i + 1) * nb] = Distribution_factor.todense()
    #
    #     Aineq_temp[i * nl:(i + 1) * nl,
    #     PWC * nh * T + nw * T + nb * T + i * nex:PWC * nh * T + nw * T + nb * T + (i + 1) * nex] = -(
    #             Distribution_factor * Cex).todense()
    #
    #     bineq_temp[i * nl:(i + 1) * nl, :] = PLMAX + Distribution_factor * (
    #             (bus[:, PD] * LOAD_PROFILE[i]).reshape(nb, 1) - Cw * WIND_PROFILE_FORECAST[i * nw:(i + 1) * nw])
    #
    # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    # bineq = concatenate((bineq, bineq_temp), axis=0)
    # For the capacity
    Aineq_temp = zeros((T, nx))
    bineq_temp = zeros((T, 1))
    for i in range(T):
        Aineq_temp[i, PWC * nh * T + nw * T + nb * T + nex * T] = 1
        Aineq_temp[i, PWC * nh * T + nw * T + nb * T + i * nex] = -1

    Aineq = concatenate((Aineq, Aineq_temp), axis=0)
    bineq = concatenate((bineq, bineq_temp), axis=0)

    (xx, obj, success) = lp(c, Aeq=Aeq, beq=beq, A=Aineq, b=bineq, xmin=lb, xmax=ub, vtypes=vtypes)
    xx = array(xx).reshape((len(xx), 1))
    # 2.3)

    return model


if __name__ == "__main__":
    from unit_commitment.test_cases.case14 import case14

    case = loadcase.loadcase(case14())
    model = problem_formulation(case)

    print(case)
