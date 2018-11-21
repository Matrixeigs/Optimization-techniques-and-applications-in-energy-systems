
import cplex as cpx


def querry_cpx_error(cpx_model, constraint_rows):
    '''
    Querry the variable names involved in specific constraints
    :param cpx_model: cplex model
    :param constraint_rows: the indices of constraints
    :return:
    '''

    # get indices and values of variables involved in infeasible constraints
    ind_var, val_var = cpx_model.linear_constraints.get_rows(constraint_rows).unpack()
    # get variable names
    name_var = cpx_model.variables.get_names(ind_var)

    return name_var


if __name__ == 'main':
    querry_cpx_error()