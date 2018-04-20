"""
Optimal power flow for hybrid AC/DC micro-grids
Two versions of optimal power flow models are proposed.
1) Single period
2) Multiple periods
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
"""









if __name__ == '__main__':
    # A test hybrid AC DC network is connected via BIC networks
    caseAC = case33.case33()
    caseDC = case118.case118()
    converters = case_converters.con()
    sol = main(Case_AC=caseAC, Case_DC=caseDC, Converters=converters)