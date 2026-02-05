import json
from Scheduler_ORtools import main as mainOR
from SchedulerZ3_3 import main as mainZ3
from Plotting import compute_rmSE_assigned_vs_target

def UpdateJson(Solver,year):

    with open('Results.json','r') as file:
        Results = json.load(file)

    if Solver == 'Z3':
        process_time = mainZ3(year)
    elif Solver in ['GUROBI','SCIP','SAT']:
        process_time = mainOR(year, Solver)
    else:
        raise ValueError("WRONG SOLVER")

    solver_rmse,manual_rmse = compute_rmSE_assigned_vs_target(Solver,year)
    Results[Solver].update({str(year):[process_time,(round(solver_rmse,2),round(manual_rmse,2))]})

    with open('Results.json','w') as file:
        json.dump(Results,file,indent=4)


if __name__ == "__main__":

    Solvers = ['SAT']# ['GUROBI','SCIP','SAT','Z3']
    Years = [2026]#[2022,2023,2024,2025,2026]

    for solver in Solvers:
        for year in Years:
            UpdateJson(solver,year)

    # Solvers = ['GUROBI']  # ['GUROBI','SCIP','SAT','Z3']
    # Years = [2022,2023,2024,2025,2026]
    #
    # for solver in Solvers:
    #     for year in Years:
    #         UpdateJson(solver, year)
