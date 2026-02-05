import os
import pandas as pd
from Plotting import plot_assigned_vs_target,plot_course_histogram,compute_rmSE_assigned_vs_target

from ortools.linear_solver import pywraplp
from time import time as tm

# -----------------------------
# PRE-PROCESSING
# -----------------------------
def preprocess_data(year):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, f'FinalRuns/{year}/{year}.xlsx')  # change name of Excel document for input
    sheets = pd.read_excel(excel_path, sheet_name=None)

    df_TAs = sheets['TAs']
    df_TA_req = sheets['TA_req']
    df_times = sheets['times']
    disqualified_list = sheets['disqualified']
    taken_courses = sheets['taken_courses']
    preferences = sheets['preferences']
    weights = sheets['weights']
    cons = sheets['cons']

    return (
        df_TAs.fillna(0),
        df_TA_req.fillna(0),
        df_times.fillna(0),
        disqualified_list.fillna(0),
        taken_courses.fillna(0),
        preferences.fillna(0),
        weights.fillna(0),
        cons.fillna(0),
    )


def find_len(df_TAs, df_TA_req):
    TA_len = len(df_TAs['Abbr.'])
    course_len = len(df_TA_req.columns[1:])
    task_len = len(df_TA_req['Course code'])
    return TA_len, course_len, task_len


def load_cons(cons):
    ALLOWED_DEV_H = int(cons['Value'].iloc[1])
    MAX_COURSES_H = int(cons['Value'].iloc[2])
    MAX_TAS_H = int(cons['Value'].iloc[3])
    MAX_SWITCH_H = int(cons['Value'].iloc[4])
    ALLOWED_DEV_S = int(cons['Value'].iloc[5])
    MAX_COURSES_S = int(cons['Value'].iloc[6])
    MAX_TAS_S = int(cons['Value'].iloc[7])
    MAX_SWITCH_S = int(cons['Value'].iloc[8])
    MIN_CHUNK_HOURS = float(cons['Value'].iloc[11])
    return ALLOWED_DEV_H, ALLOWED_DEV_S, MAX_COURSES_H, MAX_COURSES_S, MAX_TAS_H, MAX_TAS_S, MAX_SWITCH_H, MAX_SWITCH_S, MIN_CHUNK_HOURS


def TA_happiness(TA_len, course_len, df_c, taken_courses, preferences, df_TAs, df_TA_req):
    TA_index = {name: i for i, name in enumerate(taken_courses['Abbr.'])}
    course_index = {course: j for j, course in enumerate(taken_courses.columns[1:])}
    results = []
    for TA in range(TA_len):
        TA_name = df_TAs['Abbr.'].iloc[TA]
        TA_existed = TA_name in TA_index
        last_i = TA_index.get(TA_name, None)

        switch_c = 0
        prefered_c = 0
        not_prefered_c = 0
        courses_taught = 0

        for c in range(course_len):
            c_name = df_TA_req.columns[c + 1]
            course_existed = c_name in course_index
            if df_c.iloc[TA, c] == 1:
                courses_taught += 1

                # Preferences
                if preferences.iloc[TA, c] == 1:
                    prefered_c += 1
                elif preferences.iloc[TA, c] == -1:
                    not_prefered_c += 1

                # Switching
                if TA_existed and course_existed:
                    last_j = course_index[c_name] + 1
                    if taken_courses.iloc[last_i, last_j] == 0:
                        switch_c += 1

        # Compute happiness
        if courses_taught == 0:
            happiness = 0
        else:
            happiness = (switch_c * (-100) + prefered_c * 5 + not_prefered_c * (-10)) / courses_taught
        results.append([TA_name, happiness, switch_c, courses_taught])
    return pd.DataFrame(results, columns=["TA", "Happiness", "Switched Courses", "Taught courses"])


# -----------------------------
# SOLVER
# -----------------------------
def create_model(
                TA_len, course_len, task_len,
                df_TAs, df_TA_req, disqualified_list,
                taken_courses, weights,
                MAX_SWITCH_H, MAX_SWITCH_S,
                df_times,
                ALLOWED_DEV_H, ALLOWED_DEV_S,
                MAX_COURSES_H, MAX_COURSES_S, MAX_TAS_H,
                MAX_TAS_S, MIN_CHUNK_HOURS, preferences,
                timeout_ms=None,
                solver='SAT'
                 ):

    model = pywraplp.Solver.CreateSolver(solver)
    model.SetTimeLimit(timeout_ms)

    bound = 10000 # TODO find right value for this parameter AND ALL THE OTHERS

    # -----------------------------
    # VARIABLES
    # -----------------------------

    # INTEGER hours TA taught task t of course c
    t2c = [[[model.IntVar(0,bound,f"{TA}c{c}t{t}") for t in range(task_len)]for c in range(course_len)]for TA in range(TA_len)]

    # BINARY TA taught task t of course c
    bin_t2c = [[[model.IntVar(0, 1, f"bin_{TA}c{c}t{t}") for t in range(task_len)] for c in range(course_len)] for TA in
           range(TA_len)]

    # INTEGER TA total teaching time
    TA_total_time = [model.IntVar(0,bound,f'TA_tot_time{TA}') for TA in range(TA_len)]

    # BINARY if TA taught in c
    c_taught = [[model.IntVar(0,1,f'{TA}_to_{c}')for c in range(course_len)] for TA in range(TA_len)]

    # BINARY if TA teaches too many courses
    exceeding_c = [model.IntVar(0,1,f'{TA}_exceeding')for TA in range(TA_len)]

    # INTEGER number of courses switched by a TA
    switched_c = [model.IntVar(0,bound,f'{TA}_switch')for TA in range(TA_len)]

    # BINARY if a TA has switched to course c (did not teach it in year one but in following years)
    switched_TA_c = [[model.IntVar(0, 1, f'switched{TA}_{c}') for c in range(course_len)] for TA in range(TA_len)]

    # BINARY if a TA switched more courses than the MAX allowed value
    over_switched = [model.IntVar(0, 1, f'{TA}_over_switched') for TA in range(TA_len)]

    # INTEGER number of TAs assigned to task t of course c
    assigned_TAs = [[model.IntVar(0,bound,f'{c}{t}_ass') for c in range(course_len)] for t in range(task_len)]

    # BINARY too many TAs for course
    exceeding_TAs = [[model.IntVar(0,1,f'{c}{t}_exceeding_ass') for c in range(course_len)] for t in range(task_len)]

    #INTEGER deviation between intended teaching and actual teaching for a TA
    TA_dev = [model.IntVar(0, bound, f'{TA}_dev') for TA in range(TA_len)]

    # BINARY if a TA exceed the max allowed deviation (year 1-4)
    TA_exceeded_dev = [model.IntVar(0, 1, f'{TA}_exceeded_dev') for TA in range(TA_len)]

    # BINARY if a TA exceed the max allowed deviation (year 1-4)
    TA_exceeded_dev_final = [model.IntVar(0, 1, f'{TA}_exceeded_dev_final') for TA in range(TA_len)]

    # BINARY if a TA teaches a course they don't want
    undesired_c = [[model.IntVar(0, 1, f'undesired_{TA}_{c}') for c in range(course_len)] for TA in range(TA_len)]

    # BINARY if a TA does NOT teach a course they want
    desired_c = [[model.IntVar(0, 1, f'undesired_{TA}_{c}') for c in range(course_len)] for TA in range(TA_len)]

    # -----------------------------
    # CONSTRAINTS
    # -----------------------------

    # count TAs teaching in task t of course c
    for c in range(course_len):
        for t in range(task_len):
            model.Add(assigned_TAs[t][c] == model.Sum([bin_t2c[TA][c][t] for TA in range(TA_len)]))

            for TA in range(TA_len):
                model.Add(bin_t2c[TA][c][t] * 15000 >= t2c[TA][c][t]) # todo task time

    TA_index = {name: i for i, name in enumerate(df_TAs['Abbr.'])}
    C_index = {course: i for i, course in enumerate(df_TA_req.columns[1:])}

    for _, row in disqualified_list.iterrows():
        TA_name = row['TA']
        course_name = row['Course']
        TA = TA_index[TA_name]
        c = C_index[course_name]

        for t in range(task_len):
            # forbid specific TA to teach specific courses (i.e. parental leave)
            model.Add(t2c[TA][c][t] == 0)

    TA_index = {name: i for i, name in enumerate(taken_courses['Abbr.'])}
    course_index = {course: j for j, course in enumerate(taken_courses.columns[1:])}

    for TA in range(TA_len):
        TA_name = df_TAs['Abbr.'].iloc[TA]

        # Skip new TA
        if TA_name not in TA_index:
            model.Add(switched_c[TA] == 0)
            continue
        last_i = TA_index[TA_name]

        for c in range(course_len):
            c_name = df_TA_req.columns[c + 1]

            # Skip new courses
            if c_name not in course_index:
                continue
            last_j = course_index[c_name] + 1

            # Count switched courses
            model.Add(switched_TA_c[TA][c] >= c_taught[TA][c] - taken_courses.iloc[last_i, last_j])

        # Check switched courses are not too many
        model.Add(switched_c[TA] == model.Sum([switched_TA_c[TA][c] for c in range(course_len)]))
        model.Add(switched_c[TA] <= MAX_SWITCH_H)
        model.Add(over_switched[TA] * course_len >= switched_c[TA] - MAX_SWITCH_S)

    # smaller chunk of task that can be assigned to a TA
    target_min_chunk = int(MIN_CHUNK_HOURS)

    for c in range(course_len):
        for t in range(task_len):
            # skip empty task
            if float(df_times.iloc[t, c + 1]) == 0:
                for TA in range(TA_len):
                    # Set non-existing tasks to zero
                    model.Add(t2c[TA][c][t] == 0)
                continue

            # Check each TA teach within task limits
            task_time = int(float(df_times.iloc[t, c + 1]))

            # Get required number of TAs for this specific task
            req_tas = int(df_TA_req.iloc[t, c + 1])

            # Calculate the theoretical maximum chunk size
            if req_tas > 0:
                theoretical_max_per_person = task_time // req_tas
            else:
                theoretical_max_per_person = task_time

            # The actual minimum is the smaller of:
            min_assign = min(target_min_chunk, theoretical_max_per_person, task_time)

            for TA in range(TA_len):
                model.Add( t2c[TA][c][t] >= min_assign - (1-bin_t2c[TA][c][t]) * 15000) #todo find value
                model.Add( t2c[TA][c][t] <= task_time + (1-bin_t2c[TA][c][t]) * 15000)

            # Check tasks are fully covered
            model.Add(model.Sum([t2c[TA][c][t] for TA in range(TA_len)]) == task_time)


    # Set deviation to zero for TAs in their final year #
    for TA in range(TA_len):
        model.Add( TA_dev[TA] >= TA_total_time[TA] - int(df_TAs['Target'].iloc[TA] ))
        model.Add( TA_dev[TA] >= - TA_total_time[TA] + int(df_TAs['Target'].iloc[TA] ))

        if df_TAs['Year'].iloc[TA] >= 5:
            model.Add(TA_exceeded_dev_final[TA] * 15000 >= TA_total_time[TA] - int(df_TAs['Target'].iloc[TA] ) )
            model.Add(TA_exceeded_dev_final[TA] * 15000 >= - TA_total_time[TA] + int(df_TAs['Target'].iloc[TA] ))

    # Check deviation not being too big
    for TA in range(TA_len):
        model.Add(TA_dev[TA] <= ALLOWED_DEV_H )
        model.Add(TA_exceeded_dev[TA] * 15000 >= TA_dev[TA] - ALLOWED_DEV_S )
        model.Add(TA_exceeded_dev[TA] * 15000 >= - TA_dev[TA] + ALLOWED_DEV_S )


    # Check what courses TAs teach
    for TA in range(TA_len):
        for c in range(course_len):
            model.Add( c_taught[TA][c] * 15000 >= model.Sum([ t2c[TA][c][t] for t in range(task_len)]) )

    # Check TAs not teaching too many courses
    for TA in range(TA_len):
        model.Add(exceeding_c[TA] * 15000 >= model.Sum([c_taught[TA][c] for c in range(course_len)]) - MAX_COURSES_S)
        model.Add(model.Sum([c_taught[TA][c] for c in range(course_len)]) <= MAX_COURSES_H)

    # Check courses not having too many TAs
    for c in range(course_len):
        model.Add(model.Sum([c_taught[TA][c] for TA in range(TA_len)]) <= MAX_TAS_H)

        # Check course admin is only one person
        model.Add(assigned_TAs[0][c] <= 1)

        for t in range(task_len):
            # Check assigned TAs are enough
            model.Add(assigned_TAs[t][c] >= int(df_TA_req.iloc[t, c + 1]))
            model.Add(exceeding_TAs[t][c] * TA_len >= assigned_TAs[t][c] - (int(df_TA_req.iloc[t, c + 1]) + MAX_TAS_S))

    # Teach preferred courses
    for TA in range(TA_len):
        for c in range(course_len):
            if preferences.iloc[TA, c] == 1:
                model.Add(desired_c[TA][c] == c_taught[TA][c], weight=int(weights['Weight'].iloc[5]))
            elif preferences.iloc[TA, c] == -1:
                model.Add(undesired_c[TA][c] == - c_taught[TA][c], weight=int(weights['Weight'].iloc[6]))
            elif preferences.iloc[TA,c] == 0:
                model.Add(undesired_c[TA][c] == 0)
                model.Add(  desired_c[TA][c] == 0)
            # else:
            #     raise(ValueError('Wrong preference value'))


    # Objective function
    model.Minimize(
        int(weights['Weight'].iloc[0]) * model.Sum([over_switched[TA] for TA in range(TA_len) ])
        +
        int(weights['Weight'].iloc[1]) * model.Sum([TA_exceeded_dev[TA] for TA in range(TA_len)])
        +
        int(weights['Weight'].iloc[2]) * model.Sum([TA_exceeded_dev_final[TA] for TA in range(TA_len)])
        +
        int(weights['Weight'].iloc[3]) * model.Sum([exceeding_c[TA] for TA in range(TA_len)])
        +
        int(weights['Weight'].iloc[4]) * model.Sum([exceeding_TAs[t][c]
                                                    for c in range(course_len) for t in range(task_len)])
        -
        int(weights['Weight'].iloc[5]) * model.Sum([desired_c[TA][c]
                                                    for TA in range(TA_len) for c in range(course_len)])
        +
        int(weights['Weight'].iloc[6]) * model.Sum([undesired_c[TA][c]
                                                    for TA in range(TA_len) for c in range(course_len)])
    )

    # -----------------------------
    # SOLUTION
    # -----------------------------

    # Solve
    print(f"Solving with {model.SolverVersion()}")
    status = model.Solve()

    SolvingTime = pywraplp.Solver.WallTime(self=model)

    df_ass = []
    df_c = []
    TA_happy = []
    df_ta_task = []

    if status != pywraplp.Solver.INFEASIBLE:

        if SolvingTime >= timeout_ms:
            print("Timeout reached - best solution found:")
        elif SolvingTime < timeout_ms:
            print("Optimal solution found")

        print(f"Objective value = {model.Objective().Value():0.1f}")

        # print('over switched')
        # for TA in range(TA_len):
        #     print(over_switched[TA].solution_value())
        # print('exceeded dev')
        # for TA in range(TA_len):
        #     print(TA_exceeded_dev[TA].solution_value())
        # print('exceede dev final')
        # for TA in range(TA_len):
        #     print(TA_exceeded_dev_final[TA].solution_value())
        # print('exceeding courses')
        # for TA in range(TA_len):
        #     print(exceeding_c[TA].solution_value())
        # print('too many TAs')
        # for t in range(task_len):
        #     for c in range(course_len):
        #         print(exceeding_TAs[t][c].solution_value())
        # print('desired c')
        # for TA in range(TA_len):
        #     for c in range(course_len):
        #         print(desired_c[TA][c].solution_value())
        # print('undesired course')
        # for TA in range(TA_len):
        #     for c in range(course_len):
        #         print(undesired_c[TA][c].solution_value())

        # Find TA time results
        df_TAs['Assigned'] = [x.solution_value() for x in TA_total_time]
        df_TAs['Deviation'] = [x.solution_value() - df_TAs['Target'].iloc[TA] for TA, x
                               in enumerate(TA_total_time)]

        # Find TAs per task
        TA_ass = [[assigned_TAs[t][c].solution_value() for c in range(course_len)] for t in range(task_len)]
        df_ass = pd.DataFrame(TA_ass, index=df_TA_req['Course code'], columns=df_TA_req.columns[1:])

        # Find what courses each TA is assigned
        c2TA = [[c_taught[TA][c].solution_value() for c in range(course_len)] for TA in range(TA_len)]
        df_c = pd.DataFrame(c2TA, index=df_TAs['Abbr.'], columns=df_TA_req.columns[1:])

        # Calculate happiness
        TA_happy = TA_happiness(TA_len, course_len, df_c, taken_courses, preferences, df_TAs, df_TA_req)

        # Find what task each teacher is assigned
        matrix_data = []
        for t in range(task_len):
            row = []
            for c in range(course_len):
                ta_strings = []
                for TA in range(TA_len):
                    val = t2c[TA][c][t].solution_value()
                    if val > 0:  # Only include TAs assigned to this task
                        ta_strings.append(f"{df_TAs['Abbr.'].iloc[TA]}: {val}")
                row.append(", ".join(ta_strings))
            matrix_data.append(row)
        df_ta_task = pd.DataFrame(matrix_data, index=[f"Task {t}" for t in range(task_len)], columns=df_TA_req.columns[1:])

    else:
        print("Problem is infeasible")

    return df_TAs, df_ass, df_c, TA_happy, df_ta_task, status

def save_result(df_TAs, df_ass, df_c, happ, df_TA_req, df_ta_task,year,solver):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, f'FinalRuns/{year}/{year+1}_{solver}.xlsx')  # Change name of Excel document for output

    df_ass.index = df_TA_req.iloc[:, 0]
    df_ta_task.index = df_TA_req.iloc[:, 0]
    df_c.index = happ.iloc[:, 0]

    with pd.ExcelWriter(excel_path) as writer:
        df_TAs.to_excel(writer, sheet_name='TAs', index=False)
        df_ass.to_excel(writer, sheet_name='assigned_TAs', index=True)
        df_c.to_excel(writer, sheet_name='taken_courses', index=True)
        happ.to_excel(writer, sheet_name='happiness', index=False)
        df_ta_task.to_excel(writer, sheet_name='df_ta_task', index=True)


def main(year,solver):

    # Pre-processing
    df_TAs, df_TA_req, df_times, disqualified_list, taken_courses, preferences, weights, cons = preprocess_data(year)
    TA_len, course_len, task_len = find_len(df_TAs, df_TA_req)
    df_TAs['Target'] = df_TAs['Part of full time'] * 350 + df_TAs['From prev. year']

    ALLOWED_DEV_H, ALLOWED_DEV_S, MAX_COURSES_H, MAX_COURSES_S, MAX_TAS_H, MAX_TAS_S, MAX_SWITCH_H, MAX_SWITCH_S, MIN_CHUNK_HOURS = load_cons(cons)

    TIME_LIMIT_SECONDS = int(cons['Value'].iloc[0])

    start_time = tm()

    # create model and solve it. then extract the solution
    df_TAs, df_ass, df_c, TA_happy, df_ta_task, status = create_model(
                                                                TA_len, course_len, task_len,
                                                                df_TAs, df_TA_req, disqualified_list,
                                                                taken_courses, weights,
                                                                MAX_SWITCH_H, MAX_SWITCH_S,
                                                                df_times,
                                                                ALLOWED_DEV_H, ALLOWED_DEV_S,
                                                                MAX_COURSES_H, MAX_COURSES_S, MAX_TAS_H,
                                                                MAX_TAS_S, MIN_CHUNK_HOURS, preferences,
                                                                timeout_ms=TIME_LIMIT_SECONDS * 1000,
                                                                solver=solver
                                                            )

    end_time = tm() - start_time

    if status != pywraplp.Solver.INFEASIBLE:
        # Create new excel file with the solution
        save_result(df_TAs, df_ass, df_c, TA_happy, df_TA_req, df_ta_task,year,solver)

    return round(end_time,2)


# Run script
if __name__ == "__main__":
    year = 2022
    Solvers = ['GUROBI','SAT']

    for solver in Solvers:
        process_time = main(year,solver) # SCIP, SAT, GLPK, GUROBI
        print(f'Solving Time : {process_time}')

        # plot_course_histogram(year, sep=",", save=True)
        # plot_assigned_vs_target(year, sep_solver=";", sep_manual=",")
        compute_rmSE_assigned_vs_target(solver,year, sep_solver=";", sep_manual=",")