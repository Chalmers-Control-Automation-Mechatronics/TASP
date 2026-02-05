import os
import pandas as pd
from z3 import *
from time import time as tm
import time
import threading

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

    return {
        'df_TAs':df_TAs.fillna(0),
        'df_TA_req':df_TA_req.fillna(0),
        'df_times':df_times.fillna(0),
        'disqualified_list':disqualified_list.fillna(0),
        'taken_courses':taken_courses.fillna(0),
        'preferences':preferences.fillna(0),
        'weights':weights.fillna(0),
        'cons':cons.fillna(0),
    }


def find_len(df_TAs, df_TA_req):
    TA_len = len(df_TAs['Abbr.'])
    course_len = len(df_TA_req.columns[1:])
    task_len = len(df_TA_req['Course code'])
    return TA_len, course_len, task_len


def load_cons(cons):
    return {
            'ALLOWED_DEV_H': int(cons['Value'].iloc[1]),
            'MAX_COURSES_H': int(cons['Value'].iloc[2]),
            'MAX_TAS_H': int(cons['Value'].iloc[3]),
            'MAX_SWITCH_H': int(cons['Value'].iloc[4]),
            'ALLOWED_DEV_S': int(cons['Value'].iloc[5]),
            'MAX_COURSES_S': int(cons['Value'].iloc[6]),
            'MAX_TAS_S': int(cons['Value'].iloc[7]),
            'MAX_SWITCH_S': int(cons['Value'].iloc[8]),
            'MIN_CHUNK_HOURS': int(cons['Value'].iloc[11]),
            'ALLOWED_DEV_M': int(cons['Value'].iloc[12]),
            'TIME_RATIO':100
        }

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


def save_result(df_TAs, df_ass, df_c, happ, df_TA_req, df_ta_task,year):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, f'FinalRuns/{year}/{year+1}_Z3.xlsx')  # Change name of Excel document for output

    df_ass.index = df_TA_req.iloc[:, 0]
    df_ta_task.index = df_TA_req.iloc[:, 0]
    df_c.index = happ.iloc[:, 0]

    with pd.ExcelWriter(excel_path) as writer:
        df_TAs.to_excel(writer, sheet_name='TAs', index=False)
        df_ass.to_excel(writer, sheet_name='assigned_TAs', index=True)
        df_c.to_excel(writer, sheet_name='taken_courses', index=True)
        happ.to_excel(writer, sheet_name='happiness', index=False)
        df_ta_task.to_excel(writer, sheet_name='df_ta_task', index=True)

# -----------------------------
# SOLVER
# -----------------------------
def create_model(year, timeout_ms=None):

    solver = Optimize()
    if timeout_ms is not None:
        solver.set("timeout", timeout_ms)
        print('Timeout in seconds: ',timeout_ms // 1000)

    config = preprocess_data(year)

    config['df_TAs']['Target'] = config['df_TAs']['Part of full time'] * 350 + config['df_TAs']['From prev. year']


    const = load_cons(config['cons'])

    TA_len, course_len, task_len = find_len(config['df_TAs'],config['df_TA_req'])

    t2c = [[[Int(f"{TA}c{c}t{t}") for t in range(task_len)] for c in range(course_len)] for TA in range(TA_len)]
    is_active = [[[Bool(f"active_{TA}_{c}_{t}") for t in range(task_len)] for c in range(course_len)] for TA in range(TA_len)]

    TA_total_time = [Sum([t2c[TA][c][t] for c in range(course_len) for t in range(task_len)]) for TA in range(TA_len)]
    c_taught = [[Bool(f"{TA}_to_{c}") for c in range(course_len)] for TA in range(TA_len)]
    switched_c = [Int(f"{TA}_switch") for TA in range(TA_len)]
    assigned_TAs = [[Int(f"{c}{t}_ass") for c in range(course_len)] for t in range(task_len)]

    for c in range(course_len):
        for t in range(task_len):
            assigned_TAs[t][c] = Sum([ If(t2c[TA][c][t] > 0, 1, 0) for TA in range(TA_len)])

    # -----------------------------
    # CONSTRAINTS
    # -----------------------------

    TA_index = {name: i for i, name in enumerate(config['df_TAs']['Abbr.'])}
    C_index = {course: i for i, course in enumerate(config['df_TA_req'].columns[1:])}

    for _, row in config['disqualified_list'].iterrows():
        TA_name = row['TA']
        course_name = row['Course']
        TA = TA_index[TA_name]
        c = C_index[course_name]

        for t in range(task_len):
            solver.add(t2c[TA][c][t] == 0)

    TA_index = {name: i for i, name in enumerate(config['taken_courses']['Abbr.'])}
    course_index = {course: j for j, course in enumerate(config['taken_courses'].columns[1:])}

    for TA in range(TA_len):
        TA_name = config['df_TAs']['Abbr.'].iloc[TA]

        # Skip new teachers
        if TA_name not in TA_index:
            solver.add(switched_c[TA] == 0)
            continue
        last_i = TA_index[TA_name]

        switch_c = []
        for c in range(course_len):
            c_name = config['df_TA_req'].columns[c + 1]

            # Skip new courses
            if c_name not in course_index:
                continue
            last_j = course_index[c_name] + 1

            # Count switched courses
            switch_c.append(If(And(c_taught[TA][c], BoolVal(config['taken_courses'].iloc[last_i, last_j] == 0)), 1, 0))

        # Check switched courses are not too many
        solver.add(switched_c[TA] == Sum(switch_c))
        solver.add(switched_c[TA] <= const['MAX_SWITCH_H'])
        solver.add_soft(switched_c[TA] <= const['MAX_SWITCH_S'], weight=int(config['weights']['Weight'].iloc[0]))

    for c in range(course_len):
        for t in range(task_len):

            # Set non-existing tasks to zero
            if float(config['df_times'].iloc[t, c + 1]) == 0:
                for TA in range(TA_len):
                    solver.add(t2c[TA][c][t] == 0)
                continue

            # Check each TA teach within task limits
            task_time = int(float(config['df_times'].iloc[t, c + 1]) * const['TIME_RATIO'])

            # # Get required number of TAs for this specific task
            # req_tas = int(config['df_TA_req'].iloc[t, c + 1])
            #
            # # Calculate the theoretical maximum chunk size
            # if req_tas > 0:
            #     theoretical_max_per_person = task_time // req_tas
            # else:
            #     theoretical_max_per_person = task_time
            #
            # target_min_chunk = int(const['MIN_CHUNK_HOURS'] * const['TIME_RATIO'])
            #
            # # The actual minimum is the smaller of:
            # min_assign = min(target_min_chunk, theoretical_max_per_person, task_time)
            #
            # for TA in range(TA_len):
            #     # If active: must be at least the calculated min_assign
            #     solver.add(t2c[TA][c][t] >= If(is_active[TA][c][t], min_assign, 0))
            #
            #     # If active: must be at most the total task time
            #     solver.add(t2c[TA][c][t] <= If(is_active[TA][c][t], task_time, 0))

            for TA in range(TA_len):
                solver.add(t2c[TA][c][t] >= 0, t2c[TA][c][t] <= task_time)

            # Check tasks are fully covered
            solver.add(Sum([t2c[TA][c][t] for TA in range(TA_len)]) == task_time)


    # Set deviation to zero for TAs in their final year
    for TA in range(TA_len):
        if config['df_TAs']['Year'].iloc[TA] >= 5:
            solver.add_soft(TA_total_time[TA] == int(config['df_TAs']['Target'].iloc[TA] * const['TIME_RATIO']),
                            weight=int(config['weights']['Weight'].iloc[1]))

    # Check deviation not being too big
    TA_dev = [Abs(TA_total_time[TA] - int(config['df_TAs']['Target'].iloc[TA] * const['TIME_RATIO'])) for TA in range(TA_len)]
    for TA in range(TA_len):
        solver.add(TA_dev[TA] < const['ALLOWED_DEV_H'] * const['TIME_RATIO'])
        solver.add_soft(TA_dev[TA] < const['ALLOWED_DEV_S'] * const['TIME_RATIO'], weight=int(config['weights']['Weight'].iloc[2]))

    # Check what courses TAs teach
    for TA in range(TA_len):
        for c in range(course_len):
            solver.add(c_taught[TA][c] == Or([(t2c[TA][c][t] > 0) for t in range(task_len)]))

    # Check TAs not teaching too many courses
    for TA in range(TA_len):
        solver.add_soft(Sum([If(c_taught[TA][c], 1, 0) for c in range(course_len)]) <= const['MAX_COURSES_S'],
                        weight=int(config['weights']['Weight'].iloc[3]))
        solver.add(Sum([If(c_taught[TA][c], 1, 0) for c in range(course_len)]) <= const['MAX_COURSES_H'])

    # Check courses not having too many TAs
    for c in range(course_len):
        solver.add(Sum([If(c_taught[TA][c], 1, 0) for TA in range(TA_len)]) <= const['MAX_TAS_H'])

        # Check course admin is only one person
        solver.add(assigned_TAs[0][c] <= 1)

        for t in range(task_len):
            # Check assigned TAs are enough
            solver.add(assigned_TAs[t][c] >= int(config['df_TA_req'].iloc[t, c + 1]))
            solver.add_soft(assigned_TAs[t][c] <= (int(config['df_TA_req'].iloc[t, c + 1]) + const['MAX_TAS_S']),
                            weight=int(config['weights']['Weight'].iloc[4]))

    # Teach preferred courses
    for TA in range(TA_len):
        for c in range(course_len):
            if config['preferences'].iloc[TA, c] == 1:
                solver.add_soft(c_taught[TA][c], weight=int(config['weights']['Weight'].iloc[5]))
            elif config['preferences'].iloc[TA, c] == -1:
                solver.add_soft(Not(c_taught[TA][c]), weight=int(config['weights']['Weight'].iloc[6]))

    # Run solver
    result_container = {"status": None}

    def run_solver():
        result_container["status"] = solver.check()

    t = threading.Thread(target=run_solver)
    t.start()
    start = time.time()
    last_update = start
    print("Solver started...")

    # Progress loop
    while t.is_alive():
        now = time.time()
        if now - last_update >= 5:
            print(f"{now - start:0.1f}s elapsed… still solving…")
            last_update = now
        time.sleep(0.1)

    feas = solver.check()
    print(feas)

    m = solver.model()
    print('Model size: ',len(m))

    # Print results
    status = result_container["status"]

    if status == sat:
        print("Returning Optimal Solution.")

    elif status == unknown:
        print("Timeout reached – Returning best solution so far.")
    elif status == unsat:
        print("No solution exists.")

    # Find TA time results
    config['df_TAs']['Assigned'] = [m.evaluate(time_expr).as_long() / const['TIME_RATIO'] for time_expr in TA_total_time]
    config['df_TAs']['Deviation'] = [(m.evaluate(time_expr).as_long() / const['TIME_RATIO']) - config['df_TAs']['Target'].iloc[TA]
                                     for TA, time_expr in enumerate(TA_total_time)]

    # Find TAs per task
    TA_ass = [[m.evaluate(assigned_TAs[t][c]).as_long() for c in range(course_len)] for t in range(task_len)]
    df_ass = pd.DataFrame(TA_ass, index=config['df_TA_req']['Course code'], columns=config['df_TA_req'].columns[1:])

    # Find what courses each TA is assigned
    c2TA = [[1 if is_true(m.evaluate(c_taught[TA][c])) else 0 for c in range(course_len)] for TA in range(TA_len)]
    df_c = pd.DataFrame(c2TA, index=config['df_TAs']['Abbr.'], columns=config['df_TA_req'].columns[1:])

    # Calculate happiness
    TA_happy = TA_happiness(TA_len, course_len, df_c, config['taken_courses'], config['preferences'], config['df_TAs'], config['df_TA_req'])

    # Find what task each teacher is assigned
    matrix_data = []
    for t in range(task_len):
        row = []
        for c in range(course_len):
            ta_strings = []
            for TA in range(TA_len):
                val = m.evaluate(t2c[TA][c][t]).as_long()
                if val > 0:  # Only include TAs assigned to this task
                    ta_strings.append(f"{config['df_TAs']['Abbr.'].iloc[TA]}: {val / const['TIME_RATIO']}")
            row.append(", ".join(ta_strings))
        matrix_data.append(row)
    df_ta_task = pd.DataFrame(matrix_data, index=[f"Task {t}" for t in range(task_len)], columns=config['df_TA_req'].columns[1:])

    return config['df_TAs'], df_ass, df_c, TA_happy, df_ta_task

# -----------------------------
# MAIN
# -----------------------------

def main(year):
    # Pre-processing
    config = preprocess_data(year)

    TA_len, course_len, task_len = find_len(config['df_TAs'], config['df_TA_req'])

    TIME_LIMIT_SECONDS = int(config['cons']['Value'].iloc[0])

    start_time = tm()

    # Initiate solver
    df_TAs, df_ass, df_c, TA_happy, df_ta_task = create_model(year,timeout_ms=TIME_LIMIT_SECONDS * 1000)


    end_time = tm()-start_time

    # Save
    save_result(df_TAs, df_ass, df_c, TA_happy, config['df_TA_req'], df_ta_task,year)
    return round(end_time,2)


# Run script
if __name__ == "__main__":
    year = 2026
    process_time = main(year)

    print(f'Solving Time: {process_time}')