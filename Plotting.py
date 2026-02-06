import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_year_csv(year: str, filename: str = "data.csv", sep: str = ";"):
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    file_path = base_dir / "TASP/Input_Output" / year / filename

    if file_path.exists():
        return pd.read_csv(file_path, sep=sep)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def read_year_excel(year: str, filename: str, sheet_name: str):
    """
    Reads a specific sheet from an Excel file located at:
    <base_dir>/<year>/<filename>
    """
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    file_path = base_dir / "TASP/Input_Output" /year / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_excel(file_path, sheet_name=sheet_name)

def plot_assigned_vs_target_on_ax(ax, solver: list, year: int, sep_solver=";", sep_manual=",", solver_labels=None, save: bool = False, legend=False):
    """Plot Assigned vs Target for both Solver and Manual datasets for a given year."""

    solver_labels = solver_labels if solver_labels is not None else solver
    colors = plt.cm.get_cmap('Paired').colors
    method_colors = {m: colors[k % len(colors)] for k, m in enumerate(solver + ["Manual"])}

    # Generate filenames dynamically
    manual_filename = f"OutputData(plot_manual_{year}).csv"

    # Read Solver and Manual datasets
    df_solver = {name:read_year_excel(str(year), f"{year+1}_{name}.xlsx", sheet_name="TAs") for name in solver}
    df_manual = read_year_csv(str(year), manual_filename, sep=sep_manual)

    # Scatter plots
    for i, name in enumerate(solver):
        ax.scatter(df_solver[name]["Target"], df_solver[name]["Assigned"], label=f"{solver_labels[i]}", alpha=0.7, color=method_colors[name])
    ax.scatter(df_manual["Target"], df_manual["Assigned"], label="Manual", alpha=0.7, color=method_colors["Manual"])

    # reference line
    min_buffer = [df_solver[name]["Target"].min() for name in solver]
    min_buffer.append(df_manual["Target"].min())
    max_buffer = [df_solver[name]["Target"].max() for name in solver]
    max_buffer.append(df_manual["Target"].max())
    x_min = min(min_buffer)*0.95
    x_max = max(max_buffer)*1.05
    x = np.linspace(x_min, x_max, 100)
    ax.plot(x, x, color="black", linestyle="-", linewidth=0.5)#, label="y=x")

    # Labels, title, legend
    ax.set_xlabel("Target", fontsize=14)
    ax.set_ylabel("Assigned", fontsize=14)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    if legend:
        ax.legend(loc='upper left', fontsize=11, title_fontsize=14)


    # Save automatically as PDF with the year in filename
    if save:
        script_dir = Path(__file__).resolve().parent
        filename_pdf = script_dir / f"Plots/Assigned_vs_Target_{year}_{'-'.join(solver)}.pdf"
        plt.savefig(filename_pdf, format="pdf")
        print(f"Plot saved as {filename_pdf}")

def plot_assigned_vs_target(year, save=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_assigned_vs_target_on_ax(ax, Solvers, year, sep_solver=";", sep_manual=",", solver_labels=Solver_labels,legend=True)
    # ax.set_title(f'Assigned vs Target for {year}', fontsize=16)

    if save:
        script_dir = Path(__file__).resolve().parent
        filename_pdf = script_dir / f"Plots/AssignedVsTarget/Assigned_vs_Target_{year}.pdf"
        plt.savefig(filename_pdf, format="pdf")

def plot_assigned_vs_target_collected(save=False):

    Solvers = ['GUROBI', 'SCIP', 'SAT', 'Z3']
    Solver_labels = ['Gurobi', 'SCIP', 'CP-SAT', 'Z3']

    fig, axs = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True, layout='constrained')
    plot_assigned_vs_target_on_ax(axs[0][0], Solvers, 2023, sep_solver=";", sep_manual=",", solver_labels=Solver_labels, legend=True)
    plot_assigned_vs_target_on_ax(axs[0][1], Solvers, 2024, sep_solver=";", sep_manual=",", solver_labels=Solver_labels, legend=False)
    plot_assigned_vs_target_on_ax(axs[1][0], Solvers, 2025, sep_solver=";", sep_manual=",", solver_labels=Solver_labels, legend=False)
    plot_assigned_vs_target_on_ax(axs[1][1], Solvers, 2026, sep_solver=";", sep_manual=",", solver_labels=Solver_labels, legend=False)

    axs[0][0].set_xlabel('', fontsize=14)
    axs[0][1].set_xlabel('', fontsize=14)
    axs[1][0].set_xlabel('', fontsize=14)
    axs[1][1].set_xlabel('', fontsize=14)
    axs[0][0].set_ylabel('')
    axs[0][1].set_ylabel('')
    axs[1][0].set_ylabel('')
    axs[1][1].set_ylabel('')
    axs[0][0].set_title('2023', fontsize=14)
    axs[0][1].set_title('2024', fontsize=14)
    axs[1][0].set_title('2025', fontsize=14)
    axs[1][1].set_title('2026', fontsize=14)

    # fig.suptitle('Assigned Vs Target Hours by Year', fontsize=16)
    fig.supylabel('Assigned Hours', fontsize=14)
    fig.supxlabel('Target Hours', fontsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    if save:
        script_dir = Path(__file__).resolve().parent
        filename_pdf = script_dir / f"Plots/AssignedVsTarget/Assigned_vs_Target_2023-2026.pdf"
        plt.savefig(filename_pdf, format="pdf")

def compute_rmSE_assigned_vs_target(solver: str, year: int, sep_solver=";", sep_manual=","):
    """
    Computes and prints RMSE between Assigned and Target
    for Solver and Manual datasets for a given year.
    """

    # Filenames
    manual_filename = f"OutputData(plot_manual_{year}).csv"

    # Read data
    df_solver = read_year_excel(str(year), f"{year+1}_{solver}.xlsx", sheet_name="TAs")
    df_manual = read_year_csv(str(year), manual_filename, sep=sep_manual)

    # RMSE calculation
    solver_rmse = np.sqrt(
        np.mean((df_solver["Assigned"] - df_solver["Target"]) ** 2)
    )

    manual_rmse = np.sqrt(
        np.mean((df_manual["Assigned"] - df_manual["Target"]) ** 2)
    )

    print(f"Year {year}")
    print(f"  Solver RMSE : {solver_rmse:.2f}")
    print(f"  Manual RMSE : {manual_rmse:.2f}")
    print("-" * 30)

    return solver_rmse, manual_rmse

def get_course_data():

    def aggregate_course_data(assigned_courses, new_courses):

        assigned_c = dict(assigned_courses.value_counts())
        new_c = dict(new_courses.value_counts())

        return assigned_c, new_c

    solvers = ['GUROBI', 'SCIP', 'SAT', 'Z3']
    solver_labels = ['Gurobi', 'SCIP', 'CP-SAT', 'Z3']
    years = [2022, 2023, 2024, 2025, 2026]

    data = dict()

    # solvers
    for i, solver in enumerate(solvers):
        solver_label = solver_labels[i]
        data[solver_label] = dict()
        for year in years:
            df_solver = read_year_excel(str(year), f"{year + 1}_{solver}.xlsx", sheet_name="happiness")
            assigned_c, new_c = aggregate_course_data(df_solver['Taught courses'], df_solver['Switched Courses'])
            data[solver_label][year] = {'assigned': assigned_c, 'new': new_c}

    # manual
    data['Manual'] = dict()
    for i, year in enumerate(years):
        df_manual = read_year_csv(str(year), f"{year}(num_courses).csv", sep=",")
        assigned_c, new_c = aggregate_course_data(df_manual['num_man'], df_manual['new_man'])

        data['Manual'][year] = {'assigned': assigned_c, 'new': new_c}

    return data

def plot_a_course_histogram(data, ax, year, metric, solvers=None, x_label=None, y_label=None, legend=True):

    # Params
    bar_width = 0.13
    spacing = 0.01
    colors = plt.cm.get_cmap('Paired').colors

    # Init
    solvers = list(data.keys()) if solvers is None else solvers
    method_colors = {m: colors[k % len(colors)] for k, m in enumerate(solvers)}

    # plotting
    all_keys = set()
    for solver in solvers:
        all_keys.update(data[solver][year][metric].keys())
    sorted_keys = sorted(list(all_keys))
    x_indices = np.arange(len(sorted_keys))

    for j, solver in enumerate(solvers):
        solver_data = data[solver][year][metric]

        heights = [solver_data.get(k, 0) for k in sorted_keys]
        offset = (j - len(solvers) / 2 + 0.5) * (bar_width + spacing)

        ax.bar(x_indices + offset, heights,
               width=bar_width,
               label=solver,
               color=method_colors[solver],
               alpha=1)

    # formatting
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(sorted_keys)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if legend:
        ax.legend(loc='upper right', fontsize=11, title_fontsize=14)

def plot_course_histogram_collected():

    data = get_course_data()

    # Assigned courses
    fig, axs = plt.subplots(2, 2, figsize=(14, 4), sharey=True, sharex=True, layout='constrained')
    plot_a_course_histogram(data, axs[0][0], year=2023, metric='assigned', solvers=None, x_label='', legend=False)
    plot_a_course_histogram(data, axs[0][1], year=2024, metric='assigned', solvers=None, x_label='', legend=True)
    plot_a_course_histogram(data, axs[1][0], year=2025, metric='assigned', solvers=None, x_label='', legend=False)
    plot_a_course_histogram(data, axs[1][1], year=2026, metric='assigned', solvers=None, x_label='', legend=False)
    axs[0][0].set_title('2023', fontsize=14)
    axs[0][1].set_title('2024', fontsize=14)
    axs[1][0].set_title('2025', fontsize=14)
    axs[1][1].set_title('2026', fontsize=14)

    # fig.suptitle('Number of Assigned Courses by Year', fontsize=16)
    fig.supylabel('Number of TAs', fontsize=14)
    fig.supxlabel('Number of Assigned Courses', fontsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    save_path = Path(__file__).resolve().parent / f"Plots/AssignedAndNewCourses/Num_Assigned_Courses_Collected_2023-2026.pdf"
    plt.savefig(save_path, format="pdf")

    # New courses
    fig, axs = plt.subplots(2, 2, figsize=(14, 4), sharey=True, sharex=True, layout='constrained')
    plot_a_course_histogram(data, axs[0][0], year=2023, metric='new', solvers=None, x_label='', legend=False)
    plot_a_course_histogram(data, axs[0][1], year=2024, metric='new', solvers=None, x_label='', legend=True)
    plot_a_course_histogram(data, axs[1][0], year=2025, metric='new', solvers=None, x_label='', legend=False)
    plot_a_course_histogram(data, axs[1][1], year=2026, metric='new', solvers=None, x_label='', legend=False)
    axs[0][0].set_title('2023', fontsize=14)
    axs[0][1].set_title('2024', fontsize=14)
    axs[1][0].set_title('2025', fontsize=14)
    axs[1][1].set_title('2026', fontsize=14)

    # fig.suptitle('Number of Switched Courses by Year', fontsize=16)
    fig.supylabel('Number of TAs', fontsize=14)
    fig.supxlabel('Number of New Courses', fontsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    save_path = Path(__file__).resolve().parent / f"Plots/AssignedAndNewCourses/Num_Switched_Courses_Collected_2023-2026.pdf"
    plt.savefig(save_path, format="pdf")

def plot_double_histogram(year, solvers=None):

    data = get_course_data()

    solvers = list(data.keys()) if solvers is None else solvers

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharey=True, layout='constrained')
    plot_a_course_histogram(data, axs[0], year=year, metric='assigned', solvers=solvers,
                            x_label=f'Number of Assigned Courses', y_label='Number of TAs', legend=True)
    plot_a_course_histogram(data, axs[1], year=year, metric='new', solvers=solvers,
                            x_label=f'Number of New Courses', y_label='Number of TAs', legend=False)
    plt.subplots_adjust(hspace=0.4)
    save_path = Path(__file__).resolve().parent / f"Plots/AssignedAndNewCourses/Num_Courses_{year}.pdf"
    plt.savefig(save_path, format="pdf")


# Run script
if __name__ == "__main__":

    Solvers = ['GUROBI', 'SCIP', 'SAT', 'Z3']
    Solver_labels = ['Gurobi', 'SCIP', 'CP-SAT', 'Z3']
    Years = [2022] #, 2023, 2024, 2025, 2026]

    plot_course_histogram_collected()
    plot_double_histogram(2022)
    plot_assigned_vs_target(2022, save=True)
    plot_assigned_vs_target_collected(save=True)


    """    
    # for solver in Solvers:
    for year in Years:
        # plot_course_histogram(solver,year, sep=",", save=True)
        # compute_rmSE_assigned_vs_target(solver,year, sep_solver=";", sep_manual=",")
    """

    # plot_course_histogram(Solvers, Years, solver_labels=Solver_labels)

