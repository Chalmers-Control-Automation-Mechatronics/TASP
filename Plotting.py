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
    

def read_year_excel(
    year: str,
    filename: str,
    sheet_name: str
):
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


def plot_assigned_vs_target(solver: list, year: int, sep_solver=";", sep_manual=",", save: bool = True):
    """Plot Assigned vs Target for both Solver and Manual datasets for a given year."""
    
    # Generate filenames dynamically
    manual_filename = f"OutputData(plot_manual_{year}).csv"



    # Read Solver and Manual datasets
    df_solver = {name:read_year_excel(str(year), f"{year+1}_{name}.xlsx", sheet_name="TAs") for name in solver}
    df_manual = read_year_csv(str(year), manual_filename, sep=sep_manual)

    # Scatter plots
    for name in solver:
        plt.scatter(df_solver[name]["Target"], df_solver[name]["Assigned"], label=f"{name}", alpha=0.7)
    plt.scatter(df_manual["Target"], df_manual["Assigned"], label="Manually", alpha=0.7)

    # reference line
    min_buffer = [df_solver[name]["Target"].min() for name in solver]
    min_buffer.append(df_manual["Target"].min())
    max_buffer = [df_solver[name]["Target"].max() for name in solver]
    max_buffer.append(df_manual["Target"].max())
    x_min = min(min_buffer)
    x_max = max(max_buffer)
    x = np.linspace(x_min, x_max, 100)
    plt.plot(x, x, color="red", linestyle="--")#, label="y=x")

    # Labels, title, legend
    plt.xlabel("Target")
    plt.ylabel("Assigned")
    plt.title(f"Assigned vs Target {year}-{'-'.join(solver)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save automatically as PDF with the year in filename
    if save:
        script_dir = Path(__file__).resolve().parent
        filename_pdf = script_dir / f"Plots/Assigned_vs_Target_{year}_{'-'.join(solver)}.pdf"
        plt.savefig(filename_pdf, format="pdf")
        print(f"Plot saved as {filename_pdf}")

    plt.show()


def compute_rmSE_assigned_vs_target(
    solver: str,
    year: int,
    sep_solver=";",
    sep_manual=","
):
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


def plot_course_histogram(solver: str, year: int, sep=",", save=True):
    """
    Creates 4 separate histograms:
    1) Solver total courses
    2) Manual total courses
    3) Solver new courses
    4) Manual new courses
    """

    # Manual data
    filename = f"{year}(num_courses).csv"
    df_manual = read_year_csv(str(year), filename, sep=sep)
    df = df_manual.copy()

    # Solver data
    df_solver = read_year_excel(str(year), f"{year + 1}_{solver}.xlsx", sheet_name="happiness")

    # Prepare data
    solver_total = df_solver["Taught courses"]
    manual_total = df_manual["num_man"]
    solver_new = df_solver["Switched Courses"]
    manual_new = df_manual["new_man"]

    # Compute suitable bin range
    max_courses = max(
        solver_total.max(),
        manual_total.max(),
        solver_new.max(),
        manual_new.max(),
    )
    bins = np.arange(-0.5, max_courses + 1.5, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    # Define colors
    solver_color = "#1f77b4"  # default blue
    manual_color = "#ff7f0e"  # orange

    # --- Solver total ---
    axes[0, 0].hist(solver_total, bins=bins, alpha=0.8, color=solver_color)
    axes[0, 0].set_title(f"Solver – Total Courses ({year})", fontsize=22)

    # --- Manual total ---
    axes[0, 1].hist(manual_total, bins=bins, alpha=0.8, color=manual_color)
    axes[0, 1].set_title(f"Manual – Total Courses ({year})", fontsize=22)

    # --- Solver new ---
    axes[1, 0].hist(solver_new, bins=bins, alpha=0.8, color=solver_color)
    axes[1, 0].set_title(f"Solver – New Courses ({year})", fontsize=22)

    # --- Manual new ---
    axes[1, 1].hist(manual_new, bins=bins, alpha=0.8, color=manual_color)
    axes[1, 1].set_title(f"Manual – New Courses ({year})", fontsize=22)

    # Formatting for all subplots
    for ax in axes.flatten():
        ax.set_xlabel("Number of Courses", fontsize=20)
        ax.set_ylabel("Number of TAs", fontsize=20)
        ax.set_xticks(range(0, max_courses + 1))
        ax.set_yticks(range(0, df.shape[0] + 1, 5))
        ax.tick_params(axis='both', labelsize=12)  # tick labels
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save PDF
    if save:
        out = Path(__file__).resolve().parent / f"Plots/Histogram_CourseStats_{year}_{solver}.pdf"
        plt.savefig(out, format="pdf")
        print(f"Saved: {out}")

    plt.show()


def plot_num_courses_collected(solvers, years, include_manual=True):

    # ---- collect data ----
    def aggregate_course_data(assigned_courses, new_courses):

        assigned_c = dict(assigned_courses.value_counts())
        new_c = dict(new_courses.value_counts())

        return assigned_c, new_c

    data = dict()

    # manual
    data['Manual'] = []
    for i, year in enumerate(years):

        df_manual = read_year_csv(str(year), f"{year}(num_courses).csv", sep=",")
        assigned_c, new_c = aggregate_course_data(df_manual['num_man'], df_manual['new_man'])

        data['Manual'].append({
            'assigned': assigned_c,
            'new': new_c})

    # solvers
    for solver in solvers:
        data[solver] = []
        for year in years:
            df_solver = read_year_excel(str(year), f"{year + 1}_{solver}.xlsx", sheet_name="happiness")
            assigned_c, new_c = aggregate_course_data(df_solver['Taught courses'], df_solver['Switched Courses'])
            data[solver].append({
                'assigned': assigned_c,
                'new': new_c})


    # ---- create plot ----
    bar_width = 0.13
    spacing = 0.01
    colors = plt.cm.get_cmap('Paired').colors

    methods = solvers + (['Manual'] if include_manual else [])
    method_colors = {m: colors[k % len(colors)] for k, m in enumerate(methods)}

    for i, year in enumerate(years):
        fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharey=True, layout='constrained')

        metrics = ['assigned', 'new']
        x_labels = ['Number of Assigned Courses', 'Number of Switched Courses']

        for ax_idx, metric in enumerate(metrics):
            ax = axs[ax_idx]

            all_keys = set()
            for method in methods:
                all_keys.update(data[method][i][metric].keys())
            sorted_keys = sorted(list(all_keys))
            x_indices = np.arange(len(sorted_keys))

            for j, method in enumerate(methods):
                method_data = data[method][i][metric]

                heights = [method_data.get(k, 0) for k in sorted_keys]
                offset = (j - len(methods) / 2 + 0.5) * (bar_width + spacing)

                ax.bar(x_indices + offset, heights,
                       width=bar_width,
                       label=method,
                       color=method_colors[method],
                       alpha=1)

            # formatting
            ax.set_xlabel(x_labels[ax_idx], fontsize=14)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(sorted_keys)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if ax_idx == 0:
                ax.legend(loc='upper left', fontsize=14, title_fontsize=14)


        # Save PDF
        out = Path(__file__).resolve().parent / f"Plots/AssignedAndNewCourses/Num_Courses_Collected_{year}.pdf"
        plt.savefig(out, format="pdf")
        print(f"Saved: {out}")
        plt.show()

        pass


# Run script
if __name__ == "__main__":

    Solvers = ['GUROBI','SAT','SCIP','Z3']#'SCIP','SAT','Z3']
    Years = [2022]#,2023,2024,2025,2026]
    Solvers = ['GUROBI','SCIP','SAT','Z3']
    Years = [2022, 2023, 2024, 2025, 2026]

    # for solver in Solvers:
    for year in Years:
        # plot_course_histogram(solver,year, sep=",", save=True)
        plot_assigned_vs_target(Solvers,year, sep_solver=";", sep_manual=",")
        # compute_rmSE_assigned_vs_target(solver,year, sep_solver=";", sep_manual=",")

    plot_num_courses_collected(Solvers, Years)

    """
    for solver in Solvers:
        for year in Years:
            # plot_course_histogram(solver,year, sep=",", save=True)
            plot_assigned_vs_target(solver,year, sep_solver=";", sep_manual=",")
            # compute_rmSE_assigned_vs_target(solver,year, sep_solver=";", sep_manual=",")
    """
