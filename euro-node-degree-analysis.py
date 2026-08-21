#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
European network node-degree analysis
=====================================

Loads the pickled Hartmann European transmission/sub-transmission
network data and calculates complementary cumulative node-degree
distributions for:

    1. Each complete country network
    2. One combined sub-transmission network containing all
       voltage levels below 200 kV
    3. Each voltage level >= 200 kV individually

For every network, fit:

    P(K >= k) = C * exp(-k / gamma)

with both C and gamma as free parameters.

Parallel circuits are retained. Therefore duplicate edge rows count
as separate branches and contribute separately to node degree.

Outputs
-------
Console:
    Country-by-country summary containing:
        voltage group
        number of nodes
        number of branches
        mean degree
        C
        gamma
        R^2

Plots:
    euro-comparison/
        node-degree-analysis/
            Albania/
            Belgium/
            ...

Tables:
    node_degree_fit_summary.csv
    node_degree_fit_summary.pkl

    fitted_parameter_table.csv
    fitted_parameter_table.pkl

The fitted parameter table has:
    - countries as rows
    - voltage groups as columns
    - separate C and gamma columns for each voltage group
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

DATA_DIR = WORKING_DIR / "euro-comparison"

INPUT_FILE = DATA_DIR / "european_networks.pkl"

OUTPUT_DIR = DATA_DIR / "node-degree-analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200

FIGSIZE = (7.5, 5.5)
DPI = 300


# =====================================================================
# EXPONENTIAL MODEL
# =====================================================================

def exponential_degree_distribution(k, C, gamma):
    """
    Complementary cumulative exponential model:

        P(K >= k) = C * exp(-k / gamma)

    Both C and gamma are fitted freely.
    """

    return C * np.exp(-k / gamma)


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(edges):
    """
    Calculate node degrees directly from the edge list.

    Every row is treated as one physical branch.

    Parallel circuits are therefore retained and contribute
    separately to node degree.

    Parameters
    ----------
    edges : pandas.DataFrame
        Must contain:
            node_i
            node_j

    Returns
    -------
    pandas.Series
        Node ID as index and node degree as value.
    """

    endpoints = pd.concat(
        [
            edges["node_i"],
            edges["node_j"],
        ],
        ignore_index=True,
    )

    return endpoints.value_counts().sort_index()


# =====================================================================
# COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION
# =====================================================================

def calculate_degree_distribution(degrees):
    """
    Convert node degrees into the complementary cumulative
    distribution:

        P(K >= k)

    Therefore:

        P(K >= 1) = 1

    Returns
    -------
    k : numpy.ndarray
        Degree values from 1 through maximum observed degree.

    probability : numpy.ndarray
        Complementary cumulative probability P(K >= k).
    """

    if len(degrees) == 0:
        return np.array([]), np.array([])

    max_degree = int(degrees.max())

    k = np.arange(
        1,
        max_degree + 1,
        dtype=float,
    )

    probability = np.array(
        [
            np.mean(degrees >= degree)
            for degree in k
        ],
        dtype=float,
    )

    return k, probability


# =====================================================================
# EXPONENTIAL FIT
# =====================================================================

def fit_exponential_distribution(k, probability):
    """
    Fit:

        P(K >= k) = C * exp(-k / gamma)

    with both C and gamma free.

    Returns
    -------
    C : float

    gamma : float

    r2 : float

    fitted_probability : numpy.ndarray
    """

    if len(k) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.full_like(k, np.nan),
        )

    # -------------------------------------------------------------
    # Initial estimates
    # -------------------------------------------------------------

    C_initial = 1.5
    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            exponential_degree_distribution,
            k,
            probability,
            p0=[
                C_initial,
                gamma_initial,
            ],
            bounds=(
                [0.0, 1e-8],
                [np.inf, np.inf],
            ),
            maxfev=100000,
        )

        C = float(popt[0])
        gamma = float(popt[1])

    except (RuntimeError, ValueError):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.full_like(k, np.nan),
        )

    # -------------------------------------------------------------
    # Fitted probabilities
    # -------------------------------------------------------------

    fitted_probability = exponential_degree_distribution(
        k,
        C,
        gamma,
    )

    # -------------------------------------------------------------
    # R^2
    # -------------------------------------------------------------

    residual_sum_squares = np.sum(
        (probability - fitted_probability) ** 2
    )

    total_sum_squares = np.sum(
        (probability - np.mean(probability)) ** 2
    )

    if total_sum_squares > 0:

        r2 = 1.0 - (
            residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    return (
        C,
        gamma,
        r2,
        fitted_probability,
    )


# =====================================================================
# ANALYZE ONE EDGE SET
# =====================================================================

def analyze_network(edges):
    """
    Calculate node degrees, cumulative degree distribution,
    exponential fit, and summary statistics for one edge set.
    """

    degrees = calculate_node_degrees(
        edges
    )

    k, probability = calculate_degree_distribution(
        degrees
    )

    (
        C,
        gamma,
        r2,
        fitted_probability,
    ) = fit_exponential_distribution(
        k,
        probability,
    )

    n_nodes = len(degrees)
    n_branches = len(edges)

    if n_nodes > 0:
        mean_degree = degrees.mean()
    else:
        mean_degree = np.nan

    return {
        "degrees": degrees,
        "k": k,
        "probability": probability,
        "fitted_probability": fitted_probability,
        "n_nodes": n_nodes,
        "n_branches": n_branches,
        "mean_degree": mean_degree,
        "C": C,
        "gamma": gamma,
        "r2": r2,
    }


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def get_subtransmission_label(edges):
    """
    Build the display label for all voltage levels below 200 kV.

    Examples
    --------
    One voltage:
        110 kV

    Multiple voltages:
        132–165 kV
    """

    voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            < SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .unique()
        .tolist()
    )

    if len(voltages) == 0:
        return None

    if len(voltages) == 1:
        return f"{voltages[0]} kV"

    return (
        f"{voltages[0]}"
        f"\N{EN DASH}"
        f"{voltages[-1]} kV"
    )


def build_voltage_groups(edges):
    """
    Build the edge sets to analyze for one country.

    Returns
    -------
    list of dict

    Each item contains:
        key
        label
        edges

    Groups:
        all
        combined <200 kV
        each >=200 kV separately
    """

    groups = []

    # -------------------------------------------------------------
    # Complete network
    # -------------------------------------------------------------

    groups.append(
        {
            "key": "All",
            "label": "All voltage levels",
            "edges": edges.copy(),
        }
    )

    # -------------------------------------------------------------
    # Combined sub-transmission network
    # -------------------------------------------------------------

    sub_edges = edges.loc[
        edges["voltage_kv"]
        < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if len(sub_edges) > 0:

        sub_label = get_subtransmission_label(
            edges
        )

        groups.append(
            {
                "key": sub_label,
                "label": sub_label,
                "edges": sub_edges,
            }
        )

    # -------------------------------------------------------------
    # Individual >=200 kV networks
    # -------------------------------------------------------------

    transmission_voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            >= SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .unique()
        .tolist()
    )

    for voltage in transmission_voltages:

        voltage_edges = edges.loc[
            edges["voltage_kv"] == voltage
        ].copy()

        label = f"{voltage} kV"

        groups.append(
            {
                "key": label,
                "label": label,
                "edges": voltage_edges,
            }
        )

    return groups


# =====================================================================
# SAFE FILE NAMES
# =====================================================================

def safe_filename(text):
    """
    Convert a display label into a simple filename component.
    """

    return (
        str(text)
        .replace(" ", "_")
        .replace("\N{EN DASH}", "-")
        .replace("/", "-")
    )


# =====================================================================
# PLOT
# =====================================================================

def plot_degree_distribution(
    result,
    country,
    voltage_label,
    output_path,
):
    """
    Plot empirical complementary cumulative degree distribution
    and fitted exponential curve.
    """

    k = result["k"]
    probability = result["probability"]

    C = result["C"]
    gamma = result["gamma"]
    r2 = result["r2"]

    n_nodes = result["n_nodes"]
    n_branches = result["n_branches"]
    mean_degree = result["mean_degree"]

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # -------------------------------------------------------------
    # Observed cumulative distribution
    # -------------------------------------------------------------

    ax.plot(
        k,
        probability,
        "o",
        markersize=6,
        label="Observed cumulative distribution",
    )

    # -------------------------------------------------------------
    # Smooth fitted curve
    # -------------------------------------------------------------

    if (
        np.isfinite(C)
        and np.isfinite(gamma)
    ):

        k_smooth = np.linspace(
            1,
            max(k),
            500,
        )

        p_smooth = exponential_degree_distribution(
            k_smooth,
            C,
            gamma,
        )

        ax.plot(
            k_smooth,
            p_smooth,
            linewidth=2.0,
            label="Exponential fit",
        )

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Node degree, $k$"
    )

    ax.set_ylabel(
        r"Cumulative probability, $P(K \geq k)$"
    )

    ax.set_title(
        f"{country} — {voltage_label}"
    )

    if len(k) > 0:

        max_degree = int(
            max(k)
        )

        ax.set_xticks(
            np.arange(
                1,
                max_degree + 1,
                1,
            )
        )

    ax.set_ylim(
        0,
        1.05,
    )

    # -------------------------------------------------------------
    # Fit annotation
    # -------------------------------------------------------------

    if (
        np.isfinite(C)
        and np.isfinite(gamma)
    ):

        fit_text = (
            r"$P(K\geq k)=Ce^{-k/\gamma}$"
            "\n"
            rf"$C = {C:.4f}$"
            "\n"
            rf"$\gamma = {gamma:.4f}$"
            "\n"
            rf"$R^2 = {r2:.4f}$"
            "\n"
            rf"$\langle k\rangle = {mean_degree:.4f}$"
            "\n"
            rf"$N = {n_nodes}$"
            "\n"
            rf"$E = {n_branches}$"
        )

    else:

        fit_text = (
            "Fit unavailable"
            "\n"
            rf"$N = {n_nodes}$"
            "\n"
            rf"$E = {n_branches}$"
        )

    ax.text(
        0.97,
        0.97,
        fit_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(
            1.0,
            0.57,
        ),
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# ANALYZE ONE COUNTRY
# =====================================================================

def analyze_country(
    country,
    edges,
):
    """
    Analyze the complete network and the grouped voltage networks
    for one country.
    """

    country_output_dir = (
        OUTPUT_DIR
        / country
    )

    country_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    voltage_groups = build_voltage_groups(
        edges
    )

    for group in voltage_groups:

        group_key = group["key"]
        group_label = group["label"]
        group_edges = group["edges"]

        result = analyze_network(
            group_edges
        )

        # ---------------------------------------------------------
        # Plot filename
        # ---------------------------------------------------------

        if group_key == "All":

            filename = (
                f"{country}_all_voltages.png"
            )

        else:

            filename = (
                f"{country}_"
                f"{safe_filename(group_key)}"
                ".png"
            )

        plot_path = (
            country_output_dir
            / filename
        )

        plot_degree_distribution(
            result=result,
            country=country,
            voltage_label=group_label,
            output_path=plot_path,
        )

        # ---------------------------------------------------------
        # Summary row
        # ---------------------------------------------------------

        summary_rows.append(
            {
                "country": country,
                "voltage_group": group_key,
                "n_nodes": result["n_nodes"],
                "n_branches": result["n_branches"],
                "mean_degree": result["mean_degree"],
                "C": result["C"],
                "gamma": result["gamma"],
                "r2": result["r2"],
            }
        )

    return summary_rows


# =====================================================================
# PRINT COUNTRY SUMMARY
# =====================================================================

def print_summary(summary):
    """
    Print results grouped by country.
    """

    print("\n")
    print("=" * 112)
    print(
        "EUROPEAN NODE-DEGREE "
        "CUMULATIVE EXPONENTIAL FITS"
    )
    print("=" * 112)

    for country in summary["country"].unique():

        country_results = summary.loc[
            summary["country"]
            == country
        ]

        print("\n")
        print(country)
        print("-" * 112)

        print(
            f"{'Voltage':>18} "
            f"{'Nodes':>10} "
            f"{'Branches':>10} "
            f"{'<k>':>12} "
            f"{'C':>12} "
            f"{'gamma':>12} "
            f"{'R2':>12}"
        )

        for _, row in country_results.iterrows():

            print(
                f"{row['voltage_group']:>18} "
                f"{int(row['n_nodes']):>10d} "
                f"{int(row['n_branches']):>10d} "
                f"{row['mean_degree']:>12.4f} "
                f"{row['C']:>12.4f} "
                f"{row['gamma']:>12.4f} "
                f"{row['r2']:>12.4f}"
            )


# =====================================================================
# PARAMETER SUMMARY TABLE
# =====================================================================

def build_parameter_table(summary):
    """
    Build a wide table with countries as rows and voltage groups
    as columns.

    Each voltage group receives two columns:

        <voltage>_C
        <voltage>_gamma
    """

    C_table = summary.pivot(
        index="country",
        columns="voltage_group",
        values="C",
    )

    gamma_table = summary.pivot(
        index="country",
        columns="voltage_group",
        values="gamma",
    )

    # -------------------------------------------------------------
    # Combine into a two-level column index
    # -------------------------------------------------------------

    parameter_table = pd.concat(
        {
            "C": C_table,
            "gamma": gamma_table,
        },
        axis=1,
    )

    # -------------------------------------------------------------
    # Reorder so each voltage group has C then gamma
    # -------------------------------------------------------------

    voltage_groups = []

    for group in summary["voltage_group"]:

        if group not in voltage_groups:
            voltage_groups.append(
                group
            )

    ordered_columns = []

    for voltage_group in voltage_groups:

        if (
            "C",
            voltage_group,
        ) in parameter_table.columns:

            ordered_columns.append(
                (
                    "C",
                    voltage_group,
                )
            )

        if (
            "gamma",
            voltage_group,
        ) in parameter_table.columns:

            ordered_columns.append(
                (
                    "gamma",
                    voltage_group,
                )
            )

    parameter_table = (
        parameter_table.loc[
            :,
            ordered_columns,
        ]
    )

    # -------------------------------------------------------------
    # Flip levels:
    #
    #   voltage group
    #       C
    #       gamma
    #
    # reads more naturally in the CSV/table.
    # -------------------------------------------------------------

    parameter_table.columns = (
        parameter_table.columns
        .swaplevel(
            0,
            1,
        )
    )

    return parameter_table


def print_parameter_table(
    parameter_table,
):
    """
    Print the wide fitted-parameter table.
    """

    print("\n")
    print("=" * 112)
    print("FITTED PARAMETER SUMMARY")
    print("=" * 112)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.precision",
        4,
    ):

        print(
            parameter_table.to_string()
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 112)
    print(
        "LOADING EUROPEAN NETWORK DATA"
    )
    print("=" * 112)

    print(
        f"Input:\n"
        f"  {INPUT_FILE}"
    )

    euro_networks = pd.read_pickle(
        INPUT_FILE
    )

    print(
        f"\nLoaded "
        f"{len(euro_networks)} "
        f"countries."
    )

    # -----------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------

    all_summary_rows = []

    for country in sorted(
        euro_networks
    ):

        edges = euro_networks[
            country
        ]

        print(
            f"Analyzing "
            f"{country:<25} "
            f"({len(edges):>5} branches)"
        )

        country_summary = (
            analyze_country(
                country,
                edges,
            )
        )

        all_summary_rows.extend(
            country_summary
        )

    # -----------------------------------------------------------------
    # Long-form results table
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary_rows
    )

    print_summary(
        summary
    )

    # -----------------------------------------------------------------
    # Wide fitted-parameter table
    # -----------------------------------------------------------------

    parameter_table = (
        build_parameter_table(
            summary
        )
    )

    print_parameter_table(
        parameter_table
    )

    # -----------------------------------------------------------------
    # Save long-form summary
    # -----------------------------------------------------------------

    summary_csv_path = (
        OUTPUT_DIR
        / "node_degree_fit_summary.csv"
    )

    summary_pickle_path = (
        OUTPUT_DIR
        / "node_degree_fit_summary.pkl"
    )

    summary.to_csv(
        summary_csv_path,
        index=False,
    )

    summary.to_pickle(
        summary_pickle_path
    )

    # -----------------------------------------------------------------
    # Save wide fitted-parameter table
    # -----------------------------------------------------------------

    parameter_csv_path = (
        OUTPUT_DIR
        / "fitted_parameter_table.csv"
    )

    parameter_pickle_path = (
        OUTPUT_DIR
        / "fitted_parameter_table.pkl"
    )

    parameter_table.to_csv(
        parameter_csv_path
    )

    parameter_table.to_pickle(
        parameter_pickle_path
    )

    # -----------------------------------------------------------------
    # Final diagnostics
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 112)
    print("OUTPUTS")
    print("=" * 112)

    print(
        f"Plots saved to:\n"
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nLong-form summary CSV:\n"
        f"  {summary_csv_path}"
    )

    print(
        f"\nLong-form summary pickle:\n"
        f"  {summary_pickle_path}"
    )

    print(
        f"\nFitted parameter table CSV:\n"
        f"  {parameter_csv_path}"
    )

    print(
        f"\nFitted parameter table pickle:\n"
        f"  {parameter_pickle_path}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()