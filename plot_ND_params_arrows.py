#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Investigate population correlation with subtransmission A and gamma
===================================================================

Loads:

    euro-comparison/european_networks.pkl

For each country:

    - aggregate all voltage levels below 200 kV
    - convert to a SIMPLE graph
    - calculate the CCDF
    - fit, for k >= 2:

          P(K >= k) = A * exp(-(k - 2) / gamma)

Then investigate whether country population is associated with:

    A
    gamma

Outputs
-------
Plots:
    1. A vs population
    2. gamma vs population

Each plot includes:
    - country points
    - linear least-squares fit
    - line equation
    - Pearson r
    - Spearman rho
    - R^2

Console:
    - fitted subtransmission parameters
    - correlation summary

Nothing is saved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

INPUT_FILE = (
    WORKING_DIR
    / "euro-comparison"
    / "european_networks.pkl"
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200.0

MIN_FIT_DEGREE = 2

FIGSIZE = (8.5, 6.5)

# Use millions on the x-axis for readability.
POPULATION_SCALE = 1_000_000.0


# =====================================================================
# COUNTRY POPULATIONS
# =====================================================================

# Same 2024 population values used in the previous demographic analysis.

COUNTRY_POPULATION = {

    "Albania":
        2_377_128,

    "Belgium":
        11_858_610,

    "Bosnia&Herzegovina":
        3_164_253,

    "Croatia":
        3_866_000,

    "Czechia":
        10_672_118,

    "Denmark":
        5_903_037,

    "Estonia":
        1_348_840,

    "Hungary":
        9_605_074,

    "Ireland":
        5_396_000,

    "Latvia":
        1_879_383,

    "Lithuania":
        2_831_639,

    "Netherlands":
        17_700_982,

    "Portugal":
        10_434_332,

    "Slovakia":
        5_422_000,

    "Slovenia":
        2_112_076,
}


# =====================================================================
# ANCHORED EXPONENTIAL MODEL
# =====================================================================

def anchored_exponential_degree_distribution(
    k,
    A,
    gamma,
):
    """
    Anchored exponential CCDF:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    so:

        P(K >= 2) = A
    """

    return (
        A
        * np.exp(
            -(k - 2.0)
            / gamma
        )
    )


# =====================================================================
# SIMPLE GRAPH
# =====================================================================

def make_simple_graph(
    edges,
):
    """
    Collapse parallel branches connecting the same unordered node pair.
    """

    if edges.empty:

        return edges.copy()

    simple_edges = edges.copy()

    simple_edges["_pair_i"] = (
        simple_edges[
            [
                "node_i",
                "node_j",
            ]
        ]
        .min(
            axis=1
        )
    )

    simple_edges["_pair_j"] = (
        simple_edges[
            [
                "node_i",
                "node_j",
            ]
        ]
        .max(
            axis=1
        )
    )

    simple_edges = (
        simple_edges
        .drop_duplicates(
            subset=[
                "_pair_i",
                "_pair_j",
            ],
            keep="first",
        )
        .copy()
    )

    simple_edges["node_i"] = (
        simple_edges["_pair_i"]
    )

    simple_edges["node_j"] = (
        simple_edges["_pair_j"]
    )

    simple_edges = (
        simple_edges
        .drop(
            columns=[
                "_pair_i",
                "_pair_j",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return simple_edges


# =====================================================================
# NODE DEGREE
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degree.
    """

    if edges.empty:

        return pd.Series(
            dtype=float
        )

    endpoints = pd.concat(
        [
            edges["node_i"],
            edges["node_j"],
        ],
        ignore_index=True,
    )

    return (
        endpoints
        .value_counts()
        .sort_index()
    )


# =====================================================================
# CCDF
# =====================================================================

def calculate_ccdf(
    degrees,
):
    """
    Calculate:

        P(K >= k)
    """

    if len(degrees) == 0:

        return (
            np.array([]),
            np.array([]),
        )

    max_degree = int(
        degrees.max()
    )

    k = np.arange(
        1,
        max_degree + 1,
        dtype=float,
    )

    probability = np.array(
        [
            np.mean(
                degrees >= degree
            )
            for degree in k
        ],
        dtype=float,
    )

    return (
        k,
        probability,
    )


# =====================================================================
# FIT METRICS
# =====================================================================

def calculate_fit_metrics(
    observed,
    predicted,
):
    """
    Calculate R^2 and RMSE.
    """

    residuals = (
        observed
        - predicted
    )

    residual_sum_squares = np.sum(
        residuals ** 2
    )

    total_sum_squares = np.sum(
        (
            observed
            - np.mean(
                observed
            )
        ) ** 2
    )

    if total_sum_squares > 0:

        r2 = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    return (
        float(r2),
        float(rmse),
    )


# =====================================================================
# ANCHORED EXPONENTIAL FIT
# =====================================================================

def fit_anchored_exponential(
    k,
    probability,
):
    """
    Fit:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    using only k >= 2.
    """

    mask = (
        k >= MIN_FIT_DEGREE
    )

    k_fit = (
        k[
            mask
        ]
    )

    probability_fit = (
        probability[
            mask
        ]
    )

    n_fit_points = len(
        k_fit
    )

    if n_fit_points < 2:

        return {
            "A":
                np.nan,

            "gamma":
                np.nan,

            "r2":
                np.nan,

            "rmse":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    A_initial = float(
        probability_fit[0]
    )

    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            anchored_exponential_degree_distribution,
            k_fit,
            probability_fit,
            p0=[
                A_initial,
                gamma_initial,
            ],
            bounds=(
                [
                    0.0,
                    1e-8,
                ],
                [
                    1.0,
                    np.inf,
                ],
            ),
            maxfev=100000,
        )

        A = float(
            popt[0]
        )

        gamma = float(
            popt[1]
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return {
            "A":
                np.nan,

            "gamma":
                np.nan,

            "r2":
                np.nan,

            "rmse":
                np.nan,

            "n_fit_points":
                n_fit_points,
        }

    predicted = (
        anchored_exponential_degree_distribution(
            k_fit,
            A,
            gamma,
        )
    )

    (
        r2,
        rmse,
    ) = calculate_fit_metrics(
        probability_fit,
        predicted,
    )

    return {
        "A":
            A,

        "gamma":
            gamma,

        "r2":
            r2,

        "rmse":
            rmse,

        "n_fit_points":
            n_fit_points,
    }


# =====================================================================
# BUILD SUBTRANSMISSION PARAMETER TABLE
# =====================================================================

def build_subtransmission_table(
    euro_networks,
):
    """
    Build one aggregated <200 kV network per country and fit A/gamma.
    """

    rows = []

    for country in sorted(
        euro_networks
    ):

        if country not in COUNTRY_POPULATION:

            raise KeyError(
                f"No population value defined for {country}"
            )

        edges = (
            euro_networks[
                country
            ]
            .copy()
        )

        # ---------------------------------------------------------
        # Aggregate ALL <200 kV edges
        # ---------------------------------------------------------

        sub_edges = edges.loc[
            edges["voltage_kv"]
            < SUBTRANSMISSION_LIMIT_KV
        ].copy()

        if sub_edges.empty:

            continue

        # ---------------------------------------------------------
        # Simplify after aggregation
        # ---------------------------------------------------------

        simple_edges = (
            make_simple_graph(
                sub_edges
            )
        )

        degrees = (
            calculate_node_degrees(
                simple_edges
            )
        )

        (
            k,
            probability,
        ) = calculate_ccdf(
            degrees
        )

        fit = (
            fit_anchored_exponential(
                k,
                probability,
            )
        )

        rows.append(
            {
                "country":
                    country,

                "population":
                    COUNTRY_POPULATION[
                        country
                    ],

                "population_millions":
                    COUNTRY_POPULATION[
                        country
                    ]
                    / POPULATION_SCALE,

                "n_nodes":
                    len(
                        degrees
                    ),

                "n_edges":
                    len(
                        simple_edges
                    ),

                "n_fit_points":
                    fit[
                        "n_fit_points"
                    ],

                "A":
                    fit[
                        "A"
                    ],

                "gamma":
                    fit[
                        "gamma"
                    ],

                "fit_r2":
                    fit[
                        "r2"
                    ],

                "fit_rmse":
                    fit[
                        "rmse"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# CORRELATION ANALYSIS
# =====================================================================

def calculate_correlation_statistics(
    dataframe,
    parameter,
):
    """
    Calculate linear fit, Pearson r, Spearman rho, and R^2.
    """

    data = (
        dataframe
        .dropna(
            subset=[
                "population_millions",
                parameter,
            ]
        )
        .copy()
    )

    x = (
        data[
            "population_millions"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        data[
            parameter
        ]
        .to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = (
        slope
        * x
        + intercept
    )

    residual_ss = np.sum(
        (
            y
            - predicted
        ) ** 2
    )

    total_ss = np.sum(
        (
            y
            - np.mean(
                y
            )
        ) ** 2
    )

    if total_ss > 0:

        r2 = (
            1.0
            - residual_ss
            / total_ss
        )

    else:

        r2 = np.nan

    # -------------------------------------------------------------
    # Pearson and Spearman
    # -------------------------------------------------------------

    pearson_r, pearson_p = (
        pearsonr(
            x,
            y,
        )
    )

    spearman_rho, spearman_p = (
        spearmanr(
            x,
            y,
        )
    )

    return {
        "parameter":
            parameter,

        "n":
            len(
                data
            ),

        "slope":
            float(
                slope
            ),

        "intercept":
            float(
                intercept
            ),

        "r2":
            float(
                r2
            ),

        "pearson_r":
            float(
                pearson_r
            ),

        "pearson_p":
            float(
                pearson_p
            ),

        "spearman_rho":
            float(
                spearman_rho
            ),

        "spearman_p":
            float(
                spearman_p
            ),
    }


# =====================================================================
# PLOT POPULATION VS PARAMETER
# =====================================================================

def plot_population_vs_parameter(
    dataframe,
    parameter,
    ylabel,
    stats,
):
    """
    Plot one subtransmission parameter against population.
    """

    data = (
        dataframe
        .dropna(
            subset=[
                "population_millions",
                parameter,
            ]
        )
        .copy()
    )

    x = (
        data[
            "population_millions"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        data[
            parameter
        ]
        .to_numpy(
            dtype=float
        )
    )

    # -------------------------------------------------------------
    # Smooth fitted line
    # -------------------------------------------------------------

    x_line = np.linspace(
        x.min(),
        x.max(),
        300,
    )

    y_line = (
        stats[
            "slope"
        ]
        * x_line
        + stats[
            "intercept"
        ]
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.scatter(
        x,
        y,
        s=85,
        color="black",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.80,
    )

    ax.plot(
        x_line,
        y_line,
        linewidth=2.0,
        linestyle="-",
    )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Population (millions)",
        fontsize=13,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=13,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    # -------------------------------------------------------------
    # Annotation
    # -------------------------------------------------------------

    fit_text = (
        f"y = "
        f"{stats['slope']:+.4f}x "
        f"{stats['intercept']:+.4f}\n"
        f"$R^2$ = {stats['r2']:.4f}\n"
        f"Pearson r = {stats['pearson_r']:.4f}\n"
        f"Pearson p = {stats['pearson_p']:.4f}\n"
        f"Spearman $\\rho$ = {stats['spearman_rho']:.4f}\n"
        f"Spearman p = {stats['spearman_p']:.4f}"
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
            "boxstyle":
                "round",

            "facecolor":
                "white",

            "alpha":
                0.85,
        },
    )

    fig.tight_layout()

    plt.show()


# =====================================================================
# PRINT SUBTRANSMISSION TABLE
# =====================================================================

def print_parameter_table(
    dataframe,
):
    """
    Print population and fitted parameters.
    """

    print("\n")
    print("=" * 120)
    print(
        "SUBTRANSMISSION PARAMETERS AND POPULATION"
    )
    print("=" * 120)

    print(
        dataframe[
            [
                "country",
                "population",
                "n_nodes",
                "n_edges",
                "A",
                "gamma",
                "fit_r2",
                "fit_rmse",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "population":
                    "{:,.0f}".format,

                "A":
                    "{:.4f}".format,

                "gamma":
                    "{:.4f}".format,

                "fit_r2":
                    "{:.4f}".format,

                "fit_rmse":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# PRINT CORRELATION SUMMARY
# =====================================================================

def print_correlation_summary(
    statistics,
):
    """
    Print compact statistical summary.
    """

    table = pd.DataFrame(
        statistics
    )

    print("\n")
    print("=" * 115)
    print(
        "POPULATION CORRELATION WITH "
        "SUBTRANSMISSION PARAMETERS"
    )
    print("=" * 115)

    print(
        table[
            [
                "parameter",
                "n",
                "slope",
                "intercept",
                "r2",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "slope":
                    "{:+.5f}".format,

                "intercept":
                    "{:+.5f}".format,

                "r2":
                    "{:.5f}".format,

                "pearson_r":
                    "{:+.5f}".format,

                "pearson_p":
                    "{:.5f}".format,

                "spearman_rho":
                    "{:+.5f}".format,

                "spearman_p":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 100)
    print(
        "LOADING EUROPEAN NETWORK DATA"
    )
    print("=" * 100)

    print(
        f"Input:\n"
        f"  {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"  {INPUT_FILE}"
        )

    euro_networks = (
        pd.read_pickle(
            INPUT_FILE
        )
    )

    print(
        f"\nCountries loaded: "
        f"{len(euro_networks)}"
    )

    # -------------------------------------------------------------
    # Fit aggregated subtransmission networks
    # -------------------------------------------------------------

    parameter_df = (
        build_subtransmission_table(
            euro_networks
        )
    )

    print_parameter_table(
        parameter_df
    )

    # -------------------------------------------------------------
    # Correlations
    # -------------------------------------------------------------

    A_statistics = (
        calculate_correlation_statistics(
            dataframe=
                parameter_df,

            parameter=
                "A",
        )
    )

    gamma_statistics = (
        calculate_correlation_statistics(
            dataframe=
                parameter_df,

            parameter=
                "gamma",
        )
    )

    statistics = [
        A_statistics,
        gamma_statistics,
    ]

    print_correlation_summary(
        statistics
    )

    # -------------------------------------------------------------
    # Population vs A
    # -------------------------------------------------------------

    plot_population_vs_parameter(
        dataframe=
            parameter_df,

        parameter=
            "A",

        ylabel=
            r"$A$",

        stats=
            A_statistics,
    )

    # -------------------------------------------------------------
    # Population vs gamma
    # -------------------------------------------------------------

    plot_population_vs_parameter(
        dataframe=
            parameter_df,

        parameter=
            "gamma",

        ylabel=
            r"$\gamma$",

        stats=
            gamma_statistics,
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
    