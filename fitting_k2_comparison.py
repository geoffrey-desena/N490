#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
European node-degree CCDF analysis with amplitude anchored at k = 2
===================================================================

Loads:

    euro-comparison/european_networks.pkl

For each country:

    - retain the COMPLETE graph representation
      (parallel circuits are retained)

    - analyze:
        * one combined subtransmission network containing all
          voltage levels below 200 kV
        * each voltage level >= 200 kV separately

For every country / voltage group:

    1. Calculate the complementary cumulative degree distribution:

           P(K >= k)

    2. Discard k = 1 from the model fitting.

    3. Fit the anchored exponential model:

           P(K >= k) = A * exp(-(k - 2) / gamma)

       for k >= 2.

       Therefore:

           A = fitted P(K >= 2)

       and A is anchored at the first degree value actually used
       in the fit.

    4. Calculate R^2 and RMSE using only k >= 2.

Plots
-----
One figure per country.

All voltage groups for that country are shown on the same figure.

For every voltage group:
    - observed CCDF points
    - fitted anchored exponential curve
    - A
    - gamma
    - R^2
    - RMSE
    - number of fitted CCDF points

Plots are displayed only and are NOT saved.

Console output
--------------
A table containing:

    country
    voltage_group
    n_nodes
    n_edges
    mean_degree
    n_fit_points
    A
    gamma
    R2
    RMSE

No output files are written.
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

INPUT_FILE = (
    WORKING_DIR
    / "euro-comparison"
    / "european_networks.pkl"
)


# =====================================================================
# SETTINGS
# =====================================================================

SUBTRANSMISSION_LIMIT_KV = 200

MIN_FIT_DEGREE = 2

FIGSIZE = (10.5, 7.0)

COLORMAP = "tab10"


# =====================================================================
# ANCHORED EXPONENTIAL MODEL
# =====================================================================

def anchored_exponential_degree_distribution(
    k,
    A,
    gamma,
):
    """
    Complementary cumulative exponential model anchored at k = 2:

        P(K >= k) = A * exp(-(k - 2) / gamma)

    Therefore:

        P(K >= 2) = A

    Parameters
    ----------
    k : array-like
        Node degree.

    A : float
        Fitted amplitude at k = 2.

    gamma : float
        Exponential decay parameter.
    """

    return (
        A
        * np.exp(
            -(k - 2.0)
            / gamma
        )
    )


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degrees while retaining parallel circuits.

    Every edge row contributes once to each endpoint.

    Thus multiple parallel circuits between the same nodes contribute
    separately to node degree.
    """

    endpoints = pd.concat(
        [
            edges["node_i"],
            edges["node_j"],
        ],
        ignore_index=True,
    )

    degrees = (
        endpoints
        .value_counts()
        .sort_index()
    )

    return degrees


# =====================================================================
# COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION
# =====================================================================

def calculate_ccdf(
    degrees,
):
    """
    Calculate the complementary cumulative degree distribution:

        P(K >= k)

    for:

        k = 1, 2, ..., max_degree

    By construction:

        P(K >= 1) = 1

    The k = 1 point is retained here for plotting but is not used
    in the model fitting.
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
            for degree
            in k
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

    Both metrics are evaluated only on the data supplied to this
    function.

    In this script that means k >= 2.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

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

    using only:

        k >= 2

    Consequently:

        A = fitted P(K >= 2)

    and the model is never asked to reproduce the deterministic
    boundary condition:

        P(K >= 1) = 1.

    Returns
    -------
    dict

        A
        gamma
        r2
        rmse
        n_fit_points
        k_fit
        probability_fit
        predicted_fit
    """

    # -------------------------------------------------------------
    # Restrict both fitting and evaluation to k >= 2
    # -------------------------------------------------------------

    fit_mask = (
        k >= MIN_FIT_DEGREE
    )

    k_fit = (
        k[
            fit_mask
        ]
    )

    probability_fit = (
        probability[
            fit_mask
        ]
    )

    n_fit_points = len(
        k_fit
    )

    # -------------------------------------------------------------
    # Two free parameters require at least two points.
    #
    # With exactly two points, however, the model can often fit
    # essentially exactly. n_fit_points is therefore retained as an
    # important diagnostic.
    # -------------------------------------------------------------

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

            "k_fit":
                k_fit,

            "probability_fit":
                probability_fit,

            "predicted_fit":
                np.full_like(
                    probability_fit,
                    np.nan,
                ),
        }

    # -------------------------------------------------------------
    # Initial parameter estimates
    # -------------------------------------------------------------

    # Because A is explicitly the amplitude at k = 2, the observed
    # P(K >= 2) is a natural initial guess.
    A_initial = float(
        probability_fit[0]
    )

    gamma_initial = 2.0

    # -------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------

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

            "k_fit":
                k_fit,

            "probability_fit":
                probability_fit,

            "predicted_fit":
                np.full_like(
                    probability_fit,
                    np.nan,
                ),
        }

    # -------------------------------------------------------------
    # Predictions on the fitted domain
    # -------------------------------------------------------------

    predicted_fit = (
        anchored_exponential_degree_distribution(
            k_fit,
            A,
            gamma,
        )
    )

    # -------------------------------------------------------------
    # Metrics evaluated ONLY for k >= 2
    # -------------------------------------------------------------

    (
        r2,
        rmse,
    ) = calculate_fit_metrics(
        observed=
            probability_fit,

        predicted=
            predicted_fit,
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

        "k_fit":
            k_fit,

        "probability_fit":
            probability_fit,

        "predicted_fit":
            predicted_fit,
    }


# =====================================================================
# VOLTAGE GROUPING
# =====================================================================

def get_subtransmission_label(
    edges,
):
    """
    Build a readable label for the combined network below 200 kV.

    Examples
    --------

    110 kV

    132-165 kV

    110-150 kV
    """

    voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            < SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(voltages) == 0:

        return None

    if len(voltages) == 1:

        return (
            f"{voltages[0]:g} kV"
        )

    return (
        f"{voltages[0]:g}-"
        f"{voltages[-1]:g} kV"
    )


def build_voltage_groups(
    edges,
):
    """
    Build the edge sets analyzed for one country.

    Groups
    ------

    1. One combined network containing ALL voltages below 200 kV.

    2. Every voltage level >= 200 kV separately.

    Parallel circuits are retained throughout.
    """

    groups = []

    # -------------------------------------------------------------
    # Combined subtransmission network
    # -------------------------------------------------------------

    sub_edges = edges.loc[
        edges["voltage_kv"]
        < SUBTRANSMISSION_LIMIT_KV
    ].copy()

    if len(sub_edges) > 0:

        groups.append(
            {
                "label":
                    get_subtransmission_label(
                        edges
                    ),

                "edges":
                    sub_edges,

                "is_subtransmission":
                    True,
            }
        )

    # -------------------------------------------------------------
    # Individual transmission-voltage networks
    # -------------------------------------------------------------

    transmission_voltages = sorted(
        edges.loc[
            edges["voltage_kv"]
            >= SUBTRANSMISSION_LIMIT_KV,
            "voltage_kv",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    for voltage in transmission_voltages:

        voltage_edges = edges.loc[
            edges["voltage_kv"]
            == voltage
        ].copy()

        groups.append(
            {
                "label":
                    f"{voltage:g} kV",

                "edges":
                    voltage_edges,

                "is_subtransmission":
                    False,
            }
        )

    return groups


# =====================================================================
# ANALYZE ONE VOLTAGE GROUP
# =====================================================================

def analyze_voltage_group(
    edges,
):
    """
    Calculate node degrees, CCDF, and anchored exponential fit.
    """

    degrees = (
        calculate_node_degrees(
            edges
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
            k=k,
            probability=probability,
        )
    )

    if len(degrees) > 0:

        mean_degree = float(
            degrees.mean()
        )

    else:

        mean_degree = np.nan

    return {
        "degrees":
            degrees,

        "k":
            k,

        "probability":
            probability,

        "fit":
            fit,

        "n_nodes":
            len(
                degrees
            ),

        "n_edges":
            len(
                edges
            ),

        "mean_degree":
            mean_degree,
    }


# =====================================================================
# PLOT ONE COUNTRY
# =====================================================================

def plot_country(
    country,
    analyzed_groups,
):
    """
    Plot all voltage groups for one country on the same axes.

    Observed distributions include k = 1 so the deterministic starting
    point remains visible.

    Fitted curves begin at k = 2 because the model is defined for the
    portion of the CCDF being analyzed.
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    cmap = plt.get_cmap(
        COLORMAP
    )

    n_groups = len(
        analyzed_groups
    )

    colors = [
        cmap(
            index
            / max(
                n_groups - 1,
                1,
            )
        )
        for index
        in range(
            n_groups
        )
    ]

    annotation_lines = []

    # -------------------------------------------------------------
    # Plot each voltage group
    # -------------------------------------------------------------

    for (
        color,
        group,
    ) in zip(
        colors,
        analyzed_groups,
    ):

        label = (
            group[
                "label"
            ]
        )

        result = (
            group[
                "result"
            ]
        )

        k = (
            result[
                "k"
            ]
        )

        probability = (
            result[
                "probability"
            ]
        )

        fit = (
            result[
                "fit"
            ]
        )

        # ---------------------------------------------------------
        # Observed CCDF
        # ---------------------------------------------------------

        ax.plot(
            k,
            probability,
            "o",
            color=color,
            markersize=6,
            label=(
                f"{label} observed"
            ),
        )

        # ---------------------------------------------------------
        # Anchored exponential fit
        # ---------------------------------------------------------

        if (
            np.isfinite(
                fit["A"]
            )
            and np.isfinite(
                fit["gamma"]
            )
            and len(k) > 0
            and max(k) >= 2
        ):

            k_smooth = np.linspace(
                2,
                max(k),
                500,
            )

            p_smooth = (
                anchored_exponential_degree_distribution(
                    k_smooth,
                    fit["A"],
                    fit["gamma"],
                )
            )

            ax.plot(
                k_smooth,
                p_smooth,
                color=color,
                linewidth=2.0,
                linestyle="-",
                label=(
                    f"{label} exponential fit"
                ),
            )

        # ---------------------------------------------------------
        # Annotation
        # ---------------------------------------------------------

        annotation_lines.append(
            (
                f"{label}\n"
                f"  A={fit['A']:.4f}, "
                f"gamma={fit['gamma']:.4f}\n"
                f"  R2={fit['r2']:.4f}, "
                f"RMSE={fit['rmse']:.5f}, "
                f"n={fit['n_fit_points']}"
            )
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Node degree, k"
    )

    ax.set_ylabel(
        r"Complementary cumulative probability, "
        r"$P(K \geq k)$"
    )

    ax.set_title(
        country
    )

    max_degree = max(
        (
            int(
                max(
                    group[
                        "result"
                    ][
                        "k"
                    ]
                )
            )
            for group
            in analyzed_groups
            if len(
                group[
                    "result"
                ][
                    "k"
                ]
            ) > 0
        ),
        default=1,
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
    # Parameter annotation
    # -------------------------------------------------------------

    annotation_text = (
        "\n\n".join(
            annotation_lines
        )
    )

    ax.text(
        1.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
    )

    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------

    ax.legend(
        loc="lower left",
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()

    plt.show()


# =====================================================================
# PRINT COUNTRY-BY-COUNTRY SUMMARY
# =====================================================================

def print_summary_table(
    summary,
):
    """
    Print fitted anchored exponential parameters and fit metrics.
    """

    display_columns = [
        "country",
        "voltage_group",
        "n_nodes",
        "n_edges",
        "mean_degree",
        "n_fit_points",
        "A",
        "gamma",
        "R2",
        "RMSE",
    ]

    print("\n")
    print("=" * 135)
    print(
        "ANCHORED EXPONENTIAL CCDF FITS: "
        "P(K>=k) = A exp(-(k-2)/gamma), k>=2"
    )
    print("=" * 135)

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False,
            formatters={
                "mean_degree":
                    "{:.4f}".format,

                "A":
                    "{:.4f}".format,

                "gamma":
                    "{:.4f}".format,

                "R2":
                    "{:.4f}".format,

                "RMSE":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# PRINT AGGREGATE FIT STATISTICS
# =====================================================================

def print_aggregate_statistics(
    summary,
):
    """
    Summarize overall fit performance.

    NaN fits are excluded from the aggregate statistics.
    """

    valid = summary.dropna(
        subset=[
            "R2",
            "RMSE",
        ]
    ).copy()

    print("\n")
    print("=" * 90)
    print(
        "OVERALL ANCHORED-EXPONENTIAL FIT STATISTICS"
    )
    print("=" * 90)

    print(
        f"Valid fitted networks: "
        f"{len(valid)} / {len(summary)}"
    )

    if len(valid) == 0:

        return

    statistics = pd.DataFrame(
        {
            "metric": [
                "Mean R2",
                "Median R2",
                "Minimum R2",
                "Maximum R2",
                "Mean RMSE",
                "Median RMSE",
                "Minimum RMSE",
                "Maximum RMSE",
            ],

            "value": [
                valid[
                    "R2"
                ].mean(),

                valid[
                    "R2"
                ].median(),

                valid[
                    "R2"
                ].min(),

                valid[
                    "R2"
                ].max(),

                valid[
                    "RMSE"
                ].mean(),

                valid[
                    "RMSE"
                ].median(),

                valid[
                    "RMSE"
                ].min(),

                valid[
                    "RMSE"
                ].max(),
            ],
        }
    )

    print(
        statistics.to_string(
            index=False,
            formatters={
                "value":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# OPTIONAL DIAGNOSTIC: SMALL FITS
# =====================================================================

def print_small_fit_diagnostics(
    summary,
):
    """
    Identify networks for which very few CCDF points are available.

    A two-parameter exponential fit to only two points can reproduce
    those points exactly, so R2 and RMSE are not especially informative
    for such networks.
    """

    small = summary.loc[
        summary[
            "n_fit_points"
        ]
        <= 2
    ].copy()

    print("\n")
    print("=" * 90)
    print(
        "NETWORKS WITH TWO OR FEWER k>=2 FIT POINTS"
    )
    print("=" * 90)

    if small.empty:

        print(
            "None."
        )

        return

    print(
        small[
            [
                "country",
                "voltage_group",
                "n_nodes",
                "n_edges",
                "n_fit_points",
                "A",
                "gamma",
                "R2",
                "RMSE",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "A":
                    "{:.4f}".format,

                "gamma":
                    "{:.4f}".format,

                "R2":
                    "{:.4f}".format,

                "RMSE":
                    "{:.5f}".format,
            },
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load raw European network data
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Analyze every country
    # -----------------------------------------------------------------

    summary_rows = []

    for country in sorted(
        euro_networks
    ):

        edges = (
            euro_networks[
                country
            ]
            .copy()
        )

        print(
            f"\nAnalyzing "
            f"{country}"
        )

        voltage_groups = (
            build_voltage_groups(
                edges
            )
        )

        analyzed_groups = []

        for group in (
            voltage_groups
        ):

            result = (
                analyze_voltage_group(
                    group[
                        "edges"
                    ]
                )
            )

            analyzed_groups.append(
                {
                    "label":
                        group[
                            "label"
                        ],

                    "is_subtransmission":
                        group[
                            "is_subtransmission"
                        ],

                    "result":
                        result,
                }
            )

            fit = (
                result[
                    "fit"
                ]
            )

            summary_rows.append(
                {
                    "country":
                        country,

                    "voltage_group":
                        group[
                            "label"
                        ],

                    "is_subtransmission":
                        group[
                            "is_subtransmission"
                        ],

                    "n_nodes":
                        result[
                            "n_nodes"
                        ],

                    "n_edges":
                        result[
                            "n_edges"
                        ],

                    "mean_degree":
                        result[
                            "mean_degree"
                        ],

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

                    "R2":
                        fit[
                            "r2"
                        ],

                    "RMSE":
                        fit[
                            "rmse"
                        ],
                }
            )

        # -------------------------------------------------------------
        # One plot per country
        # -------------------------------------------------------------

        plot_country(
            country=
                country,

            analyzed_groups=
                analyzed_groups,
        )

    # -----------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------------------
    # Console output only
    # -----------------------------------------------------------------

    print_summary_table(
        summary
    )

    print_aggregate_statistics(
        summary
    )

    print_small_fit_diagnostics(
        summary
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()