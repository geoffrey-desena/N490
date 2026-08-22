#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mean node degree versus voltage for European networks
=====================================================

Loads:

    euro-comparison/european_networks.pkl

For every country, analyze every individual voltage level separately.

The analysis is performed twice:

    1. COMPLETE graph
       Parallel circuits are retained.

    2. SIMPLE graph
       Multiple edges connecting the same unordered node pair are
       collapsed to one edge.

For every country / voltage level calculate:

    <k> = mean node degree

Then, for every country independently, fit:

    <k> = slope * V + intercept

where V is voltage in kV.

For each graph representation:

    - plot all countries on one figure
    - show empirical mean-degree values
    - connect each country's empirical values
    - overlay a linear fit for each country

Console output:

    - long-form mean-degree table
    - wide summary table with one column per voltage level
    - slope
    - intercept
    - fit R^2
    - fit RMSE

Nothing is saved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

GRAPH_TYPES = [
    "complete",
    "simple",
]

FIGSIZE = (
    11.0,
    8.0,
)

MARKER_SIZE = 6

LINE_WIDTH = 1.4

FIT_LINE_WIDTH = 1.2

FIT_LINESTYLE = "--"


# =====================================================================
# SIMPLE GRAPH
# =====================================================================

def make_simple_graph(
    edges,
):
    """
    Collapse parallel edges connecting the same unordered node pair.

    IMPORTANT
    ---------
    This function is applied AFTER selecting a single voltage level.

    Therefore simplification occurs separately for each voltage network.
    """

    if edges.empty:

        return edges.copy()

    simple_edges = (
        edges
        .copy()
    )

    # -------------------------------------------------------------
    # Canonical unordered node pair
    # -------------------------------------------------------------

    simple_edges[
        "_pair_i"
    ] = (
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

    simple_edges[
        "_pair_j"
    ] = (
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

    # -------------------------------------------------------------
    # Keep one edge per unordered pair
    # -------------------------------------------------------------

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

    simple_edges[
        "node_i"
    ] = (
        simple_edges[
            "_pair_i"
        ]
    )

    simple_edges[
        "node_j"
    ] = (
        simple_edges[
            "_pair_j"
        ]
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


def prepare_graph(
    edges,
    graph_type,
):
    """
    Return the requested graph representation.
    """

    if graph_type == "complete":

        return (
            edges
            .copy()
        )

    if graph_type == "simple":

        return (
            make_simple_graph(
                edges
            )
        )

    raise ValueError(
        f"Unknown graph type: "
        f"{graph_type}"
    )


# =====================================================================
# NODE DEGREE
# =====================================================================

def calculate_node_degrees(
    edges,
):
    """
    Calculate node degree from the supplied edge representation.

    For complete graphs, parallel circuits each contribute separately.

    For simple graphs, parallel circuits have already been collapsed.
    """

    if edges.empty:

        return pd.Series(
            dtype=float
        )

    endpoints = pd.concat(
        [
            edges[
                "node_i"
            ],
            edges[
                "node_j"
            ],
        ],
        ignore_index=True,
    )

    return (
        endpoints
        .value_counts()
        .sort_index()
    )


# =====================================================================
# ANALYZE ONE VOLTAGE NETWORK
# =====================================================================

def analyze_voltage_network(
    original_edges,
    graph_type,
):
    """
    Calculate graph statistics for one country / voltage / graph type.
    """

    edges = (
        prepare_graph(
            original_edges,
            graph_type,
        )
    )

    degrees = (
        calculate_node_degrees(
            edges
        )
    )

    n_nodes = len(
        degrees
    )

    n_edges_original = len(
        original_edges
    )

    n_edges = len(
        edges
    )

    edges_removed = (
        n_edges_original
        - n_edges
    )

    if n_nodes > 0:

        mean_degree = float(
            degrees.mean()
        )

    else:

        mean_degree = np.nan

    return {
        "n_nodes":
            n_nodes,

        "n_edges_original":
            n_edges_original,

        "n_edges":
            n_edges,

        "edges_removed":
            edges_removed,

        "mean_degree":
            mean_degree,
    }


# =====================================================================
# BUILD MEAN-DEGREE TABLE
# =====================================================================

def build_mean_degree_table(
    euro_networks,
):
    """
    Analyze every country / voltage / graph representation.
    """

    rows = []

    print("\n")
    print("=" * 100)
    print(
        "CALCULATING MEAN NODE DEGREE BY VOLTAGE"
    )
    print("=" * 100)

    for country in sorted(
        euro_networks
    ):

        country_edges = (
            euro_networks[
                country
            ]
            .copy()
        )

        voltages = sorted(
            country_edges[
                "voltage_kv"
            ]
            .dropna()
            .unique()
        )

        print(
            f"\n{country}"
        )

        for graph_type in (
            GRAPH_TYPES
        ):

            for voltage in (
                voltages
            ):

                voltage_edges = (
                    country_edges.loc[
                        country_edges[
                            "voltage_kv"
                        ]
                        == voltage
                    ]
                    .copy()
                )

                result = (
                    analyze_voltage_network(
                        original_edges=
                            voltage_edges,

                        graph_type=
                            graph_type,
                    )
                )

                rows.append(
                    {
                        "country":
                            country,

                        "graph_type":
                            graph_type,

                        "voltage_kv":
                            float(
                                voltage
                            ),

                        "n_nodes":
                            result[
                                "n_nodes"
                            ],

                        "n_edges_original":
                            result[
                                "n_edges_original"
                            ],

                        "n_edges":
                            result[
                                "n_edges"
                            ],

                        "edges_removed":
                            result[
                                "edges_removed"
                            ],

                        "mean_degree":
                            result[
                                "mean_degree"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# LINEAR FIT
# =====================================================================

def fit_linear_relationship(
    x,
    y,
):
    """
    Fit:

        y = slope * x + intercept

    and calculate R^2 and RMSE.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = (
        x[
            valid
        ]
    )

    y = (
        y[
            valid
        ]
    )

    # -------------------------------------------------------------
    # At least two voltage levels are required to fit a line.
    # -------------------------------------------------------------

    if len(x) < 2:

        return {
            "slope":
                np.nan,

            "intercept":
                np.nan,

            "r2":
                np.nan,

            "rmse":
                np.nan,

            "n_points":
                len(
                    x
                ),
        }

    # -------------------------------------------------------------
    # Linear least-squares fit
    # -------------------------------------------------------------

    slope, intercept = (
        np.polyfit(
            x,
            y,
            1,
        )
    )

    predicted = (
        slope
        * x
        + intercept
    )

    # -------------------------------------------------------------
    # R^2
    # -------------------------------------------------------------

    residual_sum_squares = np.sum(
        (
            y
            - predicted
        ) ** 2
    )

    total_sum_squares = np.sum(
        (
            y
            - np.mean(
                y
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

    # -------------------------------------------------------------
    # RMSE
    # -------------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            (
                y
                - predicted
            ) ** 2
        )
    )

    return {
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

        "rmse":
            float(
                rmse
            ),

        "n_points":
            len(
                x
            ),
    }


# =====================================================================
# BUILD COUNTRY FIT TABLE
# =====================================================================

def build_country_fit_table(
    mean_degree_df,
):
    """
    Fit mean degree versus voltage separately for every country and
    graph representation.
    """

    rows = []

    for graph_type in (
        GRAPH_TYPES
    ):

        graph_df = (
            mean_degree_df.loc[
                mean_degree_df[
                    "graph_type"
                ]
                == graph_type
            ]
        )

        for country in sorted(
            graph_df[
                "country"
            ]
            .unique()
        ):

            country_df = (
                graph_df.loc[
                    graph_df[
                        "country"
                    ]
                    == country
                ]
                .sort_values(
                    "voltage_kv"
                )
            )

            fit = (
                fit_linear_relationship(
                    x=
                        country_df[
                            "voltage_kv"
                        ],

                    y=
                        country_df[
                            "mean_degree"
                        ],
                )
            )

            rows.append(
                {
                    "country":
                        country,

                    "graph_type":
                        graph_type,

                    "n_voltage_levels":
                        fit[
                            "n_points"
                        ],

                    "slope":
                        fit[
                            "slope"
                        ],

                    "intercept":
                        fit[
                            "intercept"
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
# COUNTRY COLORS
# =====================================================================

def generate_country_colors(
    countries,
):
    """
    Assign one consistent color to every country.
    """

    cmap = plt.get_cmap(
        "tab20"
    )

    return {
        country:
            cmap(
                index
                / max(
                    len(
                        countries
                    )
                    - 1,
                    1,
                )
            )

        for index, country
        in enumerate(
            countries
        )
    }


# =====================================================================
# PLOT
# =====================================================================

def plot_mean_degree_vs_voltage(
    mean_degree_df,
    graph_type,
):
    """
    Plot every country's mean degree versus voltage on one figure.

    Empirical:
        solid line + dots

    Linear fit:
        dashed line
    """

    graph_df = (
        mean_degree_df.loc[
            mean_degree_df[
                "graph_type"
            ]
            == graph_type
        ]
        .copy()
    )

    countries = sorted(
        graph_df[
            "country"
        ]
        .unique()
    )

    colors = (
        generate_country_colors(
            countries
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # -------------------------------------------------------------
    # Plot every country
    # -------------------------------------------------------------

    for country in (
        countries
    ):

        country_df = (
            graph_df.loc[
                graph_df[
                    "country"
                ]
                == country
            ]
            .dropna(
                subset=[
                    "voltage_kv",
                    "mean_degree",
                ]
            )
            .sort_values(
                "voltage_kv"
            )
        )

        if country_df.empty:

            continue

        color = (
            colors[
                country
            ]
        )

        x = (
            country_df[
                "voltage_kv"
            ]
            .to_numpy(
                dtype=float
            )
        )

        y = (
            country_df[
                "mean_degree"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ---------------------------------------------------------
        # Empirical values
        # ---------------------------------------------------------

        ax.plot(
            x,
            y,
            marker="o",
            markersize=
                MARKER_SIZE,
            linewidth=
                LINE_WIDTH,
            color=
                color,
            label=
                country,
            zorder=3,
        )

        # ---------------------------------------------------------
        # Linear fit
        # ---------------------------------------------------------

        fit = (
            fit_linear_relationship(
                x,
                y,
            )
        )

        if (
            np.isfinite(
                fit[
                    "slope"
                ]
            )
            and
            np.isfinite(
                fit[
                    "intercept"
                ]
            )
        ):

            x_fit = np.linspace(
                x.min(),
                x.max(),
                200,
            )

            y_fit = (
                fit[
                    "slope"
                ]
                * x_fit
                + fit[
                    "intercept"
                ]
            )

            ax.plot(
                x_fit,
                y_fit,
                linestyle=
                    FIT_LINESTYLE,
                linewidth=
                    FIT_LINE_WIDTH,
                color=
                    color,
                alpha=
                    0.75,
                zorder=2,
            )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Voltage (kV)",
        fontsize=13,
    )

    ax.set_ylabel(
        r"Mean node degree, "
        r"$\langle k \rangle$",
        fontsize=13,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    graph_label = (
        "Complete graph"
        if graph_type
        == "complete"
        else
        "Simple graph"
    )

    ax.set_title(
        graph_label
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    plt.show()


# =====================================================================
# LONG-FORM TABLE
# =====================================================================

def print_long_form_table(
    mean_degree_df,
    graph_type,
):
    """
    Print one row per country / voltage level.
    """

    graph_df = (
        mean_degree_df.loc[
            mean_degree_df[
                "graph_type"
            ]
            == graph_type
        ]
        .sort_values(
            [
                "country",
                "voltage_kv",
            ]
        )
    )

    print("\n")
    print("=" * 115)

    print(
        f"{graph_type.upper()} GRAPH: "
        f"MEAN NODE DEGREE BY VOLTAGE"
    )

    print("=" * 115)

    print(
        graph_df[
            [
                "country",
                "voltage_kv",
                "n_nodes",
                "n_edges_original",
                "n_edges",
                "edges_removed",
                "mean_degree",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "voltage_kv":
                    "{:.0f}".format,

                "mean_degree":
                    "{:.4f}".format,
            },
        )
    )


# =====================================================================
# WIDE SUMMARY TABLE
# =====================================================================

def build_wide_summary(
    mean_degree_df,
    fit_df,
    graph_type,
):
    """
    Build one row per country.

    Columns contain:

        <k> at each voltage level
        slope
        intercept
        R^2
        RMSE
    """

    graph_data = (
        mean_degree_df.loc[
            mean_degree_df[
                "graph_type"
            ]
            == graph_type
        ]
        .copy()
    )

    graph_fit = (
        fit_df.loc[
            fit_df[
                "graph_type"
            ]
            == graph_type
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Pivot mean degree by exact voltage
    # -------------------------------------------------------------

    degree_wide = (
        graph_data
        .pivot(
            index=
                "country",
            columns=
                "voltage_kv",
            values=
                "mean_degree",
        )
        .sort_index(
            axis=1
        )
    )

    # -------------------------------------------------------------
    # Rename voltage columns
    # -------------------------------------------------------------

    degree_wide.columns = [
        f"<k>_{voltage:g}_kV"
        for voltage
        in degree_wide.columns
    ]

    degree_wide = (
        degree_wide
        .reset_index()
    )

    # -------------------------------------------------------------
    # Add fit statistics
    # -------------------------------------------------------------

    summary = (
        degree_wide
        .merge(
            graph_fit[
                [
                    "country",
                    "n_voltage_levels",
                    "slope",
                    "intercept",
                    "fit_r2",
                    "fit_rmse",
                ]
            ],
            on="country",
            how="left",
        )
    )

    return summary


def print_wide_summary(
    summary,
    graph_type,
):
    """
    Print one-row-per-country summary.
    """

    print("\n")
    print("=" * 180)

    print(
        f"{graph_type.upper()} GRAPH: "
        f"COUNTRY LINEAR-FIT SUMMARY"
    )

    print("=" * 180)

    formatters = {}

    for column in (
        summary.columns
    ):

        if column.startswith(
            "<k>_"
        ):

            formatters[
                column
            ] = (
                "{:.4f}".format
            )

    formatters.update(
        {
            "slope":
                "{:+.6f}".format,

            "intercept":
                "{:+.4f}".format,

            "fit_r2":
                "{:.4f}".format,

            "fit_rmse":
                "{:.5f}".format,
        }
    )

    print(
        summary.to_string(
            index=False,
            formatters=
                formatters,
            na_rep="",
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -----------------------------------------------------------------
    # Load
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
    # Calculate mean degree
    # -----------------------------------------------------------------

    mean_degree_df = (
        build_mean_degree_table(
            euro_networks
        )
    )

    # -----------------------------------------------------------------
    # Fit each country's voltage dependence
    # -----------------------------------------------------------------

    fit_df = (
        build_country_fit_table(
            mean_degree_df
        )
    )

    # -----------------------------------------------------------------
    # Complete graph
    # -----------------------------------------------------------------

    print_long_form_table(
        mean_degree_df=
            mean_degree_df,

        graph_type=
            "complete",
    )

    complete_summary = (
        build_wide_summary(
            mean_degree_df=
                mean_degree_df,

            fit_df=
                fit_df,

            graph_type=
                "complete",
        )
    )

    print_wide_summary(
        summary=
            complete_summary,

        graph_type=
            "complete",
    )

    plot_mean_degree_vs_voltage(
        mean_degree_df=
            mean_degree_df,

        graph_type=
            "complete",
    )

    # -----------------------------------------------------------------
    # Simple graph
    # -----------------------------------------------------------------

    print_long_form_table(
        mean_degree_df=
            mean_degree_df,

        graph_type=
            "simple",
    )

    simple_summary = (
        build_wide_summary(
            mean_degree_df=
                mean_degree_df,

            fit_df=
                fit_df,

            graph_type=
                "simple",
        )
    )

    print_wide_summary(
        summary=
            simple_summary,

        graph_type=
            "simple",
    )

    plot_mean_degree_vs_voltage(
        mean_degree_df=
            mean_degree_df,

        graph_type=
            "simple",
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()