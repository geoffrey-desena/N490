#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N490 simple-graph node-degree analysis
======================================

Calculate node-degree probability distributions and complementary
cumulative distributions for the four N490 AC voltage networks:

    - 132 kV
    - 220 kV
    - 300 kV
    - 380 kV

Each voltage layer is treated as a SIMPLE GRAPH. Parallel branches between
the same pair of buses are collapsed to a single undirected edge before
node degrees are calculated.

For each voltage network the script:

1. Selects all AC lines at the requested voltage.
2. Collapses parallel branches to a single undirected edge.
3. Calculates the node-degree probability mass function P(K = k).
4. Calculates the complementary cumulative distribution P(K >= k).
5. Fits, using ONLY k >= 2,

       P(K >= k) = A * exp(-k / gamma)

   with both A and gamma free.
6. Calculates R^2 and RMSE over the fitted points (k >= 2).
7. Produces one combined CCDF figure for all four voltage networks.
8. Saves:
       - fit-summary CSV and pickle
       - empirical + fitted CCDF values CSV and pickle
       - node-degree probability distributions CSV and pickle
       - combined CCDF figure

Transformer-only buses are not part of these line graphs and therefore do
not enter the degree distributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from nordic490 import N490


# =====================================================================
# CONFIGURATION
# =====================================================================

VOLTAGE_LEVELS = [
    132,
    220,
    300,
    380,
]

OUTPUT_DIR = Path(
    "n490_node_degree_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGSIZE = (10, 6)
DPI = 300

# Single text-size control for the figure.
TEXT_SIZE = 16

MARKERS = {
    132: "D",
    220: "o",
    300: "s",
    380: "^",
}

COLORS = {
    132: "#434941",
    220: "#679805",
    300: "#d8d11c",
    380: "#c92931",
}


# =====================================================================
# ENDPOINT DETECTION
# =====================================================================

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """Identify the two bus-endpoint columns in a branch table."""

    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for bus0_col, bus1_col in candidate_pairs:
        if (
            bus0_col in lines.columns
            and bus1_col in lines.columns
        ):
            return bus0_col, bus1_col

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available columns:\n{lines.columns.tolist()}"
    )


# =====================================================================
# SELECT VOLTAGE LAYER
# =====================================================================

def select_lines_by_voltage(
    lines: pd.DataFrame,
    voltage_kv: int,
) -> pd.DataFrame:
    """Select N490 AC lines belonging to one voltage layer."""

    if "Vbase" not in lines.columns:
        raise ValueError(
            "model.line does not contain 'Vbase'."
        )

    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    mask = np.isclose(
        line_voltage,
        float(voltage_kv),
        equal_nan=False,
    )

    selected = lines.loc[mask].copy()

    if selected.empty:
        raise ValueError(
            f"No model.line branches found at {voltage_kv} kV."
        )

    return selected


# =====================================================================
# SIMPLE-GRAPH CONSTRUCTION
# =====================================================================

def build_simple_graph_edges(
    selected_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse parallel branches to one undirected edge per bus pair.

    Endpoint order is canonicalized so that (i, j) and (j, i) are treated
    as the same graph edge.
    """

    bus0_col, bus1_col = resolve_line_endpoint_columns(
        selected_lines
    )

    edges = selected_lines[
        [bus0_col, bus1_col]
    ].dropna().copy()

    # Canonical undirected endpoint ordering. Using string keys keeps this
    # robust to either numeric or string-like N490 bus identifiers.
    bus0_key = edges[bus0_col].astype(str)
    bus1_key = edges[bus1_col].astype(str)

    swap_mask = bus0_key > bus1_key

    edge_u = edges[bus0_col].copy()
    edge_v = edges[bus1_col].copy()

    edge_u.loc[swap_mask] = edges.loc[swap_mask, bus1_col]
    edge_v.loc[swap_mask] = edges.loc[swap_mask, bus0_col]

    simple_edges = pd.DataFrame(
        {
            "bus0": edge_u.to_numpy(),
            "bus1": edge_v.to_numpy(),
        }
    )

    # Remove self-loops if any exist; they are not meaningful for this
    # simple-graph degree analysis.
    simple_edges = simple_edges.loc[
        simple_edges["bus0"].astype(str)
        != simple_edges["bus1"].astype(str)
    ].copy()

    simple_edges = (
        simple_edges
        .drop_duplicates(
            subset=["bus0", "bus1"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return simple_edges


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_nodal_degrees(
    simple_edges: pd.DataFrame,
) -> np.ndarray:
    """Calculate node degrees from a simple undirected edge table."""

    endpoints = pd.concat(
        [
            simple_edges["bus0"],
            simple_edges["bus1"],
        ],
        ignore_index=True,
    ).dropna()

    degrees = (
        endpoints
        .value_counts()
        .astype(int)
        .to_numpy()
    )

    degree_sum = int(
        degrees.sum()
    )

    expected_degree_sum = (
        2 * len(simple_edges)
    )

    if degree_sum != expected_degree_sum:
        raise RuntimeError(
            "Degree check failed for simple graph: "
            f"sum(k)={degree_sum}, 2E={expected_degree_sum}."
        )

    return degrees


# =====================================================================
# DEGREE PROBABILITY DISTRIBUTION
# =====================================================================

def calculate_degree_probability_distribution(
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ordinary node-degree probability distribution P(K = k).
    """

    if len(degrees) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    max_degree = int(
        np.max(degrees)
    )

    k = np.arange(
        1,
        max_degree + 1,
        dtype=float,
    )

    counts = np.array(
        [
            np.sum(degrees == degree)
            for degree in k
        ],
        dtype=int,
    )

    probability = (
        counts / len(degrees)
    )

    return (
        k,
        counts,
        probability.astype(float),
    )


# =====================================================================
# COMPLEMENTARY CUMULATIVE DISTRIBUTION
# =====================================================================

def calculate_degree_ccdf(
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the complementary cumulative distribution P(K >= k)."""

    if len(degrees) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    max_degree = int(
        np.max(degrees)
    )

    k = np.arange(
        1,
        max_degree + 1,
        dtype=float,
    )

    counts = np.array(
        [
            np.sum(degrees >= degree)
            for degree in k
        ],
        dtype=int,
    )

    probability = (
        counts / len(degrees)
    )

    return (
        k,
        counts,
        probability.astype(float),
    )


# =====================================================================
# EXPONENTIAL CCDF MODEL
# =====================================================================

def exponential_ccdf(
    k: np.ndarray,
    A: float,
    gamma: float,
) -> np.ndarray:
    """
    Complementary cumulative exponential model:

        P(K >= k) = A * exp(-k / gamma)
    """

    return (
        A
        * np.exp(
            -k / gamma
        )
    )


# =====================================================================
# GOODNESS OF FIT
# =====================================================================

def calculate_fit_statistics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """Return R^2 and RMSE in cumulative-probability space."""

    residuals = (
        observed
        - predicted
    )

    ss_res = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                observed
                - np.mean(observed)
            ) ** 2
        )
    )

    if ss_tot > 0:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return (
        float(r_squared),
        rmse,
    )


# =====================================================================
# FIT EXPONENTIAL TO k >= 2 ONLY
# =====================================================================

def fit_exponential_distribution(
    k: np.ndarray,
    probability: np.ndarray,
    mean_degree: float,
) -> dict:
    """
    Fit the Hartmann-style free-parameter exponential model:

        P(K >= k) = A * exp(-k / gamma)

    using ONLY observations with k >= 2.

    Both A and gamma are fitted freely. R^2 and RMSE are also calculated
    only over the fitted k >= 2 observations.
    """

    fit_mask = (
        k >= 2
    )

    k_fit = k[fit_mask]
    probability_fit = probability[fit_mask]

    if len(k_fit) < 2:
        return {
            "A": np.nan,
            "gamma": np.nan,
            "r_squared": np.nan,
            "rmse": np.nan,
            "n_fit_points": len(k_fit),
        }

    initial_gamma = max(
        float(mean_degree),
        1e-3,
    )

    # Choose an initial A so the initial exponential passes approximately
    # through the first fitted observation at k=2.
    initial_A = float(
        probability_fit[0]
        * np.exp(
            k_fit[0] / initial_gamma
        )
    )

    try:
        popt, _ = curve_fit(
            exponential_ccdf,
            k_fit,
            probability_fit,
            p0=[
                initial_A,
                initial_gamma,
            ],
            bounds=(
                [
                    0.0,
                    1e-12,
                ],
                [
                    np.inf,
                    np.inf,
                ],
            ),
            maxfev=100000,
        )

        A, gamma = popt

    except (
        RuntimeError,
        ValueError,
    ):
        return {
            "A": np.nan,
            "gamma": np.nan,
            "r_squared": np.nan,
            "rmse": np.nan,
            "n_fit_points": len(k_fit),
        }

    predicted = exponential_ccdf(
        k_fit,
        A,
        gamma,
    )

    r_squared, rmse = calculate_fit_statistics(
        observed=probability_fit,
        predicted=predicted,
    )

    return {
        "A": float(A),
        "gamma": float(gamma),
        "r_squared": r_squared,
        "rmse": rmse,
        "n_fit_points": len(k_fit),
    }


# =====================================================================
# ANALYZE ONE VOLTAGE NETWORK
# =====================================================================

def analyze_network(
    lines: pd.DataFrame,
    voltage_kv: int,
) -> dict:
    """Analyze one voltage-specific N490 simple graph."""

    selected_lines = select_lines_by_voltage(
        lines=lines,
        voltage_kv=voltage_kv,
    )

    simple_edges = build_simple_graph_edges(
        selected_lines
    )

    degrees = calculate_nodal_degrees(
        simple_edges
    )

    (
        pmf_k,
        pmf_counts,
        pmf_probability,
    ) = calculate_degree_probability_distribution(
        degrees
    )

    (
        ccdf_k,
        ccdf_counts,
        ccdf_probability,
    ) = calculate_degree_ccdf(
        degrees
    )

    mean_degree = float(
        np.mean(degrees)
    )

    fit = fit_exponential_distribution(
        k=ccdf_k,
        probability=ccdf_probability,
        mean_degree=mean_degree,
    )

    n_original_lines = len(
        selected_lines
    )

    n_simple_edges = len(
        simple_edges
    )

    return {
        "voltage_kv": voltage_kv,
        "degrees": degrees,
        "pmf_k": pmf_k,
        "pmf_counts": pmf_counts,
        "pmf_probability": pmf_probability,
        "ccdf_k": ccdf_k,
        "ccdf_counts": ccdf_counts,
        "ccdf_probability": ccdf_probability,
        "n_nodes": len(degrees),
        "n_original_lines": n_original_lines,
        "n_simple_edges": n_simple_edges,
        "n_parallel_removed": (
            n_original_lines
            - n_simple_edges
        ),
        "mean_degree": mean_degree,
        "A": fit["A"],
        "gamma": fit["gamma"],
        "r_squared": fit["r_squared"],
        "rmse": fit["rmse"],
        "n_fit_points": fit["n_fit_points"],
    }


# =====================================================================
# COMBINED CCDF PLOT
# =====================================================================

def plot_degree_ccdfs(
    results_by_voltage: dict[int, dict],
) -> Path:
    """
    Plot all empirical CCDFs and their k>=2 free-A/free-gamma exponential
    fits using the same visual style as plot_overlap_statistics().
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    fit_text_lines = [
        r"$P(K\geq k)=A e^{-k/\gamma}$",
        r"fit over $k\geq2$",
        "",
    ]

    for voltage_kv in VOLTAGE_LEVELS:
        result = results_by_voltage[
            voltage_kv
        ]

        k = result["ccdf_k"]
        probability = result[
            "ccdf_probability"
        ]

        color = COLORS.get(
            voltage_kv,
            "#000000",
        )

        marker = MARKERS.get(
            voltage_kv,
            "o",
        )

        # ---------------------------------------------------------
        # Empirical CCDF
        # ---------------------------------------------------------

        ax.plot(
            k,
            probability,
            marker=marker,
            color=color,
            markersize=9,
            linestyle="None",
            linewidth=0,
            label=f"{voltage_kv} kV",
            zorder=5,
        )

        # ---------------------------------------------------------
        # Exponential fit
        # ---------------------------------------------------------

        A = result["A"]
        gamma = result["gamma"]

        if (
            np.isfinite(A)
            and np.isfinite(gamma)
        ):
            k_fit = np.linspace(
                2.0,
                float(k.max()),
                300,
            )

            fitted_probability = exponential_ccdf(
                k_fit,
                A,
                gamma,
            )

            ax.plot(
                k_fit,
                fitted_probability,
                linestyle="--",
                linewidth=2.2,
                color=color,
                label="_nolegend_",
                zorder=4,
            )

        fit_text_lines.append(
            (
                rf"{voltage_kv} kV: "
                rf"$A={result['A']:.3f}$, "
                rf"$\gamma={result['gamma']:.3f}$, "
                rf"$R^2={result['r_squared']:.3f}$"
            )
        )

    # -------------------------------------------------------------
    # Axes formatting
    # -------------------------------------------------------------

    max_degree = max(
        int(
            results_by_voltage[v][
                "ccdf_k"
            ].max()
        )
        for v in VOLTAGE_LEVELS
    )

    ax.set_xticks(
        np.arange(
            1,
            max_degree + 1,
        )
    )

    ax.set_xlabel(
        "Nodal degree $k$",
        fontsize=TEXT_SIZE,
        color="#000000",
    )

    ax.set_ylabel(
        r"Complementary cumulative probability $P(K \geq k)$",
        fontsize=TEXT_SIZE,
        color="#000000",
    )

    ax.set_ylim(
        bottom=0
    )

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=TEXT_SIZE,
        colors="#000000",
    )

    ax.grid(False)

    # -------------------------------------------------------------
    # Axis spines
    # -------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_color(
        "#000000"
    )
    ax.spines["left"].set_color(
        "#000000"
    )

    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------

    legend = ax.legend(
        fontsize=TEXT_SIZE,
        frameon=False,
        loc="lower left"
    )

    for text in legend.get_texts():
        text.set_color(
            "#000000"
        )

    # -------------------------------------------------------------
    # Fit diagnostics block
    # -------------------------------------------------------------

    fit_text = "\n".join(
        fit_text_lines
    )

    ax.text(
        0.45,
        0.98,
        fit_text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=TEXT_SIZE,
        color="#000000",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_ccdf_exponential_fits.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved figure:\n  {output_path}"
    )

    return output_path


# =====================================================================
# PRINT SUMMARY
# =====================================================================

def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print a clean parameter summary."""

    print("\n")
    print("=" * 125)
    print(
        "N490 SIMPLE-GRAPH NODE-DEGREE EXPONENTIAL FITS (FIT OVER k >= 2)"
    )
    print("=" * 125)

    display_columns = [
        "voltage_kv",
        "n_nodes",
        "n_original_lines",
        "n_simple_edges",
        "n_parallel_removed",
        "mean_degree",
        "A",
        "gamma",
        "R2",
        "RMSE",
        "n_fit_points",
    ]

    print(
        summary[
            display_columns
        ]
        .round(
            {
                "mean_degree": 4,
                "A": 4,
                "gamma": 4,
                "R2": 4,
                "RMSE": 4,
            }
        )
        .to_string(
            index=False
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    # -----------------------------------------------------------------
    # Load N490
    # -----------------------------------------------------------------

    model = N490(
        year=2018
    )

    lines = model.line.copy()

    # -----------------------------------------------------------------
    # Run voltage-specific analyses
    # -----------------------------------------------------------------

    results_by_voltage = {}

    summary_rows = []
    ccdf_rows = []
    pmf_rows = []

    for voltage_kv in VOLTAGE_LEVELS:

        print("\n")
        print(
            f"Analyzing {voltage_kv} kV simple graph"
        )

        result = analyze_network(
            lines=lines,
            voltage_kv=voltage_kv,
        )

        results_by_voltage[
            voltage_kv
        ] = result

        # ---------------------------------------------------------
        # Fit-summary row
        # ---------------------------------------------------------

        summary_rows.append(
            {
                "voltage_kv": voltage_kv,
                "n_nodes": result["n_nodes"],
                "n_original_lines": result[
                    "n_original_lines"
                ],
                "n_simple_edges": result[
                    "n_simple_edges"
                ],
                "n_parallel_removed": result[
                    "n_parallel_removed"
                ],
                "mean_degree": result[
                    "mean_degree"
                ],
                "A": result["A"],
                "gamma": result["gamma"],
                "R2": result[
                    "r_squared"
                ],
                "RMSE": result["rmse"],
                "n_fit_points": result[
                    "n_fit_points"
                ],
            }
        )

        # ---------------------------------------------------------
        # CCDF rows: empirical plus fitted value at every integer k
        # ---------------------------------------------------------

        for (
            degree,
            count,
            probability,
        ) in zip(
            result["ccdf_k"],
            result["ccdf_counts"],
            result["ccdf_probability"],
        ):

            if (
                degree >= 2
                and np.isfinite(result["A"])
                and np.isfinite(result["gamma"])
            ):
                fitted_probability = float(
                    exponential_ccdf(
                        np.array([degree]),
                        result["A"],
                        result["gamma"],
                    )[0]
                )
            else:
                fitted_probability = np.nan

            ccdf_rows.append(
                {
                    "voltage_kv": voltage_kv,
                    "degree": int(degree),
                    "n_nodes_degree_or_higher": int(
                        count
                    ),
                    "cumulative_probability": float(
                        probability
                    ),
                    "used_in_fit": bool(
                        degree >= 2
                    ),
                    "fitted_cumulative_probability": (
                        fitted_probability
                    ),
                }
            )

        # ---------------------------------------------------------
        # Ordinary node-degree probability distribution P(K = k)
        # ---------------------------------------------------------

        for (
            degree,
            count,
            probability,
        ) in zip(
            result["pmf_k"],
            result["pmf_counts"],
            result["pmf_probability"],
        ):
            pmf_rows.append(
                {
                    "voltage_kv": voltage_kv,
                    "degree": int(degree),
                    "n_nodes_with_degree": int(
                        count
                    ),
                    "probability": float(
                        probability
                    ),
                }
            )

    # -----------------------------------------------------------------
    # Build tables
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    ccdf_distributions = pd.DataFrame(
        ccdf_rows
    )

    probability_distributions = pd.DataFrame(
        pmf_rows
    )

    print_summary(
        summary
    )

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    plot_degree_ccdfs(
        results_by_voltage
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    summary_csv = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_fit_summary.csv"
    )

    summary_pickle = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_fit_summary.pkl"
    )

    ccdf_csv = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_ccdf.csv"
    )

    ccdf_pickle = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_ccdf.pkl"
    )

    probability_csv = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_probability_distributions.csv"
    )

    probability_pickle = (
        OUTPUT_DIR
        / "N490_simple_graph_degree_probability_distributions.pkl"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    summary.to_pickle(
        summary_pickle
    )

    ccdf_distributions.to_csv(
        ccdf_csv,
        index=False,
    )

    ccdf_distributions.to_pickle(
        ccdf_pickle
    )

    probability_distributions.to_csv(
        probability_csv,
        index=False,
    )

    probability_distributions.to_pickle(
        probability_pickle
    )

    # -----------------------------------------------------------------
    # Final diagnostics
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 125)
    print("OUTPUTS")
    print("=" * 125)

    print(
        f"Fit summary CSV:\n"
        f"  {summary_csv}"
    )

    print(
        f"\nFit summary pickle:\n"
        f"  {summary_pickle}"
    )

    print(
        f"\nCCDF values CSV:\n"
        f"  {ccdf_csv}"
    )

    print(
        f"\nCCDF values pickle:\n"
        f"  {ccdf_pickle}"
    )

    print(
        f"\nProbability distributions CSV:\n"
        f"  {probability_csv}"
    )

    print(
        f"\nProbability distributions pickle:\n"
        f"  {probability_pickle}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
