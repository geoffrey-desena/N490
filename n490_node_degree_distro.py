#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N490 cumulative node-degree analysis
====================================

Fit exponential decay functions to Nordic490 complementary cumulative
node-degree distributions.

For each voltage-specific AC-line network:

    - 220 kV
    - 300 kV
    - 380 kV

and for the aggregated line-only HV network:

    - 220 + 300 + 380 kV

the script:

1. Constructs nodal degrees directly from model.line endpoints.
2. Retains parallel lines as separate branches.
3. Calculates the complementary cumulative degree distribution:

       P(K >= k)

4. Fits the exponential model:

       P(K >= k) = C * exp(-k / gamma)

   with both C and gamma fitted freely.

5. Calculates R^2 and RMSE.
6. Plots the empirical cumulative distribution together with the
   fitted exponential curve.
7. Prints a summary table containing:
       network
       voltage levels
       number of nodes
       number of lines
       mean degree
       C
       gamma
       R^2
       RMSE
8. Saves the summary table and empirical distributions.

Notes
-----
Parallel lines are counted independently.

Transformer-only buses are not part of these line graphs and therefore
do not enter the degree distributions.

The aggregated network is useful as a whole-network diagnostic, but
its fitted parameters should not be interpreted as directly comparable
to the individual voltage-layer fits.
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

ANALYSIS_CASES = {
    "220 kV": [220],
    "300 kV": [300],
    "380 kV": [380],
    "Aggregated HV lines": [220, 300, 380],
}


OUTPUT_DIR = Path(
    "n490_degree_decay_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FIGSIZE = (8.5, 5.8)
DPI = 300

BASE_FONTSIZE = (
    plt.rcParams["font.size"]
    * 1.35
)


# =====================================================================
# ENDPOINT DETECTION
# =====================================================================

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus-endpoint columns in a branch table.
    """

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
            return (
                bus0_col,
                bus1_col,
            )

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# =====================================================================
# SELECT LINES
# =====================================================================

def select_lines_by_voltage(
    lines: pd.DataFrame,
    voltage_levels: list[int],
) -> pd.DataFrame:
    """
    Select model.line branches belonging to the requested voltage
    levels.
    """

    if "Vbase" not in lines.columns:

        raise ValueError(
            "model.line does not contain 'Vbase'."
        )

    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    mask = np.zeros(
        len(lines),
        dtype=bool,
    )

    for voltage_kv in voltage_levels:

        mask |= np.isclose(
            line_voltage,
            float(voltage_kv),
            equal_nan=False,
        )

    selected = lines.loc[
        mask
    ].copy()

    if selected.empty:

        raise ValueError(
            "No model.line branches found for "
            f"voltage levels {voltage_levels}."
        )

    return selected


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_nodal_degrees(
    lines: pd.DataFrame,
    voltage_levels: list[int],
) -> np.ndarray:
    """
    Calculate nodal degree for the requested N490 line network.

    Every physical line row contributes one degree to each endpoint.

    Parallel lines are therefore retained and counted independently.
    """

    selected_lines = select_lines_by_voltage(
        lines=lines,
        voltage_levels=voltage_levels,
    )

    bus0_col, bus1_col = (
        resolve_line_endpoint_columns(
            selected_lines
        )
    )

    endpoints = pd.concat(
        [
            selected_lines[bus0_col],
            selected_lines[bus1_col],
        ],
        ignore_index=True,
    ).dropna()

    degrees = (
        endpoints
        .value_counts()
        .astype(int)
        .to_numpy()
    )

    # -------------------------------------------------------------
    # Handshaking-lemma sanity check
    # -------------------------------------------------------------

    degree_sum = int(
        degrees.sum()
    )

    expected_degree_sum = (
        2 * len(selected_lines)
    )

    if degree_sum != expected_degree_sum:

        raise RuntimeError(
            "Degree check failed: "
            f"sum(k)={degree_sum}, "
            f"2E={expected_degree_sum}."
        )

    return degrees


# =====================================================================
# COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION
# =====================================================================

def calculate_degree_ccdf(
    degrees: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate the complementary cumulative degree distribution:

        P(K >= k)

    for every integer degree from 1 through the maximum observed
    degree.

    Returns
    -------
    k
        Integer degree values.

    counts
        Number of nodes with degree >= k.

    probability
        Fraction of nodes with degree >= k.

    Notes
    -----
    By construction:

        P(K >= 1) = 1
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
            np.sum(
                degrees >= degree
            )
            for degree in k
        ],
        dtype=int,
    )

    probability = (
        counts
        / len(degrees)
    )

    return (
        k,
        counts,
        probability.astype(float),
    )


# =====================================================================
# EXPONENTIAL MODEL
# =====================================================================

def exponential_ccdf(
    k: np.ndarray,
    C: float,
    gamma: float,
) -> np.ndarray:
    """
    Complementary cumulative exponential model:

        P(K >= k) = C * exp(-k / gamma)

    Both C and gamma are fitted freely.
    """

    return (
        C
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
    """
    Return R^2 and RMSE in cumulative-probability space.
    """

    residuals = (
        observed
        - predicted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (
            observed
            - np.mean(observed)
        ) ** 2
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
# FIT EXPONENTIAL
# =====================================================================

def fit_exponential_distribution(
    k: np.ndarray,
    probability: np.ndarray,
    mean_degree: float,
) -> dict:
    """
    Fit:

        P(K >= k) = C * exp(-k / gamma)

    with both C and gamma free.
    """

    if len(k) < 2:

        return {
            "C": np.nan,
            "gamma": np.nan,
            "r_squared": np.nan,
            "rmse": np.nan,
        }

    # -------------------------------------------------------------
    # Initial estimates
    # -------------------------------------------------------------

    initial_gamma = max(
        float(mean_degree),
        1e-3,
    )

    # Since P(K>=1)=1, a C somewhat above 1 is generally sensible.
    initial_C = float(
        np.exp(
            1.0 / initial_gamma
        )
    )

    try:

        popt, _ = curve_fit(
            exponential_ccdf,
            k,
            probability,
            p0=[
                initial_C,
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

        C, gamma = popt

    except (
        RuntimeError,
        ValueError,
    ):

        return {
            "C": np.nan,
            "gamma": np.nan,
            "r_squared": np.nan,
            "rmse": np.nan,
        }

    predicted = exponential_ccdf(
        k,
        C,
        gamma,
    )

    r_squared, rmse = (
        calculate_fit_statistics(
            observed=probability,
            predicted=predicted,
        )
    )

    return {
        "C": float(C),
        "gamma": float(gamma),
        "r_squared": r_squared,
        "rmse": rmse,
    }


# =====================================================================
# ANALYZE ONE NETWORK
# =====================================================================

def analyze_network(
    lines: pd.DataFrame,
    voltage_levels: list[int],
) -> dict:
    """
    Calculate degrees, cumulative distribution, exponential fit,
    and basic statistics for one N490 line network.
    """

    selected_lines = select_lines_by_voltage(
        lines=lines,
        voltage_levels=voltage_levels,
    )

    degrees = calculate_nodal_degrees(
        lines=lines,
        voltage_levels=voltage_levels,
    )

    (
        k,
        counts,
        probability,
    ) = calculate_degree_ccdf(
        degrees
    )

    mean_degree = float(
        np.mean(degrees)
    )

    fit = fit_exponential_distribution(
        k=k,
        probability=probability,
        mean_degree=mean_degree,
    )

    return {
        "degrees": degrees,
        "k": k,
        "counts": counts,
        "probability": probability,
        "n_nodes": len(degrees),
        "n_lines": len(selected_lines),
        "mean_degree": mean_degree,
        "C": fit["C"],
        "gamma": fit["gamma"],
        "r_squared": fit["r_squared"],
        "rmse": fit["rmse"],
    }


# =====================================================================
# PLOT
# =====================================================================

def plot_degree_decay(
    case_name: str,
    result: dict,
) -> None:
    """
    Plot empirical cumulative degree distribution and exponential fit.
    """

    k = result["k"]
    probability = result["probability"]

    C = result["C"]
    gamma = result["gamma"]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # -------------------------------------------------------------
    # Empirical cumulative distribution
    # -------------------------------------------------------------

    ax.plot(
        k,
        probability,
        "o",
        markersize=7,
        label="Observed cumulative distribution",
        zorder=5,
    )

    # -------------------------------------------------------------
    # Fitted curve
    # -------------------------------------------------------------

    if (
        np.isfinite(C)
        and np.isfinite(gamma)
    ):

        k_fit = np.linspace(
            float(k.min()),
            float(k.max()),
            400,
        )

        fitted_probability = (
            exponential_ccdf(
                k_fit,
                C,
                gamma,
            )
        )

        ax.plot(
            k_fit,
            fitted_probability,
            linewidth=2.2,
            label="Exponential fit",
            zorder=4,
        )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------

    ax.set_xlabel(
        "Nodal degree $k$",
        fontsize=BASE_FONTSIZE,
    )

    ax.set_ylabel(
        r"Cumulative probability $P(K \geq k)$",
        fontsize=BASE_FONTSIZE,
    )

    ax.set_title(
        f"N490 {case_name} nodal-degree distribution"
    )

    ax.set_xticks(
        np.arange(
            int(k.min()),
            int(k.max()) + 1,
        )
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.tick_params(
        axis="both",
        labelsize=BASE_FONTSIZE,
    )

    ax.grid(
        False
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.legend(
        fontsize=BASE_FONTSIZE * 0.85
    )

    # -------------------------------------------------------------
    # Fit statistics
    # -------------------------------------------------------------

    statistics_text = (
        r"$P(K\geq k)=Ce^{-k/\gamma}$"
        "\n"
        rf"$C={result['C']:.4f}$"
        "\n"
        rf"$\gamma={result['gamma']:.4f}$"
        "\n"
        rf"$R^2={result['r_squared']:.4f}$"
        "\n"
        rf"RMSE={result['rmse']:.4f}"
        "\n"
        rf"$\langle k\rangle={result['mean_degree']:.4f}$"
        "\n"
        rf"$N={result['n_nodes']}$"
        "\n"
        rf"$E={result['n_lines']}$"
    )

    ax.text(
        0.98,
        0.97,
        statistics_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=BASE_FONTSIZE * 0.82,
    )

    plt.tight_layout()

    safe_name = (
        case_name
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"N490_{safe_name}_degree_exponential_fit.png"
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved:\n  {output_path}"
    )


# =====================================================================
# PRINT SUMMARY
# =====================================================================

def print_summary(
    summary: pd.DataFrame,
) -> None:
    """
    Print a clean parameter summary.
    """

    print("\n")
    print("=" * 112)
    print(
        "N490 CUMULATIVE NODE-DEGREE EXPONENTIAL FITS"
    )
    print("=" * 112)

    display_columns = [
        "network",
        "voltage_levels",
        "n_nodes",
        "n_lines",
        "mean_degree",
        "C",
        "gamma",
        "R2",
        "RMSE",
    ]

    print(
        summary[
            display_columns
        ]
        .round(
            {
                "mean_degree": 4,
                "C": 4,
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
    # Load model
    # -----------------------------------------------------------------

    model = N490(
        year=2018
    )

    lines = model.line.copy()

    # -----------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------

    summary_rows = []
    distribution_rows = []

    for (
        case_name,
        voltage_levels,
    ) in ANALYSIS_CASES.items():

        print("\n")
        print(
            f"Analyzing {case_name}"
        )

        result = analyze_network(
            lines=lines,
            voltage_levels=voltage_levels,
        )

        # ---------------------------------------------------------
        # Summary row
        # ---------------------------------------------------------

        summary_rows.append(
            {
                "network": case_name,
                "voltage_levels": ",".join(
                    str(v)
                    for v in voltage_levels
                ),
                "n_nodes": result["n_nodes"],
                "n_lines": result["n_lines"],
                "mean_degree": result["mean_degree"],
                "C": result["C"],
                "gamma": result["gamma"],
                "R2": result["r_squared"],
                "RMSE": result["rmse"],
            }
        )

        # ---------------------------------------------------------
        # Distribution rows
        # ---------------------------------------------------------

        for (
            degree,
            count,
            probability,
        ) in zip(
            result["k"],
            result["counts"],
            result["probability"],
        ):

            distribution_rows.append(
                {
                    "network": case_name,
                    "voltage_levels": ",".join(
                        str(v)
                        for v in voltage_levels
                    ),
                    "degree": int(degree),
                    "n_nodes_degree_or_higher": int(
                        count
                    ),
                    "cumulative_probability": float(
                        probability
                    ),
                }
            )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        plot_degree_decay(
            case_name=case_name,
            result=result,
        )

    # -----------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    distributions = pd.DataFrame(
        distribution_rows
    )

    print_summary(
        summary
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    summary_csv = (
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.csv"
    )

    summary_pickle = (
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.pkl"
    )

    distributions_csv = (
        OUTPUT_DIR
        / "N490_degree_distributions.csv"
    )

    distributions_pickle = (
        OUTPUT_DIR
        / "N490_degree_distributions.pkl"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    summary.to_pickle(
        summary_pickle
    )

    distributions.to_csv(
        distributions_csv,
        index=False,
    )

    distributions.to_pickle(
        distributions_pickle
    )

    # -----------------------------------------------------------------
    # Final diagnostics
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 112)
    print("OUTPUTS")
    print("=" * 112)

    print(
        f"Summary CSV:\n"
        f"  {summary_csv}"
    )

    print(
        f"\nSummary pickle:\n"
        f"  {summary_pickle}"
    )

    print(
        f"\nCumulative distributions CSV:\n"
        f"  {distributions_csv}"
    )

    print(
        f"\nCumulative distributions pickle:\n"
        f"  {distributions_pickle}"
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()