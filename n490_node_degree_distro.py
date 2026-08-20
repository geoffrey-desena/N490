#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Fit exponential decay functions to Nordic490 nodal-degree distributions.

For each voltage-specific AC-line network (220, 300, and 380 kV):

1. Construct nodal degrees directly from model.line endpoints.
2. Calculate the empirical degree probability distribution P(k).
3. Fit two exponential models:

       Method 1:
           P(k) = C * exp(-k / gamma)

       Method 2, following the form stated by Hartmann & Cirunay:
           P(k) = (1 / gamma) * exp(-k / gamma)

4. Calculate goodness-of-fit statistics.
5. Plot the empirical degree distribution and both fitted functions.
6. Print and save a summary of the fitted decay parameters.

Parallel lines are counted independently.

Transformer-only buses that do not occur in model.line are not part of
the line graph and therefore do not enter the degree distribution.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ANALYSIS_CASES = {
    "220 kV": {
        "voltage_levels": [220],
        "include_transformers": False,
    },
    "300 kV": {
        "voltage_levels": [300],
        "include_transformers": False,
    },
    "380 kV": {
        "voltage_levels": [380],
        "include_transformers": False,
    },
    "Aggregated HV lines only": {
        "voltage_levels": [220, 300, 380],
        "include_transformers": False,
    },
    "Aggregated HV incl. transformers": {
        "voltage_levels": [220, 300, 380],
        "include_transformers": True,
    },
}

OUTPUT_DIR = Path(
    "n490_degree_decay_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_FONTSIZE = (
    plt.rcParams["font.size"]
    * 1.35
)


# ---------------------------------------------------------------------
# Endpoint detection
# ---------------------------------------------------------------------

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus-endpoint columns in model.line.
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
            return bus0_col, bus1_col

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available columns:\n"
        f"{lines.columns.tolist()}"
    )


# ---------------------------------------------------------------------
# Degree calculation
# ---------------------------------------------------------------------

def calculate_nodal_degrees(
    lines: pd.DataFrame,
    trafos: pd.DataFrame | None = None,
    voltage_levels: list[int] | None = None,
    include_transformers: bool = False,
) -> np.ndarray:
    """
    Calculate nodal degrees from N490 branches.

    Parameters
    ----------
    lines:
        N490 model.line table.

    trafos:
        N490 model.trafo table. Required if include_transformers=True.

    voltage_levels:
        Voltage levels of model.line to include.

        Examples
        --------
        [220]             -> 220 kV line network
        [300]             -> 300 kV line network
        [380]             -> 380 kV line network
        [220, 300, 380]   -> aggregated line-only HV network
        None              -> all model.line branches

    include_transformers:
        If True, transformer branches from model.trafo are added to
        the graph as undirected edges.

    Notes
    -----
    Each branch contributes one degree to each terminal bus.

    Parallel lines and parallel transformers are counted separately.
    """
    if "Vbase" not in lines.columns:
        raise ValueError(
            "model.line does not contain 'Vbase'."
        )

    line_bus0, line_bus1 = (
        resolve_line_endpoint_columns(
            lines
        )
    )

    # -------------------------------------------------------------
    # Select AC lines
    # -------------------------------------------------------------
    if voltage_levels is None:
        selected_lines = lines[
            [line_bus0, line_bus1]
        ].copy()

    else:
        line_voltage = pd.to_numeric(
            lines["Vbase"],
            errors="coerce",
        )

        line_mask = np.zeros(
            len(lines),
            dtype=bool,
        )

        for voltage_kv in voltage_levels:
            line_mask |= np.isclose(
                line_voltage,
                float(voltage_kv),
                equal_nan=False,
            )

        selected_lines = lines.loc[
            line_mask,
            [line_bus0, line_bus1],
        ].copy()

    if selected_lines.empty:
        raise ValueError(
            "No model.line branches found for requested "
            "voltage levels."
        )

    # -------------------------------------------------------------
    # Collect line endpoints
    # -------------------------------------------------------------
    endpoint_series = [
        selected_lines[line_bus0],
        selected_lines[line_bus1],
    ]

    n_branches = len(
        selected_lines
    )

    # -------------------------------------------------------------
    # Optionally add transformers
    # -------------------------------------------------------------
    if include_transformers:

        if trafos is None:
            raise ValueError(
                "trafos must be supplied when "
                "include_transformers=True."
            )

        trafo_bus0, trafo_bus1 = (
            resolve_line_endpoint_columns(
                trafos
            )
        )

        selected_trafos = trafos[
            [trafo_bus0, trafo_bus1]
        ].copy()

        endpoint_series.extend(
            [
                selected_trafos[trafo_bus0],
                selected_trafos[trafo_bus1],
            ]
        )

        n_branches += len(
            selected_trafos
        )

    # -------------------------------------------------------------
    # Count endpoint occurrences
    # -------------------------------------------------------------
    endpoints = pd.concat(
        endpoint_series,
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

    expected = (
        2 * n_branches
    )

    if degree_sum != expected:
        raise RuntimeError(
            "Degree check failed: "
            f"sum(k)={degree_sum}, "
            f"2E={expected}."
        )

    return degrees

# ---------------------------------------------------------------------
# Empirical PDF
# ---------------------------------------------------------------------

def calculate_degree_pdf(
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate empirical P(k).

    Returns
    -------
    k
        Observed integer node degrees.
    counts
        Number of nodes having each degree.
    probability
        Fraction of nodes having each degree.
    """
    k, counts = np.unique(
        degrees,
        return_counts=True,
    )

    probability = (
        counts
        / counts.sum()
    )

    return (
        k.astype(float),
        counts.astype(int),
        probability.astype(float),
    )


# ---------------------------------------------------------------------
# Exponential models
# ---------------------------------------------------------------------

def exponential_free(
    k: np.ndarray,
    C: float,
    gamma: float,
) -> np.ndarray:
    """
    P(k) = C exp(-k / gamma)
    """
    return (
        C
        * np.exp(
            -k / gamma
        )
    )


def exponential_paper(
    k: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    P(k) = (1 / gamma) exp(-k / gamma)

    This imposes C = 1 / gamma.
    """
    return (
        (1.0 / gamma)
        * np.exp(
            -k / gamma
        )
    )


# ---------------------------------------------------------------------
# Goodness-of-fit
# ---------------------------------------------------------------------

def calculate_fit_statistics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """
    Return R-squared and RMSE on empirical PDF values.
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


# ---------------------------------------------------------------------
# Fit method 1:
# free C and gamma
# ---------------------------------------------------------------------

def fit_free_exponential(
    k: np.ndarray,
    probability: np.ndarray,
    mean_degree: float,
) -> dict:
    """
    Fit

        P(k) = C exp(-k / gamma)

    with both C and gamma estimated.
    """
    initial_C = float(
        probability.max()
    )

    initial_gamma = float(
        mean_degree
    )

    popt, pcov = curve_fit(
        exponential_free,
        k,
        probability,
        p0=[
            initial_C,
            initial_gamma,
        ],
        bounds=(
            [0.0, 1e-12],
            [np.inf, np.inf],
        ),
        maxfev=100000,
    )

    C, gamma = popt

    predicted = exponential_free(
        k,
        C,
        gamma,
    )

    r_squared, rmse = (
        calculate_fit_statistics(
            probability,
            predicted,
        )
    )

    return {
        "C": float(C),
        "gamma": float(gamma),
        "r_squared": r_squared,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------
# Fit method 2:
# C = 1 / gamma
# ---------------------------------------------------------------------

def fit_paper_exponential(
    k: np.ndarray,
    probability: np.ndarray,
    mean_degree: float,
) -> dict:
    """
    Fit the form stated in the comparison paper:

        P(k) = (1 / gamma) exp(-k / gamma)

    Only gamma is fitted.
    """
    popt, pcov = curve_fit(
        exponential_paper,
        k,
        probability,
        p0=[
            float(mean_degree)
        ],
        bounds=(
            [1e-12],
            [np.inf],
        ),
        maxfev=100000,
    )

    gamma = float(
        popt[0]
    )

    predicted = exponential_paper(
        k,
        gamma,
    )

    r_squared, rmse = (
        calculate_fit_statistics(
            probability,
            predicted,
        )
    )

    return {
        "C": float(
            1.0 / gamma
        ),
        "gamma": gamma,
        "r_squared": r_squared,
        "rmse": rmse,
    }

def count_selected_branches(
    lines: pd.DataFrame,
    trafos: pd.DataFrame,
    voltage_levels: list[int],
    include_transformers: bool,
) -> tuple[int, int]:
    """
    Return number of selected AC lines and transformers.
    """
    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    line_mask = np.zeros(
        len(lines),
        dtype=bool,
    )

    for voltage_kv in voltage_levels:
        line_mask |= np.isclose(
            line_voltage,
            float(voltage_kv),
            equal_nan=False,
        )

    n_lines = int(
        line_mask.sum()
    )

    n_trafos = (
        len(trafos)
        if include_transformers
        else 0
    )

    return n_lines, n_trafos


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_degree_decay(
    case_name: str,
    k: np.ndarray,
    counts: np.ndarray,
    probability: np.ndarray,
    free_fit: dict,
    paper_fit: dict,
) -> None:
    """
    Plot empirical P(k) together with both exponential fits.
    """
    fig, ax = plt.subplots(
        figsize=(8.5, 5.8)
    )

    # -------------------------------------------------------------
    # Empirical distribution
    # -------------------------------------------------------------
    ax.bar(
        k,
        probability,
        width=0.72,
        alpha=0.55,
        label="Observed degree distribution",
        zorder=2,
    )

    # -------------------------------------------------------------
    # Smooth fitted curves
    # -------------------------------------------------------------
    k_fit = np.linspace(
        float(k.min()),
        float(k.max()),
        400,
    )

    free_probability = (
        exponential_free(
            k_fit,
            free_fit["C"],
            free_fit["gamma"],
        )
    )

    paper_probability = (
        exponential_paper(
            k_fit,
            paper_fit["gamma"],
        )
    )

    ax.plot(
        k_fit,
        free_probability,
        linewidth=2.2,
        label=(
            "Free $C$: "
            rf"$\gamma={free_fit['gamma']:.3f}$"
        ),
        zorder=4,
    )

    ax.plot(
        k_fit,
        paper_probability,
        linestyle="--",
        linewidth=2.2,
        label=(
            r"$C=1/\gamma$: "
            rf"$\gamma={paper_fit['gamma']:.3f}$"
        ),
        zorder=4,
    )

    # Show empirical points clearly on top of bars.
    ax.scatter(
        k,
        probability,
        s=45,
        zorder=5,
    )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------
    ax.set_xlabel(
        "Nodal degree $k$",
        fontsize=BASE_FONTSIZE,
    )

    ax.set_ylabel(
        "Probability $P(k)$",
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
        bottom=0
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
        "Free $C$\n"
        rf"$R^2={free_fit['r_squared']:.3f}$"
        "\n"
        rf"RMSE={free_fit['rmse']:.4f}"
        "\n\n"
        r"$C=1/\gamma$"
        "\n"
        rf"$R^2={paper_fit['r_squared']:.3f}$"
        "\n"
        rf"RMSE={paper_fit['rmse']:.4f}"
    )

    ax.text(
        0.98,
        0.97,
        statistics_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=BASE_FONTSIZE * 0.85,
    )

    plt.tight_layout()

    safe_name = (
        case_name
        .lower()
        .replace(" ", "_")
    )
    
    output_path = (
        OUTPUT_DIR
        / f"N490_{safe_name}_degree_exponential_fits.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved:\n  {output_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    model = N490(
        year=2018
    )
    
    lines = model.line.copy()
    trafos = model.trafo.copy()

    lines = model.line.copy()

    summary_rows = []
    distribution_rows = []
    
    

    for case_name, case_config in ANALYSIS_CASES.items():
    
        voltage_levels = case_config[
            "voltage_levels"
        ]
    
        include_transformers = case_config[
            "include_transformers"
        ]
    
        degrees = calculate_nodal_degrees(
            lines=lines,
            trafos=trafos,
            voltage_levels=voltage_levels,
            include_transformers=include_transformers,
        )
    
        k, counts, probability = (
            calculate_degree_pdf(
                degrees
            )
        )
    
        mean_degree = float(
            np.mean(degrees)
        )
    
        free_fit = fit_free_exponential(
            k=k,
            probability=probability,
            mean_degree=mean_degree,
        )
    
        paper_fit = fit_paper_exponential(
            k=k,
            probability=probability,
            mean_degree=mean_degree,
        )
        
        n_lines, n_trafos = (
        count_selected_branches(
            lines=lines,
            trafos=trafos,
            voltage_levels=voltage_levels,
            include_transformers=include_transformers,
        )
    )
    
        summary_rows.append(
            {
                "network": case_name,
                "voltage_levels": ",".join(
                    str(v)
                    for v in voltage_levels
                ),
                "include_transformers":
                    include_transformers,
    
                "n_nodes":
                    len(degrees),
    
                "mean_degree":
                    mean_degree,
    
                "gamma_free_C":
                    free_fit["gamma"],
    
                "C_free":
                    free_fit["C"],
    
                "R2_free_C":
                    free_fit["r_squared"],
    
                "RMSE_free_C":
                    free_fit["rmse"],
    
                "gamma_C_equals_1_over_gamma":
                    paper_fit["gamma"],
    
                "C_equals_1_over_gamma":
                    paper_fit["C"],
    
                "R2_C_equals_1_over_gamma":
                    paper_fit["r_squared"],
    
                "RMSE_C_equals_1_over_gamma":
                    paper_fit["rmse"],
                    
                "n_lines": n_lines,
                "n_transformers": n_trafos,
                "n_total_branches": n_lines + n_trafos,
            }
        )
    
        for degree, count, prob in zip(
            k,
            counts,
            probability,
        ):
            distribution_rows.append(
                {
                    "network":
                        case_name,
    
                    "voltage_levels":
                        ",".join(
                            str(v)
                            for v in voltage_levels
                        ),
    
                    "include_transformers":
                        include_transformers,
    
                    "degree":
                        int(degree),
    
                    "n_nodes":
                        int(count),
    
                    "probability":
                        float(prob),
                }
            )
    
        plot_degree_decay(
            case_name=case_name,
            k=k,
            counts=counts,
            probability=probability,
            free_fit=free_fit,
            paper_fit=paper_fit,
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

    print("\n")
    print("=" * 100)
    print("N490 exponential nodal-degree decay fits")
    print("=" * 100)

    print(
        summary
        .round(
            {
                "mean_degree": 4,
                "gamma_free_C": 4,
                "C_free": 4,
                "R2_free_C": 4,
                "RMSE_free_C": 4,
                "gamma_C_equals_1_over_gamma": 4,
                "C_equals_1_over_gamma": 4,
                "R2_C_equals_1_over_gamma": 4,
                "RMSE_C_equals_1_over_gamma": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    # Cleaner comparison table focused on gamma.
    gamma_summary = summary[
        [
            "network",
            "voltage_levels",
            "n_nodes",
            "mean_degree",
            "gamma_free_C",
            "gamma_C_equals_1_over_gamma",
            "R2_free_C",
            "R2_C_equals_1_over_gamma",
        ]
    ].copy()

    print("\n")
    print("=" * 100)
    print("Decay-parameter comparison")
    print("=" * 100)

    print(
        gamma_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    summary.to_csv(
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.csv",
        index=False,
    )

    summary.to_pickle(
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.pkl"
    )

    distributions.to_csv(
        OUTPUT_DIR
        / "N490_degree_distributions.csv",
        index=False,
    )

    distributions.to_pickle(
        OUTPUT_DIR
        / "N490_degree_distributions.pkl"
    )

    print("\nSaved:")
    print(
        " ",
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.csv",
    )
    print(
        " ",
        OUTPUT_DIR
        / "N490_degree_decay_fit_summary.pkl",
    )
    print(
        " ",
        OUTPUT_DIR
        / "N490_degree_distributions.csv",
    )
    print(
        " ",
        OUTPUT_DIR
        / "N490_degree_distributions.pkl",
    )


if __name__ == "__main__":
    main()