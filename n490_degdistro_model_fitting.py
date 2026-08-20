# -*- coding: utf-8 -*-
"""
Compare candidate probability distributions for Nordic490 nodal degrees.

Networks analyzed
-----------------
- 220 kV line network
- 300 kV line network
- 380 kV line network
- Aggregated 220 + 300 + 380 kV line network

Only model.line branches are used. Transformers are not included.

Candidate degree distributions
------------------------------
1. Weibull
2. Gamma
3. Lognormal
4. Shifted negative binomial

For the continuous distributions, probability at integer degree k is
calculated as the probability mass in the bin

    k - 0.5 <= X < k + 0.5

conditional on X >= 0.5.

Thus each model defines a proper discrete PMF on degrees 1, 2, 3, ...

Models are fitted by maximum likelihood to the observed node degrees.

Outputs
-------
- one fitted-distribution plot per network
- CSV and pickle files containing fitted parameters and fit statistics
- empirical degree distributions
- console summary ranked by AIC
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import (
    weibull_min,
    gamma as gamma_dist,
    lognorm,
    nbinom,
)

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VOLTAGE_LEVELS = [220, 300, 380]

ANALYSIS_CASES = {
    "220 kV": [220],
    "300 kV": [300],
    "380 kV": [380],
    "Aggregated HV": [220, 300, 380],
}

OUTPUT_DIR = Path(
    "n490_degree_distribution_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_FONTSIZE = (
    plt.rcParams["font.size"]
    * 1.30
)


# ---------------------------------------------------------------------
# N490 graph construction
# ---------------------------------------------------------------------

def resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two endpoint columns in model.line.
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


def calculate_nodal_degrees(
    lines: pd.DataFrame,
    voltage_levels: list[int],
) -> np.ndarray:
    """
    Calculate nodal degree from model.line endpoints.

    The node set consists only of buses that actually occur as line
    endpoints in the selected voltage network.

    Parallel lines are counted separately.
    """
    if "Vbase" not in lines.columns:
        raise ValueError(
            "model.line does not contain 'Vbase'."
        )

    bus0_col, bus1_col = (
        resolve_line_endpoint_columns(
            lines
        )
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

    selected_lines = lines.loc[
        mask,
        [bus0_col, bus1_col],
    ].copy()

    if selected_lines.empty:
        raise ValueError(
            "No lines found for voltage levels "
            f"{voltage_levels}."
        )

    endpoints = pd.concat(
        [
            selected_lines[bus0_col],
            selected_lines[bus1_col],
        ],
        ignore_index=True,
    ).dropna()

    degree_series = (
        endpoints
        .value_counts()
        .astype(int)
    )

    degrees = (
        degree_series
        .to_numpy(dtype=int)
    )

    # Handshaking-lemma sanity check.
    degree_sum = int(
        degrees.sum()
    )

    expected = (
        2 * len(selected_lines)
    )

    if degree_sum != expected:

        raise RuntimeError(
            "Degree check failed: "
            f"sum(k)={degree_sum}, "
            f"2E={expected}."
        )

    return degrees


# ---------------------------------------------------------------------
# Empirical distribution
# ---------------------------------------------------------------------

def empirical_degree_distribution(
    degrees: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return integer degrees, counts, and empirical probabilities.
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
        k.astype(int),
        counts.astype(int),
        probability.astype(float),
    )


# ---------------------------------------------------------------------
# Discretized continuous distributions
# ---------------------------------------------------------------------

def discretized_continuous_pmf(
    k: np.ndarray,
    distribution,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Convert a positive continuous distribution into a discrete degree PMF.

    P(K=k) is the probability between k-0.5 and k+0.5,
    conditional on X >= 0.5.

    This gives a proper probability distribution over
    k = 1, 2, 3, ...
    """
    k = np.asarray(
        k,
        dtype=float,
    )

    lower = (
        k - 0.5
    )

    upper = (
        k + 0.5
    )

    numerator = (
        distribution.cdf(
            upper,
            *args,
            **kwargs,
        )
        - distribution.cdf(
            lower,
            *args,
            **kwargs,
        )
    )

    denominator = distribution.sf(
        0.5,
        *args,
        **kwargs,
    )

    return (
        numerator
        / denominator
    )


# ---------------------------------------------------------------------
# Candidate PMFs
# ---------------------------------------------------------------------

def weibull_pmf(
    k: np.ndarray,
    shape: float,
    scale: float,
) -> np.ndarray:
    """
    Discretized Weibull distribution.
    """
    return discretized_continuous_pmf(
        k,
        weibull_min,
        shape,
        loc=0,
        scale=scale,
    )


def gamma_pmf(
    k: np.ndarray,
    shape: float,
    scale: float,
) -> np.ndarray:
    """
    Discretized gamma distribution.
    """
    return discretized_continuous_pmf(
        k,
        gamma_dist,
        shape,
        loc=0,
        scale=scale,
    )


def lognormal_pmf(
    k: np.ndarray,
    sigma: float,
    scale: float,
) -> np.ndarray:
    """
    Discretized lognormal distribution.

    scale = exp(mu).
    """
    return discretized_continuous_pmf(
        k,
        lognorm,
        sigma,
        loc=0,
        scale=scale,
    )


def negative_binomial_pmf(
    k: np.ndarray,
    n: float,
    p: float,
) -> np.ndarray:
    """
    Shifted negative binomial distribution.

    scipy.stats.nbinom is defined for x = 0, 1, 2, ...

    Here:

        degree = x + 1

    so the degree distribution has support 1, 2, 3, ...
    """
    k = np.asarray(
        k,
        dtype=int,
    )

    return nbinom.pmf(
        k - 1,
        n,
        p,
    )


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------

def negative_log_likelihood(
    probability: np.ndarray,
    counts: np.ndarray,
) -> float:
    """
    Multinomial negative log likelihood, excluding constants.
    """
    probability = np.clip(
        probability,
        1e-300,
        1.0,
    )

    return float(
        -np.sum(
            counts
            * np.log(probability)
        )
    )


def calculate_fit_statistics(
    empirical: np.ndarray,
    fitted: np.ndarray,
    nll: float,
    n_parameters: int,
    n_observations: int,
) -> dict:
    """
    Calculate R2, RMSE, AIC and BIC.
    """
    residuals = (
        empirical
        - fitted
    )

    ss_res = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                empirical
                - np.mean(empirical)
            ) ** 2
        )
    )

    r_squared = (
        1.0
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    aic = (
        2 * n_parameters
        + 2 * nll
    )

    bic = (
        n_parameters
        * np.log(n_observations)
        + 2 * nll
    )

    return {
        "nll": float(nll),
        "AIC": float(aic),
        "BIC": float(bic),
        "R2": float(r_squared),
        "RMSE": rmse,
    }


# ---------------------------------------------------------------------
# Weibull fit
# ---------------------------------------------------------------------

def fit_weibull(
    k,
    counts,
    empirical,
) -> dict:
    """
    Fit discretized Weibull distribution.
    """

    def objective(theta):

        shape = np.exp(
            theta[0]
        )

        scale = np.exp(
            theta[1]
        )

        probability = weibull_pmf(
            k,
            shape,
            scale,
        )

        return negative_log_likelihood(
            probability,
            counts,
        )

    mean_degree = np.average(
        k,
        weights=counts,
    )

    result = minimize(
        objective,
        x0=np.log(
            [
                2.0,
                mean_degree,
            ]
        ),
        method="L-BFGS-B",
        bounds=[
            (-5, 5),
            (-5, 7),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "Weibull fit failed: "
            f"{result.message}"
        )

    shape, scale = np.exp(
        result.x
    )

    fitted = weibull_pmf(
        k,
        shape,
        scale,
    )

    statistics = calculate_fit_statistics(
        empirical=empirical,
        fitted=fitted,
        nll=result.fun,
        n_parameters=2,
        n_observations=int(
            counts.sum()
        ),
    )

    return {
        "model": "Weibull",
        "shape": float(shape),
        "scale": float(scale),
        "fitted": fitted,
        **statistics,
    }


# ---------------------------------------------------------------------
# Gamma fit
# ---------------------------------------------------------------------

def fit_gamma(
    k,
    counts,
    empirical,
) -> dict:
    """
    Fit discretized gamma distribution.
    """

    degree_samples = np.repeat(
        k,
        counts,
    )

    mean_degree = float(
        np.mean(degree_samples)
    )

    variance = float(
        np.var(
            degree_samples
        )
    )

    if variance > 0:
        initial_shape = (
            mean_degree ** 2
            / variance
        )

        initial_scale = (
            variance
            / mean_degree
        )

    else:
        initial_shape = 2.0
        initial_scale = mean_degree / 2

    initial_shape = max(
        initial_shape,
        0.1,
    )

    initial_scale = max(
        initial_scale,
        0.1,
    )

    def objective(theta):

        shape = np.exp(
            theta[0]
        )

        scale = np.exp(
            theta[1]
        )

        probability = gamma_pmf(
            k,
            shape,
            scale,
        )

        return negative_log_likelihood(
            probability,
            counts,
        )

    result = minimize(
        objective,
        x0=np.log(
            [
                initial_shape,
                initial_scale,
            ]
        ),
        method="L-BFGS-B",
        bounds=[
            (-5, 7),
            (-5, 7),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "Gamma fit failed: "
            f"{result.message}"
        )

    shape, scale = np.exp(
        result.x
    )

    fitted = gamma_pmf(
        k,
        shape,
        scale,
    )

    statistics = calculate_fit_statistics(
        empirical=empirical,
        fitted=fitted,
        nll=result.fun,
        n_parameters=2,
        n_observations=int(
            counts.sum()
        ),
    )

    return {
        "model": "Gamma",
        "shape": float(shape),
        "scale": float(scale),
        "fitted": fitted,
        **statistics,
    }


# ---------------------------------------------------------------------
# Lognormal fit
# ---------------------------------------------------------------------

def fit_lognormal(
    k,
    counts,
    empirical,
) -> dict:
    """
    Fit discretized lognormal distribution.
    """
    degree_samples = np.repeat(
        k,
        counts,
    ).astype(float)

    log_degree = np.log(
        degree_samples
    )

    initial_sigma = max(
        float(
            np.std(log_degree)
        ),
        0.1,
    )

    initial_scale = float(
        np.exp(
            np.mean(log_degree)
        )
    )

    def objective(theta):

        sigma = np.exp(
            theta[0]
        )

        scale = np.exp(
            theta[1]
        )

        probability = lognormal_pmf(
            k,
            sigma,
            scale,
        )

        return negative_log_likelihood(
            probability,
            counts,
        )

    result = minimize(
        objective,
        x0=np.log(
            [
                initial_sigma,
                initial_scale,
            ]
        ),
        method="L-BFGS-B",
        bounds=[
            (-5, 5),
            (-5, 7),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "Lognormal fit failed: "
            f"{result.message}"
        )

    sigma, scale = np.exp(
        result.x
    )

    fitted = lognormal_pmf(
        k,
        sigma,
        scale,
    )

    statistics = calculate_fit_statistics(
        empirical=empirical,
        fitted=fitted,
        nll=result.fun,
        n_parameters=2,
        n_observations=int(
            counts.sum()
        ),
    )

    return {
        "model": "Lognormal",
        "sigma": float(sigma),
        "scale": float(scale),
        "mu": float(
            np.log(scale)
        ),
        "fitted": fitted,
        **statistics,
    }


# ---------------------------------------------------------------------
# Negative-binomial fit
# ---------------------------------------------------------------------

def fit_negative_binomial(
    k,
    counts,
    empirical,
) -> dict:
    """
    Fit shifted negative binomial distribution.
    """
    samples = np.repeat(
        k,
        counts,
    )

    x = (
        samples - 1
    )

    mean_x = float(
        np.mean(x)
    )

    variance_x = float(
        np.var(x)
    )

    # Method-of-moments starting guess.
    if (
        variance_x > mean_x
        and mean_x > 0
    ):
        initial_n = (
            mean_x ** 2
            / (
                variance_x
                - mean_x
            )
        )

        initial_p = (
            initial_n
            / (
                initial_n
                + mean_x
            )
        )

    else:
        # Near-Poisson / underdispersed starting point.
        initial_n = 50.0

        initial_p = (
            initial_n
            / (
                initial_n
                + max(
                    mean_x,
                    1e-3,
                )
            )
        )

    def logit(p):
        return np.log(
            p / (1 - p)
        )

    def sigmoid(x):
        return (
            1.0
            / (
                1.0
                + np.exp(-x)
            )
        )

    def objective(theta):

        n = np.exp(
            theta[0]
        )

        p = sigmoid(
            theta[1]
        )

        probability = (
            negative_binomial_pmf(
                k,
                n,
                p,
            )
        )

        return negative_log_likelihood(
            probability,
            counts,
        )

    result = minimize(
        objective,
        x0=[
            np.log(initial_n),
            logit(initial_p),
        ],
        method="L-BFGS-B",
        bounds=[
            (-5, 10),
            (-10, 10),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "Negative-binomial fit failed: "
            f"{result.message}"
        )

    n = np.exp(
        result.x[0]
    )

    p = sigmoid(
        result.x[1]
    )

    fitted = (
        negative_binomial_pmf(
            k,
            n,
            p,
        )
    )

    statistics = calculate_fit_statistics(
        empirical=empirical,
        fitted=fitted,
        nll=result.fun,
        n_parameters=2,
        n_observations=int(
            counts.sum()
        ),
    )

    return {
        "model": "Negative binomial",
        "n": float(n),
        "p": float(p),
        "fitted": fitted,
        **statistics,
    }


# ---------------------------------------------------------------------
# Fit all candidate distributions
# ---------------------------------------------------------------------

def fit_candidate_distributions(
    degrees: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    """
    Fit all candidate models to one network.
    """
    k, counts, empirical = (
        empirical_degree_distribution(
            degrees
        )
    )

    fits = [
        fit_weibull(
            k,
            counts,
            empirical,
        ),
        fit_gamma(
            k,
            counts,
            empirical,
        ),
        fit_lognormal(
            k,
            counts,
            empirical,
        ),
        fit_negative_binomial(
            k,
            counts,
            empirical,
        ),
    ]

    # AIC ranking.
    fits = sorted(
        fits,
        key=lambda result:
            result["AIC"],
    )

    best_aic = (
        fits[0]["AIC"]
    )

    for fit in fits:
        fit["delta_AIC"] = (
            fit["AIC"]
            - best_aic
        )

    return (
        k,
        counts,
        empirical,
        fits,
    )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_distribution_fits(
    case_name: str,
    degrees: np.ndarray,
    k: np.ndarray,
    empirical: np.ndarray,
    fits: list[dict],
) -> None:
    """
    Plot empirical degree PMF with all fitted models.
    """
    max_degree = int(
        max(degrees)
    )

    k_plot = np.arange(
        1,
        max_degree + 4,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # -------------------------------------------------------------
    # Observed distribution
    # -------------------------------------------------------------
    ax.bar(
        k,
        empirical,
        width=0.72,
        alpha=0.45,
        label="Observed",
        zorder=2,
    )

    ax.scatter(
        k,
        empirical,
        s=45,
        zorder=5,
    )

    # -------------------------------------------------------------
    # Model curves
    # -------------------------------------------------------------
    for fit in fits:

        model = fit["model"]

        if model == "Weibull":

            probability = weibull_pmf(
                k_plot,
                fit["shape"],
                fit["scale"],
            )

        elif model == "Gamma":

            probability = gamma_pmf(
                k_plot,
                fit["shape"],
                fit["scale"],
            )

        elif model == "Lognormal":

            probability = lognormal_pmf(
                k_plot,
                fit["sigma"],
                fit["scale"],
            )

        elif model == "Negative binomial":

            probability = (
                negative_binomial_pmf(
                    k_plot,
                    fit["n"],
                    fit["p"],
                )
            )

        else:
            continue

        ax.plot(
            k_plot,
            probability,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=(
                f"{model} "
                rf"($\Delta$AIC={fit['delta_AIC']:.1f})"
            ),
            zorder=4,
        )

    # -------------------------------------------------------------
    # Formatting
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
        f"N490 {case_name} degree distribution"
    )

    ax.set_xticks(
        np.arange(
            1,
            max_degree + 1,
        )
    )

    ax.set_ylim(
        bottom=0
    )

    ax.tick_params(
        axis="both",
        labelsize=BASE_FONTSIZE,
    )

    ax.grid(False)

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        fontsize=BASE_FONTSIZE * 0.78
    )

    plt.tight_layout()

    safe_name = (
        case_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    output_path = (
        OUTPUT_DIR
        / f"N490_{safe_name}_distribution_fits.png"
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

    fit_summary_rows = []
    empirical_rows = []

    # -------------------------------------------------------------
    # Analyze each network
    # -------------------------------------------------------------
    for (
        case_name,
        voltage_levels,
    ) in ANALYSIS_CASES.items():

        print("\n")
        print("=" * 100)
        print(case_name)
        print("=" * 100)

        degrees = calculate_nodal_degrees(
            lines=lines,
            voltage_levels=voltage_levels,
        )

        (
            k,
            counts,
            empirical,
            fits,
        ) = fit_candidate_distributions(
            degrees
        )

        # ---------------------------------------------------------
        # Print summary
        # ---------------------------------------------------------
        print(
            f"Nodes:       {len(degrees)}"
        )

        print(
            f"Mean degree: {np.mean(degrees):.4f}"
        )

        print(
            f"Std. degree: {np.std(degrees):.4f}"
        )

        print(
            f"Max degree:  {np.max(degrees)}"
        )

        print("\nModel ranking by AIC:")
        print("-" * 100)

        print(
            f"{'Model':22s}"
            f"{'AIC':>12s}"
            f"{'Delta AIC':>14s}"
            f"{'BIC':>12s}"
            f"{'R2':>12s}"
            f"{'RMSE':>12s}"
        )

        for fit in fits:

            print(
                f"{fit['model']:22s}"
                f"{fit['AIC']:12.3f}"
                f"{fit['delta_AIC']:14.3f}"
                f"{fit['BIC']:12.3f}"
                f"{fit['R2']:12.4f}"
                f"{fit['RMSE']:12.4f}"
            )

        # ---------------------------------------------------------
        # Save fitted-model summaries
        # ---------------------------------------------------------
        for rank, fit in enumerate(
            fits,
            start=1,
        ):

            row = {
                "network":
                    case_name,

                "voltage_levels":
                    ",".join(
                        str(v)
                        for v
                        in voltage_levels
                    ),

                "n_nodes":
                    len(degrees),

                "mean_degree":
                    float(
                        np.mean(degrees)
                    ),

                "std_degree":
                    float(
                        np.std(degrees)
                    ),

                "max_degree":
                    int(
                        np.max(degrees)
                    ),

                "AIC_rank":
                    rank,

                "model":
                    fit["model"],

                "AIC":
                    fit["AIC"],

                "delta_AIC":
                    fit["delta_AIC"],

                "BIC":
                    fit["BIC"],

                "R2":
                    fit["R2"],

                "RMSE":
                    fit["RMSE"],

                "NLL":
                    fit["nll"],
            }

            # Store model-specific parameters.
            for parameter in [
                "shape",
                "scale",
                "sigma",
                "mu",
                "n",
                "p",
            ]:

                if parameter in fit:
                    row[
                        parameter
                    ] = fit[
                        parameter
                    ]

            fit_summary_rows.append(
                row
            )

        # ---------------------------------------------------------
        # Save empirical distribution
        # ---------------------------------------------------------
        for (
            degree,
            count,
            probability,
        ) in zip(
            k,
            counts,
            empirical,
        ):

            empirical_rows.append(
                {
                    "network":
                        case_name,

                    "voltage_levels":
                        ",".join(
                            str(v)
                            for v
                            in voltage_levels
                        ),

                    "degree":
                        int(degree),

                    "n_nodes":
                        int(count),

                    "probability":
                        float(probability),
                }
            )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------
        plot_distribution_fits(
            case_name=case_name,
            degrees=degrees,
            k=k,
            empirical=empirical,
            fits=fits,
        )

    # -----------------------------------------------------------------
    # Final tables
    # -----------------------------------------------------------------
    fit_summary = pd.DataFrame(
        fit_summary_rows
    )

    empirical_summary = pd.DataFrame(
        empirical_rows
    )

    # Best model for each network.
    best_models = (
        fit_summary
        .loc[
            fit_summary[
                "AIC_rank"
            ] == 1
        ]
        .copy()
    )

    print("\n")
    print("=" * 100)
    print("Best-fitting model by network")
    print("=" * 100)

    print(
        best_models[
            [
                "network",
                "n_nodes",
                "mean_degree",
                "model",
                "AIC",
                "BIC",
                "R2",
                "RMSE",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    fit_summary.to_csv(
        OUTPUT_DIR
        / "N490_degree_distribution_fit_summary.csv",
        index=False,
    )

    fit_summary.to_pickle(
        OUTPUT_DIR
        / "N490_degree_distribution_fit_summary.pkl"
    )

    empirical_summary.to_csv(
        OUTPUT_DIR
        / "N490_empirical_degree_distributions.csv",
        index=False,
    )

    empirical_summary.to_pickle(
        OUTPUT_DIR
        / "N490_empirical_degree_distributions.pkl"
    )

    best_models.to_csv(
        OUTPUT_DIR
        / "N490_best_degree_distribution_models.csv",
        index=False,
    )

    print("\nSaved results in:")
    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()