#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hartmann & Cirunay (2026) node-degree exponential-fit investigation
==================================================================

Loads:
    euro-comparison/european_networks.pkl

For each complete country network:

    1. Calculate node degrees, retaining parallel circuits.

    2. Calculate the complementary cumulative degree distribution:

           P(K >= k)

    3. Fit the one-parameter exponential function:

           P(K >= k) = exp(-k / gamma)

       using nonlinear least squares in ordinary probability space.

       IMPORTANT:
       ----------
       gamma is the ONLY fitted parameter.

       The prefactor C is not fitted independently. It is constrained by:

           C = 1 / gamma

    4. Compare the fitted gamma with the published value in
       Hartmann & Cirunay (2026), Table 1:

           "gamma of HV grid exp. fit"

No voltage-specific subnetworks are analyzed in this script.
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

OUTPUT_DIR = DATA_DIR / "hartmann-investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = (
    OUTPUT_DIR
    / "hartmann_gamma_fit_comparison.csv"
)

PLOT_DIR = OUTPUT_DIR / "country-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# PLOT SETTINGS
# =====================================================================

FIGSIZE = (7.0, 5.0)
DPI = 300
TEXT_SIZE = 12


# =====================================================================
# PUBLISHED VALUES
# Hartmann & Cirunay (2026), Table 1
# "gamma of HV grid exp. fit"
# =====================================================================

PUBLISHED_GAMMA = {
    "Albania": 1.108,
    "Bosnia&Herzegovina": 1.376,
    "Belgium": 3.845,
    "Czechia": 2.266,
    "Denmark": 2.066,
    "Estonia": 1.503,
    "Croatia": 1.514,
    "Hungary": 1.726,
    "Ireland": 2.081,
    "Lithuania": 1.039,
    "Latvia": 1.409,
    "Netherlands": 3.027,
    "Portugal": 3.748,
    "Slovenia": 1.663,
    "Slovakia": 2.248,
}


# =====================================================================
# MODEL
# =====================================================================

def exponential_cumulative(k, gamma):
    """
    One-parameter exponential model:

        P(K >= k) = exp(-k / gamma)

    gamma is the only fitted parameter.
    """

    return (
        np.exp(-k / gamma)
    )


# =====================================================================
# DEGREE CALCULATION
# =====================================================================

def calculate_node_degrees(edges):
    """
    Calculate node degree directly from the edge list.

    Every row is treated as one physical branch, so parallel circuits
    are retained and contribute separately to node degree.
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
# COMPLEMENTARY CUMULATIVE DISTRIBUTION
# =====================================================================

def calculate_cumulative_distribution(degrees):
    """
    Calculate:

        P(K >= k)

    for every integer degree from 1 through max degree.

    By construction:

        P(K >= 1) = 1
    """

    if len(degrees) == 0:
        return np.array([]), np.array([])

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

    return k, probability


# =====================================================================
# FIT QUALITY
# =====================================================================

def calculate_r2(
    observed,
    fitted,
):
    """
    Standard R^2 in ordinary probability space.
    """

    ss_res = np.sum(
        (observed - fitted) ** 2
    )

    ss_tot = np.sum(
        (
            observed
            - np.mean(observed)
        ) ** 2
    )

    if ss_tot <= 0:
        return np.nan

    return (
        1.0
        - ss_res / ss_tot
    )


def calculate_rmse(
    observed,
    fitted,
):
    """
    RMSE in ordinary probability space.
    """

    if len(observed) == 0:
        return np.nan

    return np.sqrt(
        np.mean(
            (observed - fitted) ** 2
        )
    )


# =====================================================================
# NONLINEAR ONE-PARAMETER FIT
# =====================================================================

def fit_gamma(
    k,
    probability,
):
    """
    Fit:

        P(K >= k)
        = exp(-k / gamma)

    using nonlinear least squares in ordinary probability space.

    Only gamma is fitted.

    Returns
    -------
    gamma : float
        Best-fitting decay parameter.

    r2 : float
        R^2 evaluated in ordinary probability space.

    rmse : float
        RMSE evaluated in ordinary probability space.
    """

    if len(k) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            exponential_cumulative,
            k,
            probability,
            p0=[
                gamma_initial,
            ],
            bounds=(
                [1e-8],
                [np.inf],
            ),
            maxfev=100000,
        )

        gamma = float(
            popt[0]
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    fitted = (
        exponential_cumulative(
            k,
            gamma,
        )
    )

    r2 = calculate_r2(
        probability,
        fitted,
    )

    rmse = calculate_rmse(
        probability,
        fitted,
    )

    return (
        gamma,
        r2,
        rmse,
    )



# =====================================================================
# PLOTTING
# =====================================================================

def plot_country_fit(
    country,
    k,
    probability,
    gamma_fit,
    gamma_published,
):
    """
    Plot the empirical CCDF together with the exponential model evaluated
    using:

        1. the gamma estimated from our data, and
        2. the gamma reported by Hartmann & Cirunay (2026).

    The model is always:

        P(K >= k) = exp(-k / gamma)

    No parameters are re-estimated for the published curve.
    """

    # Use a dense x-grid so the two model curves appear smooth.
    k_smooth = np.linspace(
        float(k.min()),
        float(k.max()),
        400,
    )

    fitted_curve = exponential_cumulative(
        k_smooth,
        gamma_fit,
    )

    published_curve = exponential_cumulative(
        k_smooth,
        gamma_published,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    # Empirical CCDF
    ax.scatter(
        k,
        probability,
        s=45,
        label="Empirical CCDF",
        zorder=3,
    )

    # Our fitted gamma
    ax.plot(
        k_smooth,
        fitted_curve,
        linewidth=2.0,
        label=(
            f"Our fit: "
            f"$\\gamma={gamma_fit:.3f}$"
        ),
    )

    # Published gamma
    ax.plot(
        k_smooth,
        published_curve,
        linewidth=2.0,
        linestyle="--",
        label=(
            f"Hartmann & Cirunay: "
            f"$\\gamma={gamma_published:.3f}$"
        ),
    )

    ax.set_xlabel(
        "Node degree, $k$",
        fontsize=TEXT_SIZE,
    )

    ax.set_ylabel(
        r"$P(K \geq k)$",
        fontsize=TEXT_SIZE,
    )

    ax.tick_params(
        axis="both",
        labelsize=TEXT_SIZE - 1,
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.set_xlim(
        left=max(0.8, float(k.min()) - 0.2),
        right=float(k.max()) + 0.2,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.legend(
        frameon=False,
        fontsize=TEXT_SIZE - 1,
    )

    ax.set_title(
        country,
        fontsize=TEXT_SIZE + 1,
    )

    fig.tight_layout()

    safe_country = (
        country
        .replace("&", "and")
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_file = (
        PLOT_DIR
        / f"{safe_country}_gamma_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_file


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 115)
    print("HARTMANN & CIRUNAY (2026): ONE-PARAMETER GAMMA FIT")
    print("=" * 115)

    print("\nLoading:")
    print(
        f"  {INPUT_FILE}"
    )

    # -----------------------------------------------------------------
    # Method summary
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 115)
    print("METHOD OF ESTIMATION")
    print("=" * 115)

    print(
        """
For each country, the complete transmission network is used and
parallel circuits are retained when calculating node degree.

The empirical complementary cumulative distribution is calculated as:

    P(K >= k)

for each integer degree k from 1 through the maximum observed degree.

The fitted model is:

    P(K >= k) = exp(-k / gamma)

The model therefore contains ONE fitted parameter only:

    gamma

gamma is estimated using nonlinear least squares in ordinary probability
space with scipy.optimize.curve_fit. The optimizer minimizes the sum of
squared residuals between the empirical CCDF probabilities and the model
predictions.

No logarithmic transformation is applied before fitting.
"""
    )

    euro_networks = pd.read_pickle(
        INPUT_FILE
    )

    results = []

    # -----------------------------------------------------------------
    # Analyze complete network for each country
    # -----------------------------------------------------------------

    for country in sorted(
        euro_networks
    ):

        edges = (
            euro_networks[country]
        )

        degrees = (
            calculate_node_degrees(
                edges
            )
        )

        (
            k,
            probability,
        ) = (
            calculate_cumulative_distribution(
                degrees
            )
        )

        (
            gamma_fit,
            r2,
            rmse,
        ) = fit_gamma(
            k,
            probability,
        )

        gamma_published = (
            PUBLISHED_GAMMA[
                country
            ]
        )

        gamma_difference = (
            gamma_fit
            - gamma_published
        )

        gamma_error_percent = (
            100.0
            * gamma_difference
            / gamma_published
        )

        plot_file = plot_country_fit(
            country=country,
            k=k,
            probability=probability,
            gamma_fit=gamma_fit,
            gamma_published=gamma_published,
        )

        results.append(
            {
                "country": country,
                "n_nodes": len(
                    degrees
                ),
                "n_branches": len(
                    edges
                ),
                "gamma_fit": gamma_fit,
                "gamma_published": (
                    gamma_published
                ),
                "gamma_difference": (
                    gamma_difference
                ),
                "gamma_error_percent": (
                    gamma_error_percent
                ),
                "r2": r2,
                "rmse": rmse,
                "plot_file": str(plot_file),
            }
        )

    results = pd.DataFrame(
        results
    )

    # -----------------------------------------------------------------
    # Main comparison table
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 115)
    print("COMPARISON WITH HARTMANN & CIRUNAY (2026), TABLE 1")
    print("=" * 115)

    print(
        f"{'Country':<25}"
        f"{'gamma fit':>14}"
        f"{'gamma paper':>14}"
        f"{'difference':>14}"
        f"{'error %':>12}"
        f"{'R2':>10}"
        f"{'RMSE':>12}"
    )

    print(
        "-" * 115
    )

    for _, row in (
        results.iterrows()
    ):

        print(
            f"{row['country']:<25}"
            f"{row['gamma_fit']:>14.4f}"
            f"{row['gamma_published']:>14.4f}"
            f"{row['gamma_difference']:>14.4f}"
            f"{row['gamma_error_percent']:>12.2f}"
            f"{row['r2']:>10.4f}"
            f"{row['rmse']:>12.4f}"
        )

    # -----------------------------------------------------------------
    # Overall agreement
    # -----------------------------------------------------------------

    mean_abs_error = (
        results[
            "gamma_difference"
        ]
        .abs()
        .mean()
    )

    mean_abs_percent_error = (
        results[
            "gamma_error_percent"
        ]
        .abs()
        .mean()
    )

    median_abs_percent_error = (
        results[
            "gamma_error_percent"
        ]
        .abs()
        .median()
    )

    print("\n")
    print("=" * 115)
    print("OVERALL AGREEMENT")
    print("=" * 115)

    print(
        f"Mean absolute gamma error          : "
        f"{mean_abs_error:.4f}"
    )

    print(
        f"Mean absolute percentage error     : "
        f"{mean_abs_percent_error:.2f} %"
    )

    print(
        f"Median absolute percentage error   : "
        f"{median_abs_percent_error:.2f} %"
    )

    # -----------------------------------------------------------------
    # Compact table for email
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 115)
    print("COMPACT GAMMA TABLE FOR EMAIL")
    print("=" * 115)

    print(
        "| Country | Fitted gamma | Published gamma | Difference | Error (%) |"
    )

    print(
        "|---|---:|---:|---:|---:|"
    )

    for _, row in (
        results.iterrows()
    ):

        print(
            f"| {row['country']} "
            f"| {row['gamma_fit']:.3f} "
            f"| {row['gamma_published']:.3f} "
            f"| {row['gamma_difference']:+.3f} "
            f"| {row['gamma_error_percent']:+.1f} |"
        )

    # -----------------------------------------------------------------
    # Save full results
    # -----------------------------------------------------------------

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print("\n")
    print("=" * 115)
    print("SAVED")
    print("=" * 115)

    print(
        f"Results:\n"
        f"  {RESULTS_FILE}"
    )

    print(
        f"\nCountry comparison plots:\n"
        f"  {PLOT_DIR}"
    )

    print("\n")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
