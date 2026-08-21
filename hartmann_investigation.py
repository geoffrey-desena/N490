#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
European HV node-degree analysis: free-C cumulative exponential fit
==================================================================

Loads:
    euro-comparison/european_networks.pkl

For each complete country network:

    1. Calculate node degrees, retaining parallel circuits.
    2. Calculate the complementary cumulative degree distribution:

           P(K >= k)

    3. Fit:

           P(K >= k) = C * exp(-k / gamma)

       with both C and gamma free.

    4. Compare the fitted gamma with the published value in
       Hartmann & Cirunay (2026), Table 1:

           "gamma of HV grid exp. fit"

No voltage-specific subnetworks are analyzed in this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


# =====================================================================
# PATHS
# =====================================================================

WORKING_DIR = Path.cwd()

DATA_DIR = WORKING_DIR / "euro-comparison"

INPUT_FILE = DATA_DIR / "european_networks.pkl"


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

def exponential_cumulative(k, C, gamma):
    """
    Free-C exponential model:

        P(K >= k) = C * exp(-k / gamma)
    """

    return C * np.exp(-k / gamma)


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
# FIT
# =====================================================================

def fit_free_C_exponential(k, probability):
    """
    Fit:

        P(K >= k) = C * exp(-k / gamma)

    with both C and gamma free.

    Returns
    -------
    C : float
    gamma : float
    r2 : float
    """

    if len(k) < 2:
        return np.nan, np.nan, np.nan

    # -------------------------------------------------------------
    # Initial guesses
    #
    # Since P(K>=1) = 1:
    #
    #     C * exp(-1/gamma) ~= 1
    #
    # A C slightly above 1 is therefore a reasonable starting point.
    # -------------------------------------------------------------

    C_initial = 1.5
    gamma_initial = 2.0

    try:

        popt, _ = curve_fit(
            exponential_cumulative,
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

        return np.nan, np.nan, np.nan

    # -------------------------------------------------------------
    # Fitted values
    # -------------------------------------------------------------

    fitted = exponential_cumulative(
        k,
        C,
        gamma,
    )

    # -------------------------------------------------------------
    # R^2
    # -------------------------------------------------------------

    ss_res = np.sum(
        (probability - fitted) ** 2
    )

    ss_tot = np.sum(
        (probability - np.mean(probability)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return C, gamma, r2


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("\n")
    print("=" * 110)
    print("FREE-C CUMULATIVE EXPONENTIAL FIT")
    print("=" * 110)

    print(f"\nLoading:")
    print(f"  {INPUT_FILE}")

    euro_networks = pd.read_pickle(
        INPUT_FILE
    )

    results = []

    # -----------------------------------------------------------------
    # Analyze complete network for each country
    # -----------------------------------------------------------------

    for country in sorted(euro_networks):

        edges = euro_networks[country]

        degrees = calculate_node_degrees(
            edges
        )

        k, probability = (
            calculate_cumulative_distribution(
                degrees
            )
        )

        C, gamma, r2 = (
            fit_free_C_exponential(
                k,
                probability,
            )
        )

        published_gamma = (
            PUBLISHED_GAMMA[country]
        )

        gamma_difference = (
            gamma - published_gamma
        )

        gamma_error_percent = (
            100.0
            * gamma_difference
            / published_gamma
        )

        results.append(
            {
                "country": country,
                "n_nodes": len(degrees),
                "n_branches": len(edges),
                "mean_degree": degrees.mean(),
                "C": C,
                "gamma_fit": gamma,
                "gamma_published": published_gamma,
                "difference": gamma_difference,
                "error_percent": gamma_error_percent,
                "r2": r2,
            }
        )

    results = pd.DataFrame(
        results
    )

    # -----------------------------------------------------------------
    # Print comparison
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("COMPARISON WITH HARTMANN & CIRUNAY (2026), TABLE 1")
    print("=" * 110)

    print(
        f"{'Country':<25}"
        f"{'C':>10}"
        f"{'gamma fit':>14}"
        f"{'gamma paper':>14}"
        f"{'difference':>14}"
        f"{'error %':>12}"
        f"{'R2':>10}"
    )

    print("-" * 110)

    for _, row in results.iterrows():

        print(
            f"{row['country']:<25}"
            f"{row['C']:>10.4f}"
            f"{row['gamma_fit']:>14.4f}"
            f"{row['gamma_published']:>14.4f}"
            f"{row['difference']:>14.4f}"
            f"{row['error_percent']:>12.2f}"
            f"{row['r2']:>10.4f}"
        )

    # -----------------------------------------------------------------
    # Overall error diagnostics
    # -----------------------------------------------------------------

    mean_abs_error = (
        results["difference"]
        .abs()
        .mean()
    )

    mean_abs_percent_error = (
        results["error_percent"]
        .abs()
        .mean()
    )

    print("\n")
    print("=" * 110)
    print("OVERALL AGREEMENT")
    print("=" * 110)

    print(
        f"Mean absolute gamma error       : "
        f"{mean_abs_error:.4f}"
    )

    print(
        f"Mean absolute percentage error  : "
        f"{mean_abs_percent_error:.2f} %"
    )

    print("\n")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()