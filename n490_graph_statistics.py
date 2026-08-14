# -*- coding: utf-8 -*-
"""
Calculate graph statistics for the Nordic490 transmission network.

The analysis is performed independently for the 220, 300, and 380 kV
networks.

Current statistics
------------------
For each voltage level, calculate:

- number of buses,
- number of AC transmission lines,
- lines per bus, L / N,
- average nodal degree, 2L / N.

Parallel transmission lines are counted separately because they are separate
branches in the N490 network.

Outputs
-------
The script prints the summary table and saves it as:

    n490_graph_statistics/N490_line_bus_statistics.pkl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from shapely.geometry import LineString

from nordic490 import N490


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DELAUNAY_DIR = Path("n490_delaunay_analysis")

SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:3845"

OUTPUT_DIR = Path("n490_graph_statistics")

VOLTAGE_LEVELS = [220, 300, 380]

BASE_FONTSIZE = plt.rcParams["font.size"] * 1.50

DELAUNAY_ANALYSIS_DIR = Path(
    "n490_delaunay_overlap_analysis"
)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def calculate_line_bus_statistics(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> pd.DataFrame:
    """
    Calculate line-to-bus statistics for each voltage network.

    Parameters
    ----------
    buses:
        N490 bus table containing ``Vbase``.
    lines:
        N490 AC line table containing ``Vbase``.
    voltage_levels:
        Voltage levels to analyze.

    Returns
    -------
    pandas.DataFrame
        Summary containing:

        - ``Vbase``
        - ``n_buses``
        - ``n_lines``
        - ``lines_per_bus``
        - ``average_degree``

    Notes
    -----
    For an undirected graph with N buses and L branches,

        lines_per_bus = L / N

    whereas the average nodal degree is

        average_degree = 2L / N

    because every branch is incident on two buses.

    Parallel N490 lines are counted as separate branches.
    """
    if "Vbase" not in buses.columns:
        raise ValueError(
            "N490 bus table does not contain 'Vbase'."
        )

    if "Vbase" not in lines.columns:
        raise ValueError(
            "N490 line table does not contain 'Vbase'."
        )

    bus_voltage = pd.to_numeric(
        buses["Vbase"],
        errors="coerce",
    )

    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    rows = []

    for voltage_kv in voltage_levels:

        n_buses = int(
            np.isclose(
                bus_voltage,
                float(voltage_kv),
                equal_nan=False,
            ).sum()
        )

        n_lines = int(
            np.isclose(
                line_voltage,
                float(voltage_kv),
                equal_nan=False,
            ).sum()
        )

        if n_buses == 0:
            raise ValueError(
                f"No N490 buses found at {voltage_kv} kV."
            )

        lines_per_bus = (
            n_lines / n_buses
        )

        average_degree = (
            2.0 * n_lines / n_buses
        )

        rows.append(
            {
                "Vbase": int(voltage_kv),
                "n_buses": n_buses,
                "n_lines": n_lines,
                "lines_per_bus": float(lines_per_bus),
                "average_degree": float(average_degree),
            }
        )

    return pd.DataFrame(rows)

def calculate_line_to_mst_length_ratio(
    lines: pd.DataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
    delaunay_dir: Path = DELAUNAY_DIR,
    source_crs: str = SOURCE_CRS,
    target_crs: str = TARGET_CRS,
) -> pd.DataFrame:
    """
    Calculate total N490 line length divided by total MST length at each
    voltage level.

    Parameters
    ----------
    lines:
        N490 line table containing ``Vbase``, ``lat``, and ``lon``.
    voltage_levels:
        Voltage levels to analyze.
    delaunay_dir:
        Directory containing the voltage-specific MST GeoJSON files produced
        by ``N490_delaunay_creation.py``.
    source_crs:
        CRS of the N490 latitude/longitude coordinates.
    target_crs:
        Projected metric CRS used for length calculations.

    Returns
    -------
    pandas.DataFrame
        Table containing:

        - ``Vbase``
        - ``n490_total_length_km``
        - ``mst_total_length_km``
        - ``line_to_mst_length_ratio``

    Notes
    -----
    N490 line length is calculated from the full supplied branch geometry,
    rather than simply using the straight-line distance between terminal
    buses.

    Parallel N490 branches are counted separately, so their lengths contribute
    independently to the total network length.
    """
    required = {
        "Vbase",
        "lat",
        "lon",
    }

    missing = required - set(lines.columns)

    if missing:
        raise ValueError(
            f"N490 line table is missing columns: {sorted(missing)}"
        )

    line_voltage = pd.to_numeric(
        lines["Vbase"],
        errors="coerce",
    )

    rows = []

    for voltage_kv in voltage_levels:

        # ---------------------------------------------------------
        # Actual N490 line length
        # ---------------------------------------------------------
        voltage_lines = lines[
            np.isclose(
                line_voltage,
                float(voltage_kv),
                equal_nan=False,
            )
        ].copy()

        if voltage_lines.empty:
            raise ValueError(
                f"No N490 lines found at {voltage_kv} kV."
            )

        geometries = []

        for lats, lons in zip(
            voltage_lines["lat"],
            voltage_lines["lon"],
        ):
            if lats is None or lons is None:
                continue

            points = [
                (float(lon), float(lat))
                for lat, lon in zip(lats, lons)
                if pd.notna(lat) and pd.notna(lon)
            ]

            if len(points) < 2:
                continue

            geometries.append(
                LineString(points)
            )

        if not geometries:
            raise ValueError(
                f"No valid {voltage_kv} kV N490 line geometries."
            )

        n490_gdf = gpd.GeoDataFrame(
            geometry=geometries,
            crs=source_crs,
        ).to_crs(target_crs)

        n490_total_length_km = float(
            n490_gdf.geometry.length.sum()
            / 1000.0
        )

        # ---------------------------------------------------------
        # Voltage-specific MST length
        # ---------------------------------------------------------
        mst_path = (
            delaunay_dir
            / f"n490_{voltage_kv}kv_mst.geojson"
        )

        if not mst_path.exists():
            raise FileNotFoundError(
                f"Could not find MST file: {mst_path}"
            )

        mst = gpd.read_file(
            mst_path
        )

        if mst.empty:
            raise ValueError(
                f"MST file is empty: {mst_path}"
            )

        if "length_km" in mst.columns:
            mst_total_length_km = float(
                pd.to_numeric(
                    mst["length_km"],
                    errors="raise",
                ).sum()
            )

        else:
            if mst.crs is None:
                raise ValueError(
                    f"MST file has no CRS: {mst_path}"
                )

            mst_projected = mst.to_crs(
                target_crs
            )

            mst_total_length_km = float(
                mst_projected.geometry.length.sum()
                / 1000.0
            )

        if mst_total_length_km <= 0:
            raise ValueError(
                f"Invalid MST length at {voltage_kv} kV."
            )

        ratio = (
            n490_total_length_km
            / mst_total_length_km
        )

        rows.append(
            {
                "Vbase": int(voltage_kv),
                "n490_total_length_km": n490_total_length_km,
                "mst_total_length_km": mst_total_length_km,
                "line_to_mst_length_ratio": float(ratio),
            }
        )

    return pd.DataFrame(rows)

def build_final_statistics_summary(
    statistics: pd.DataFrame,
    length_statistics: pd.DataFrame,
    edge_fit: dict,
    length_fit: dict,
    empirical_delaunay_path: Path,
    delaunay_132_path: Path,
) -> pd.DataFrame:
    """
    Combine empirical N490 statistics with fitted 132 kV estimates.

    The 220, 300, and 380 kV rows use measured N490 values throughout.

    The 132 kV row uses:
    - exponential-fit estimates for MST/Delaunay overlap proportions,
    - linear-fit extrapolation for lines per bus,
    - linear-fit extrapolation for total-line-length / MST-length ratio.
    """
    empirical_delaunay = pd.read_pickle(
        empirical_delaunay_path
    )

    overlap_132 = pd.read_pickle(
        delaunay_132_path
    )

    delaunay_columns = [
        "Vbase",
        "MST",
        "Delaunay1",
        "Delaunay2",
        "Delaunay3",
    ]

    # -------------------------------------------------------------
    # Empirical 220/300/380 Delaunay statistics
    # -------------------------------------------------------------
    empirical_delaunay = (
        empirical_delaunay[
            delaunay_columns
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Fitted 132 kV Delaunay statistics
    # -------------------------------------------------------------
    overlap_132 = (
        overlap_132[
            delaunay_columns
        ]
        .copy()
    )

    delaunay_all = pd.concat(
        [
            overlap_132,
            empirical_delaunay,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Empirical line-count statistic
    # -------------------------------------------------------------
    graph_stats = statistics[
        [
            "Vbase",
            "lines_per_bus",
        ]
    ].copy()

    # -------------------------------------------------------------
    # Empirical length/MST statistic
    # -------------------------------------------------------------
    length_stats = length_statistics[
        [
            "Vbase",
            "line_to_mst_length_ratio",
        ]
    ].copy()

    graph_stats = graph_stats.merge(
        length_stats,
        on="Vbase",
        how="outer",
        validate="one_to_one",
    )

    # -------------------------------------------------------------
    # Add fitted 132 kV graph statistics
    # -------------------------------------------------------------
    row_132 = pd.DataFrame(
        [
            {
                "Vbase": 132,
                "line_to_mst_length_ratio": float(
                    length_fit["extrapolated_value"]
                ),
                "lines_per_bus": float(
                    edge_fit["extrapolated_value"]
                ),
            }
        ]
    )

    graph_stats = pd.concat(
        [
            row_132,
            graph_stats,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Final table
    # -------------------------------------------------------------
    summary = delaunay_all.merge(
        graph_stats,
        on="Vbase",
        how="inner",
        validate="one_to_one",
    )

    summary = (
        summary[
            [
                "Vbase",
                "MST",
                "Delaunay1",
                "Delaunay2",
                "Delaunay3",
                "line_to_mst_length_ratio",
                "lines_per_bus",
            ]
        ]
        .sort_values("Vbase")
        .reset_index(drop=True)
    )

    return summary

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------

def fit_linear_voltage_series(
    data: pd.DataFrame,
    value_column: str,
    extrapolate_to_kv: float = 132.0,
) -> dict:
    """
    Fit a linear relationship between voltage and one graph statistic.

    The fitted model is

        y = slope * V + intercept

    where V is voltage in kV.

    R-squared and RMSE are calculated using only the empirical voltage
    levels. The fitted relationship is then extrapolated downward to
    ``extrapolate_to_kv``.

    Parameters
    ----------
    data:
        DataFrame containing ``Vbase`` and ``value_column``.
    value_column:
        Name of the statistic to fit.
    extrapolate_to_kv:
        Voltage at which to evaluate the extrapolated fitted value.

    Returns
    -------
    dict
        Dictionary containing the fitted parameters, goodness-of-fit
        statistics, empirical data, and extrapolated value.
    """
    required = {"Vbase", value_column}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Input table is missing columns: {sorted(missing)}"
        )

    x = data["Vbase"].to_numpy(dtype=float)
    y = data[value_column].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError(
            "At least two voltage levels are required for a linear fit."
        )

    # -------------------------------------------------------------
    # Linear least-squares fit
    # -------------------------------------------------------------
    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    y_pred = slope * x + intercept
    residuals = y - y_pred

    # -------------------------------------------------------------
    # Goodness of fit
    # -------------------------------------------------------------
    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r_squared = (
        1.0 - ss_res / ss_tot
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

    # -------------------------------------------------------------
    # Extrapolated value
    # -------------------------------------------------------------
    extrapolated_value = (
        slope * float(extrapolate_to_kv)
        + intercept
    )

    return {
        "x": x,
        "y": y,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "extrapolate_to_kv": float(extrapolate_to_kv),
        "extrapolated_value": float(extrapolated_value),
    }


def plot_graph_statistics_fits(
    statistics: pd.DataFrame,
    length_statistics: pd.DataFrame,
    extrapolate_to_kv: float = 132.0,
) -> tuple[dict, dict]:
    """
    Plot edges per node and total-line-length/MST-length ratio together.

    Both statistics are fitted independently as linear functions of voltage.
    Empirical observations are shown as filled circular markers and the
    extrapolated value as an ``x`` marker of the same series color.

    Parameters
    ----------
    statistics:
        Output from ``calculate_line_bus_statistics()``.
    length_statistics:
        Output from ``calculate_line_to_mst_length_ratio()``.
    extrapolate_to_kv:
        Lowest voltage shown and voltage at which the fitted relationships
        are extrapolated.

    Returns
    -------
    edge_fit:
        Fit results for edges per node.
    length_fit:
        Fit results for total line length divided by MST length.
    """
    # -------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------
    EDGE_COLOR = "#900069"
    LENGTH_COLOR = "#5c7c22"
    BLACK = "#000000"

    # -------------------------------------------------------------
    # Fit both statistics
    # -------------------------------------------------------------
    edge_fit = fit_linear_voltage_series(
        data=statistics,
        value_column="lines_per_bus",
        extrapolate_to_kv=extrapolate_to_kv,
    )

    length_fit = fit_linear_voltage_series(
        data=length_statistics,
        value_column="line_to_mst_length_ratio",
        extrapolate_to_kv=extrapolate_to_kv,
    )

    # Common fitted-voltage range.
    max_voltage = max(
        edge_fit["x"].max(),
        length_fit["x"].max(),
    )

    x_fit = np.linspace(
        float(extrapolate_to_kv),
        float(max_voltage),
        300,
    )

    edge_y_fit = (
        edge_fit["slope"] * x_fit
        + edge_fit["intercept"]
    )

    length_y_fit = (
        length_fit["slope"] * x_fit
        + length_fit["intercept"]
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # -------------------------------------------------------------
    # Edges per node
    # -------------------------------------------------------------
    ax.plot(
        x_fit,
        edge_y_fit,
        color=EDGE_COLOR,
        linewidth=2.2,
        label="Edges per node",
        zorder=2,
    )

    ax.scatter(
        edge_fit["x"],
        edge_fit["y"],
        color=EDGE_COLOR,
        s=75,
        marker="o",
        zorder=4,
    )

    ax.scatter(
        [extrapolate_to_kv],
        [edge_fit["extrapolated_value"]],
        color=EDGE_COLOR,
        s=90,
        marker="x",
        linewidths=2.2,
        zorder=5,
    )
    
    # -------------------------------------------------------------
    # Extrapolated-value labels at 132 kV
    # -------------------------------------------------------------
    
    # Edges per node: place above the extrapolated point.
    ax.annotate(
        rf"{edge_fit['extrapolated_value']:.4f}",
        xy=(
            extrapolate_to_kv,
            edge_fit["extrapolated_value"],
        ),
        xytext=(10, 13),
        textcoords="offset points",
        horizontalalignment="center",
        verticalalignment="bottom",
        color=EDGE_COLOR,
        fontsize=BASE_FONTSIZE,
    )
    
    # Total length / MST: place to the right of the extrapolated point.
    ax.annotate(
        rf"{length_fit['extrapolated_value']:.4f}",
        xy=(
            extrapolate_to_kv,
            length_fit["extrapolated_value"],
        ),
        xytext=(18, 0),
        textcoords="offset points",
        horizontalalignment="left",
        verticalalignment="center",
        color=LENGTH_COLOR,
        fontsize=BASE_FONTSIZE,
    )

    # -------------------------------------------------------------
    # Total length / MST
    # -------------------------------------------------------------
    ax.plot(
        x_fit,
        length_y_fit,
        color=LENGTH_COLOR,
        linewidth=2.2,
        label="Ratio: total length to MST",
        zorder=2,
    )

    ax.scatter(
        length_fit["x"],
        length_fit["y"],
        color=LENGTH_COLOR,
        s=75,
        marker="o",
        zorder=4,
    )

    ax.scatter(
        [extrapolate_to_kv],
        [length_fit["extrapolated_value"]],
        color=LENGTH_COLOR,
        s=90,
        marker="x",
        linewidths=2.2,
        zorder=5,
    )

    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------
    ax.set_xlabel(
        "Voltage level [kV]",
        color=BLACK,
        fontsize=BASE_FONTSIZE,
    )
    
    ax.tick_params(
        axis="both",
        colors=BLACK,
        labelsize=BASE_FONTSIZE,
    )

    # Deliberately no y-axis label.
    ax.set_ylabel("")

    # No title.
    ax.set_title("")

    # No grid.
    ax.grid(False)

    # Only primary x and y axis lines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_color(BLACK)

    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)

    # Explicit voltage ticks including extrapolated 132 kV point.
    voltage_ticks = sorted(
        set(
            [float(extrapolate_to_kv)]
            + edge_fit["x"].tolist()
            + length_fit["x"].tolist()
        )
    )

    ax.set_xticks(
        voltage_ticks
    )
    
    # -------------------------------------------------------------
    # Direct series labels at 300 kV
    # -------------------------------------------------------------
    edge_300 = statistics.loc[
        statistics["Vbase"] == 300,
        "lines_per_bus",
    ].iloc[0]
    
    length_300 = length_statistics.loc[
        length_statistics["Vbase"] == 300,
        "line_to_mst_length_ratio",
    ].iloc[0]
    
    ax.annotate(
        "Edges per node",
        xy=(300, edge_300),
        xytext=(10, -12),
        textcoords="offset points",
        horizontalalignment="left",
        verticalalignment="top",
        color=EDGE_COLOR,
        fontsize=BASE_FONTSIZE,
    )
    
    ax.annotate(
        "Ratio: total length to MST",
        xy=(300, length_300),
        xytext=(-10, 12),
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="bottom",
        color=LENGTH_COLOR,
        fontsize=BASE_FONTSIZE,
    )


    # -------------------------------------------------------------
    # Fit-statistics annotations
    #
    # Edges-per-node statistics above length/MST statistics.
    # Both are placed on the far right.
    # -------------------------------------------------------------
    edge_text = (
        rf"$y={edge_fit['slope']:.5f}V"
        rf"{edge_fit['intercept']:+.3f}$"
        "\n"
        rf"$R^2={edge_fit['r_squared']:.4f}$"
        "\n"
        rf"RMSE = {edge_fit['rmse']:.4f}"
    )
    
    length_text = (
        rf"$y={length_fit['slope']:.5f}V"
        rf"{length_fit['intercept']:+.3f}$"
        "\n"
        rf"$R^2={length_fit['r_squared']:.4f}$"
        "\n"
        rf"RMSE = {length_fit['rmse']:.4f}"
    )

    ax.text(
        0.98,
        0.10,
        edge_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        color=EDGE_COLOR,
        fontsize=BASE_FONTSIZE,
    )

    ax.text(
        0.98,
        0.68,
        length_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color=LENGTH_COLOR,
        fontsize=BASE_FONTSIZE,
    )

    # Make all remaining standard text explicitly black.
    ax.xaxis.label.set_color(BLACK)
    ax.xaxis.label.set_fontsize(BASE_FONTSIZE)
    
    for tick in ax.get_xticklabels():
        tick.set_color(BLACK)
        tick.set_fontsize(BASE_FONTSIZE)
    
    for tick in ax.get_yticklabels():
        tick.set_color(BLACK)
        tick.set_fontsize(BASE_FONTSIZE)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Console diagnostics
    # -------------------------------------------------------------
    print("\nLinear fit: edges per node")
    print("--------------------------")
    print(
        f"Slope:                 "
        f"{edge_fit['slope']:.6f} edges/node/kV"
    )
    print(
        f"Intercept:             "
        f"{edge_fit['intercept']:.6f}"
    )
    print(
        f"R-squared:             "
        f"{edge_fit['r_squared']:.6f}"
    )
    print(
        f"RMSE:                  "
        f"{edge_fit['rmse']:.6f}"
    )
    print(
        f"Predicted at "
        f"{extrapolate_to_kv:g} kV:    "
        f"{edge_fit['extrapolated_value']:.4f}"
    )

    print("\nLinear fit: total length / MST")
    print("--------------------------------")
    print(
        f"Slope:                 "
        f"{length_fit['slope']:.6f} ratio/kV"
    )
    print(
        f"Intercept:             "
        f"{length_fit['intercept']:.6f}"
    )
    print(
        f"R-squared:             "
        f"{length_fit['r_squared']:.6f}"
    )
    print(
        f"RMSE:                  "
        f"{length_fit['rmse']:.6f}"
    )
    print(
        f"Predicted at "
        f"{extrapolate_to_kv:g} kV:    "
        f"{length_fit['extrapolated_value']:.4f}"
    )

    return edge_fit, length_fit

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Calculate, print, and save voltage-specific N490 graph statistics.
    """
    model = N490(year=2018)

    buses = model.bus.copy()
    lines = model.line.copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = calculate_line_bus_statistics(
        buses=buses,
        lines=lines,
    )
    
    length_statistics = calculate_line_to_mst_length_ratio(
        lines=lines,
    )
    
    print("\n")
    print("=" * 72)
    print("N490 total line length relative to MST length")
    print("=" * 72)
    
    print(
        length_statistics
        .round(
            {
                "n490_total_length_km": 2,
                "mst_total_length_km": 2,
                "line_to_mst_length_ratio": 4,
            }
        )
        .to_string(index=False)
    )
    
    length_statistics.to_pickle(
        OUTPUT_DIR
        / "N490_line_mst_length_statistics.pkl"
    )
    
    print(
        "\nSaved:"
        f"\n  {OUTPUT_DIR / 'N490_line_mst_length_statistics.pkl'}"
    )
    
    edge_fit, length_fit = plot_graph_statistics_fits(
        statistics=statistics,
        length_statistics=length_statistics,
        extrapolate_to_kv=132,
    )
    print("\n")
    print("=" * 72)
    print("N490 line-to-bus statistics by voltage")
    print("=" * 72)

    print(
        statistics
        .round(
            {
                "lines_per_bus": 4,
                "average_degree": 4,
            }
        )
        .to_string(index=False)
    )
    
    # -------------------------------------------------------------
    # Final combined statistics summary
    # -------------------------------------------------------------
    final_summary = build_final_statistics_summary(
        statistics=statistics,
        length_statistics=length_statistics,
        edge_fit=edge_fit,
        length_fit=length_fit,
        empirical_delaunay_path=(
            DELAUNAY_ANALYSIS_DIR
            / "N490_Delaunay_stats_by_voltage.pkl"
        ),
        delaunay_132_path=(
            DELAUNAY_ANALYSIS_DIR
            / "N490_Delaunay_132kv_estimate.pkl"
        ),
    )
    
    print("\n")
    print("=" * 96)
    print("N490 summary statistics")
    print("=" * 96)
    
    print(
        final_summary
        .round(
            {
                "MST": 4,
                "Delaunay1": 4,
                "Delaunay2": 4,
                "Delaunay3": 4,
                "line_to_mst_length_ratio": 4,
                "lines_per_bus": 4,
            }
        )
        .to_string(index=False)
    )
    
    final_summary.to_pickle(
        OUTPUT_DIR
        / "N490_summary_statistics.pkl"
    )
    
    print(
        "\nSaved:"
        f"\n  {OUTPUT_DIR / 'N490_summary_statistics.pkl'}"
    )



if __name__ == "__main__":
    main()