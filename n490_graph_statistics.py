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

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------

def plot_lines_per_bus_fit(
    statistics: pd.DataFrame,
    extrapolate_to_kv: float = 132,
) -> None:
    """
    Plot N490 lines per bus against voltage and add a linear regression fit.

    The regression is fitted using the voltage levels present in
    ``statistics`` and extrapolated downward to ``extrapolate_to_kv``.

    Parameters
    ----------
    statistics:
        Summary table produced by ``calculate_line_bus_statistics()``.
        Must contain ``Vbase`` and ``lines_per_bus``.
    extrapolate_to_kv:
        Lowest voltage to which the fitted line should be extrapolated.

    Returns
    -------
    None
    """
    required = {"Vbase", "lines_per_bus"}
    missing = required - set(statistics.columns)

    if missing:
        raise ValueError(
            f"Statistics table is missing columns: {sorted(missing)}"
        )

    x = statistics["Vbase"].to_numpy(dtype=float)
    y = statistics["lines_per_bus"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError(
            "At least two voltage levels are required for a linear fit."
        )

    # -------------------------------------------------------------
    # Linear least-squares fit: y = slope * V + intercept
    # -------------------------------------------------------------
    slope, intercept = np.polyfit(
        x,
        y,
        deg=1,
    )

    x_fit = np.linspace(
        float(extrapolate_to_kv),
        float(x.max()),
        200,
    )

    y_fit = slope * x_fit + intercept

    # Predicted value at the extrapolation voltage.
    extrapolated_value = (
        slope * float(extrapolate_to_kv)
        + intercept
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x,
        y,
        s=90,
        zorder=3,
        label="N490",
    )

    ax.plot(
        x_fit,
        y_fit,
        linewidth=2.0,
        label="Linear fit",
    )

    # Explicitly mark the extrapolated 132-kV estimate.
    ax.scatter(
        [extrapolate_to_kv],
        [extrapolated_value],
        s=90,
        marker="x",
        linewidths=2.0,
        zorder=4,
        label=f"{extrapolate_to_kv:g} kV extrapolation",
    )

    ax.set_xlabel("Voltage level (kV)")
    ax.set_ylabel("Lines per bus")

    ax.set_title(
        "N490 lines per bus by voltage level"
    )

    ax.grid(
        alpha=0.3,
    )

    ax.legend()

    # Include the extrapolated voltage explicitly on the x axis.
    voltage_ticks = sorted(
        set(
            [float(extrapolate_to_kv)]
            + x.tolist()
        )
    )

    ax.set_xticks(voltage_ticks)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    print("\nLinear fit")
    print("----------")
    print(f"Slope:                 {slope:.6f} lines/bus/kV")
    print(f"Intercept:             {intercept:.6f}")
    print(
        f"Predicted at "
        f"{extrapolate_to_kv:g} kV:     "
        f"{extrapolated_value:.4f} lines/bus"
    )
    
def plot_line_to_mst_length_ratio_fit(
    length_statistics: pd.DataFrame,
    extrapolate_to_kv: float = 132,
) -> None:
    """
    Plot N490 total-line-length / MST-length ratio against voltage and add
    a linear regression fit.

    The regression is fitted using the voltage levels present in
    ``length_statistics`` and extrapolated downward to
    ``extrapolate_to_kv``.

    Parameters
    ----------
    length_statistics:
        Summary table produced by
        ``calculate_line_to_mst_length_ratio()``. Must contain ``Vbase`` and
        ``line_to_mst_length_ratio``.
    extrapolate_to_kv:
        Lowest voltage to which the fitted line should be extrapolated.

    Returns
    -------
    None
    """
    required = {
        "Vbase",
        "line_to_mst_length_ratio",
    }

    missing = required - set(length_statistics.columns)

    if missing:
        raise ValueError(
            f"Length-statistics table is missing columns: {sorted(missing)}"
        )

    x = length_statistics["Vbase"].to_numpy(dtype=float)

    y = length_statistics[
        "line_to_mst_length_ratio"
    ].to_numpy(dtype=float)

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

    x_fit = np.linspace(
        float(extrapolate_to_kv),
        float(x.max()),
        200,
    )

    y_fit = (
        slope * x_fit
        + intercept
    )

    extrapolated_value = (
        slope * float(extrapolate_to_kv)
        + intercept
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x,
        y,
        s=90,
        zorder=3,
        label="N490",
    )

    ax.plot(
        x_fit,
        y_fit,
        linewidth=2.0,
        label="Linear fit",
    )

    ax.scatter(
        [extrapolate_to_kv],
        [extrapolated_value],
        s=90,
        marker="x",
        linewidths=2.0,
        zorder=4,
        label=f"{extrapolate_to_kv:g} kV extrapolation",
    )

    ax.set_xlabel("Voltage level (kV)")

    ax.set_ylabel(
        "Total line length / MST length"
    )

    ax.set_title(
        "N490 total line length relative to MST length"
    )

    ax.grid(
        alpha=0.3,
    )

    ax.legend()

    voltage_ticks = sorted(
        set(
            [float(extrapolate_to_kv)]
            + x.tolist()
        )
    )

    ax.set_xticks(
        voltage_ticks
    )

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    print("\nLinear fit: line length / MST length")
    print("------------------------------------")

    print(
        f"Slope:                 "
        f"{slope:.6f} ratio/kV"
    )

    print(
        f"Intercept:             "
        f"{intercept:.6f}"
    )

    print(
        f"Predicted at "
        f"{extrapolate_to_kv:g} kV:     "
        f"{extrapolated_value:.4f}"
    )
    

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
    
    plot_lines_per_bus_fit(
        statistics=statistics,
        extrapolate_to_kv=132,
    )
    
    plot_line_to_mst_length_ratio_fit(
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

    statistics.to_pickle(
        OUTPUT_DIR
        / "N490_line_bus_statistics.pkl"
    )

    print(
        "\nSaved:"
        f"\n  {OUTPUT_DIR / 'N490_line_bus_statistics.pkl'}"
    )


if __name__ == "__main__":
    main()