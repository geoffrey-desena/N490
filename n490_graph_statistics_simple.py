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

All graph statistics are calculated from a simple-graph representation of
each voltage network: parallel circuits between the same unordered bus pair
are collapsed to one edge.

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
# Simple-graph construction
# ---------------------------------------------------------------------

def build_simple_graph_lines(
    lines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse parallel AC circuits to one edge per unordered bus pair.

    The first branch encountered for each ``(Vbase, bus pair)`` is retained as
    the representative edge. Its supplied geographic geometry is therefore
    used when calculating the total physical length of the simple graph.

    Returns
    -------
    simple_lines:
        Copy of ``lines`` with parallel circuits removed.
    diagnostics:
        One row per voltage containing original edge count, simple edge count,
        and number of parallel circuits removed.
    """
    if "Vbase" not in lines.columns:
        raise ValueError("N490 line table does not contain 'Vbase'.")

    from_col, to_col = _resolve_line_endpoint_columns(lines)

    work = lines.copy()
    work["_voltage_numeric"] = pd.to_numeric(
        work["Vbase"],
        errors="coerce",
    )

    def canonical_pair(row: pd.Series) -> tuple:
        a = row[from_col]
        b = row[to_col]
        if pd.isna(a) or pd.isna(b):
            return (a, b)
        return tuple(sorted((a, b), key=lambda value: str(value)))

    work["_simple_pair"] = work.apply(
        canonical_pair,
        axis=1,
    )

    duplicate_mask = work.duplicated(
        subset=["_voltage_numeric", "_simple_pair"],
        keep="first",
    )

    simple_lines = (
        work.loc[~duplicate_mask]
        .drop(columns=["_voltage_numeric", "_simple_pair"])
        .copy()
    )

    rows = []
    original_voltage = pd.to_numeric(lines["Vbase"], errors="coerce")
    simple_voltage = pd.to_numeric(simple_lines["Vbase"], errors="coerce")

    for voltage_kv in VOLTAGE_LEVELS:
        n_original = int(
            np.isclose(original_voltage, float(voltage_kv), equal_nan=False).sum()
        )
        n_simple = int(
            np.isclose(simple_voltage, float(voltage_kv), equal_nan=False).sum()
        )
        rows.append(
            {
                "Vbase": int(voltage_kv),
                "n_lines_original": n_original,
                "n_lines_simple": n_simple,
                "n_parallel_removed": n_original - n_simple,
            }
        )

    return simple_lines, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def calculate_line_bus_statistics(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> pd.DataFrame:
    """Calculate edge-to-node statistics for simple voltage graphs.

    ``lines`` is expected to contain one edge per unordered terminal-bus pair.
    Thus ``lines_per_bus`` is E/N for the simple graph and
    ``average_degree`` is 2E/N.
    """
    if "Vbase" not in buses.columns:
        raise ValueError("N490 bus table does not contain 'Vbase'.")
    if "Vbase" not in lines.columns:
        raise ValueError("N490 line table does not contain 'Vbase'.")

    bus_voltage = pd.to_numeric(buses["Vbase"], errors="coerce")
    line_voltage = pd.to_numeric(lines["Vbase"], errors="coerce")

    rows = []
    for voltage_kv in voltage_levels:
        n_buses = int(
            np.isclose(bus_voltage, float(voltage_kv), equal_nan=False).sum()
        )
        n_lines = int(
            np.isclose(line_voltage, float(voltage_kv), equal_nan=False).sum()
        )

        if n_buses == 0:
            raise ValueError(f"No N490 buses found at {voltage_kv} kV.")

        rows.append(
            {
                "Vbase": int(voltage_kv),
                "n_buses": n_buses,
                "n_edges_simple": n_lines,
                "lines_per_bus": float(n_lines / n_buses),
                "average_degree": float(2.0 * n_lines / n_buses),
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
    """Calculate total simple-graph line length divided by MST length.

    ``lines`` must already be reduced to one representative branch per
    unordered terminal-bus pair. Each retained edge contributes its full
    supplied route geometry to the total line length.
    """
    required = {"Vbase", "lat", "lon"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(
            f"N490 line table is missing columns: {sorted(missing)}"
        )

    line_voltage = pd.to_numeric(lines["Vbase"], errors="coerce")
    rows = []

    for voltage_kv in voltage_levels:
        voltage_lines = lines[
            np.isclose(line_voltage, float(voltage_kv), equal_nan=False)
        ].copy()

        if voltage_lines.empty:
            raise ValueError(f"No N490 lines found at {voltage_kv} kV.")

        geometries = []
        for lats, lons in zip(voltage_lines["lat"], voltage_lines["lon"]):
            if lats is None or lons is None:
                continue

            points = [
                (float(lon), float(lat))
                for lat, lon in zip(lats, lons)
                if pd.notna(lat) and pd.notna(lon)
            ]
            if len(points) >= 2:
                geometries.append(LineString(points))

        if not geometries:
            raise ValueError(
                f"No valid {voltage_kv} kV N490 line geometries."
            )

        n490_gdf = gpd.GeoDataFrame(
            geometry=geometries,
            crs=source_crs,
        ).to_crs(target_crs)

        n490_total_length_km = float(
            n490_gdf.geometry.length.sum() / 1000.0
        )

        mst_path = delaunay_dir / f"n490_{voltage_kv}kv_mst.geojson"
        if not mst_path.exists():
            raise FileNotFoundError(f"Could not find MST file: {mst_path}")

        mst = gpd.read_file(mst_path)
        if mst.empty:
            raise ValueError(f"MST file is empty: {mst_path}")

        if "length_km" in mst.columns:
            mst_total_length_km = float(
                pd.to_numeric(mst["length_km"], errors="raise").sum()
            )
        else:
            if mst.crs is None:
                raise ValueError(f"MST file has no CRS: {mst_path}")
            mst_total_length_km = float(
                mst.to_crs(target_crs).geometry.length.sum() / 1000.0
            )

        if mst_total_length_km <= 0:
            raise ValueError(f"Invalid MST length at {voltage_kv} kV.")

        rows.append(
            {
                "Vbase": int(voltage_kv),
                "n_edges_simple": int(len(voltage_lines)),
                "n490_total_length_km": n490_total_length_km,
                "mst_total_length_km": mst_total_length_km,
                "line_to_mst_length_ratio": float(
                    n490_total_length_km / mst_total_length_km
                ),
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
    - linear-fit extrapolation for total-line-length / MST-length ratio.

    ``lines_per_bus`` is deliberately left undefined (NaN) at 132 kV.
    The edge-per-node voltage fit is retained only as a diagnostic because
    comparison-network analysis showed that extrapolating E/N by voltage is
    not defensible.
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
                "lines_per_bus": np.nan,
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


def _resolve_line_endpoint_columns(
    lines: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the two bus-endpoint columns in the N490 line table.

    Returns
    -------
    tuple[str, str]
        Names of the from-bus and to-bus columns.
    """
    candidate_pairs = [
        ("bus0", "bus1"),
        ("from_bus", "to_bus"),
        ("from_bus_id", "to_bus_id"),
        ("bus1", "bus2"),
        ("fbus", "tbus"),
        ("from", "to"),
    ]

    for from_col, to_col in candidate_pairs:
        if from_col in lines.columns and to_col in lines.columns:
            return from_col, to_col

    raise ValueError(
        "Could not identify line endpoint columns.\n"
        f"Available line columns are:\n{lines.columns.tolist()}\n\n"
        "Add the appropriate endpoint-column pair to "
        "_resolve_line_endpoint_columns()."
    )


def _resolve_bus_ids(
    buses: pd.DataFrame,
    endpoint_values: pd.Index,
) -> pd.Series:
    """
    Determine which bus identifier corresponds to the line endpoint values.

    First tries the DataFrame index, then common bus-ID column names.
    """
    endpoint_values = pd.Index(
        pd.Series(endpoint_values)
        .dropna()
        .unique()
    )

    # Most convenient case: line endpoints refer directly to bus index.
    if endpoint_values.isin(buses.index).all():
        return pd.Series(
            buses.index,
            index=buses.index,
            name="bus_id",
        )

    candidate_columns = [
        "bus",
        "bus_id",
        "Bus",
        "BusID",
        "id",
        "ID",
    ]

    for column in candidate_columns:
        if column not in buses.columns:
            continue

        bus_ids = buses[column]

        if endpoint_values.isin(
            pd.Index(bus_ids.dropna().unique())
        ).all():
            return pd.Series(
                bus_ids.to_numpy(),
                index=buses.index,
                name="bus_id",
            )

    raise ValueError(
        "Could not match line endpoint identifiers to N490 buses.\n"
        f"Bus columns are:\n{buses.columns.tolist()}"
    )


def calculate_nodal_degrees(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> pd.DataFrame:
    """
    Calculate the nodal degree of every N490 bus at each voltage level.

    Each simple-graph edge contributes one degree to each terminal bus.
    The caller should therefore pass the deduplicated edge table produced by
    ``build_simple_graph_lines()``.

    Buses with no incident lines are retained with degree zero.

    Returns
    -------
    pandas.DataFrame
        One row per bus containing:

        - ``Vbase``
        - ``bus_id``
        - ``degree``
    """
    if "Vbase" not in buses.columns:
        raise ValueError(
            "N490 bus table does not contain 'Vbase'."
        )

    if "Vbase" not in lines.columns:
        raise ValueError(
            "N490 line table does not contain 'Vbase'."
        )

    from_col, to_col = _resolve_line_endpoint_columns(
        lines
    )

    endpoint_values = pd.Index(
        pd.concat(
            [
                lines[from_col],
                lines[to_col],
            ],
            ignore_index=True,
        )
    )

    bus_ids = _resolve_bus_ids(
        buses=buses,
        endpoint_values=endpoint_values,
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

        # ---------------------------------------------------------
        # Buses belonging to this voltage network
        # ---------------------------------------------------------
        bus_mask = np.isclose(
            bus_voltage,
            float(voltage_kv),
            equal_nan=False,
        )

        voltage_bus_ids = bus_ids.loc[
            bus_mask
        ]

        if voltage_bus_ids.empty:
            raise ValueError(
                f"No N490 buses found at {voltage_kv} kV."
            )

        # ---------------------------------------------------------
        # Lines belonging to this voltage network
        # ---------------------------------------------------------
        line_mask = np.isclose(
            line_voltage,
            float(voltage_kv),
            equal_nan=False,
        )

        voltage_lines = lines.loc[
            line_mask,
            [from_col, to_col],
        ]

        if voltage_lines.empty:
            raise ValueError(
                f"No N490 lines found at {voltage_kv} kV."
            )

        # ---------------------------------------------------------
        # Every occurrence of a bus as a line endpoint contributes
        # one to that bus's nodal degree.
        # ---------------------------------------------------------
        endpoints = pd.concat(
            [
                voltage_lines[from_col],
                voltage_lines[to_col],
            ],
            ignore_index=True,
        )

        degrees = (
            endpoints
            .value_counts()
            .astype(int)
        )

        rows.extend(
            {
                "Vbase": int(voltage_kv),
                "bus_id": bus_id,
                "degree": int(degree),
            }
            for bus_id, degree in degrees.items()
        )

    result = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Sanity check:
    #
    # sum(degree) must equal 2 * number of branches at each voltage.
    # -------------------------------------------------------------
    for voltage_kv in voltage_levels:
        degree_sum = int(
            result.loc[
                result["Vbase"] == voltage_kv,
                "degree",
            ].sum()
        )

        n_lines = int(
            np.isclose(
                line_voltage,
                float(voltage_kv),
                equal_nan=False,
            ).sum()
        )

        expected = 2 * n_lines

        if degree_sum != expected:
            raise RuntimeError(
                f"Nodal-degree check failed at {voltage_kv} kV: "
                f"sum(degree)={degree_sum}, "
                f"but 2 * n_lines={expected}."
            )

    return result


def calculate_degree_distribution(
    nodal_degrees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert bus-level nodal degrees into a degree-frequency distribution.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``Vbase``
        - ``degree``
        - ``n_nodes``
        - ``fraction_nodes``
    """
    distribution = (
        nodal_degrees
        .groupby(
            ["Vbase", "degree"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "n_nodes",
            }
        )
    )

    distribution["fraction_nodes"] = (
        distribution["n_nodes"]
        / distribution.groupby("Vbase")["n_nodes"]
        .transform("sum")
    )

    return distribution


def plot_degree_distribution(
    degree_distribution: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Plot and save the nodal-degree distribution for each voltage level.

    Because degree is discrete, the plot uses one bar per integer degree.
    """
    voltage_levels = sorted(
        degree_distribution["Vbase"].unique()
    )

    max_degree = int(
        degree_distribution["degree"].max()
    )

    # -------------------------------------------------------------
    # One figure per voltage level
    # -------------------------------------------------------------
    for voltage_kv in voltage_levels:

        data = (
            degree_distribution.loc[
                degree_distribution["Vbase"]
                == voltage_kv
            ]
            .set_index("degree")
            .reindex(
                range(0, max_degree + 1),
                fill_value=0,
            )
            .reset_index()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5.5)
        )

        ax.bar(
            data["degree"],
            data["n_nodes"],
            width=0.8,
        )

        ax.set_xlabel(
            "Nodal degree",
            fontsize=BASE_FONTSIZE,
        )

        ax.set_ylabel(
            "Number of nodes",
            fontsize=BASE_FONTSIZE,
        )

        ax.set_xticks(
            range(0, max_degree + 1)
        )

        ax.tick_params(
            axis="both",
            labelsize=BASE_FONTSIZE,
        )

        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        figure_path = (
            output_dir
            / f"N490_{voltage_kv}kv_nodal_degree_distribution.png"
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

        print(
            f"Saved:\n  {figure_path}"
        )
        
        

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
    output_dir: Path = OUTPUT_DIR,
) -> tuple[dict, dict]:
    """Fit both graph statistics, but plot only total length / MST.

    The E/N fit is retained as a diagnostic and saved by ``main()``. It is not
    plotted and its 132 kV extrapolation is not used in the final parameter
    summary.
    """
    LENGTH_COLOR = "#5c7c22"
    BLACK = "#000000"

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

    x_fit = np.linspace(
        float(extrapolate_to_kv),
        float(length_fit["x"].max()),
        300,
    )
    y_fit = length_fit["slope"] * x_fit + length_fit["intercept"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_fit,
        y_fit,
        color=LENGTH_COLOR,
        linewidth=2.2,
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

    ax.annotate(
        rf"{length_fit['extrapolated_value']:.4f}",
        xy=(extrapolate_to_kv, length_fit["extrapolated_value"]),
        xytext=(15, 0),
        textcoords="offset points",
        horizontalalignment="left",
        verticalalignment="center",
        color=LENGTH_COLOR,
        fontsize=BASE_FONTSIZE,
    )

    fit_text = (
        rf"$y={length_fit['slope']:.5f}V"
        rf"{length_fit['intercept']:+.3f}$"
        "\n"
        rf"$R^2={length_fit['r_squared']:.4f}$"
        "\n"
        rf"RMSE = {length_fit['rmse']:.4f}"
    )
    ax.text(
        0.98,
        0.55,
        fit_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color=BLACK,
        fontsize=BASE_FONTSIZE,
    )

    ax.set_xlabel(
        "Voltage level [kV]",
        color=BLACK,
        fontsize=BASE_FONTSIZE,
    )
    ax.set_ylabel(
        "Total line length / MST length",
        color=BLACK,
        fontsize=BASE_FONTSIZE,
    )
    ax.set_title("")
    ax.grid(False)

    ax.tick_params(
        axis="both",
        colors=BLACK,
        labelsize=BASE_FONTSIZE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_color(BLACK)

    voltage_ticks = sorted(
        set([float(extrapolate_to_kv)] + length_fit["x"].tolist())
    )
    ax.set_xticks(voltage_ticks)

    plt.tight_layout()

    figure_path = output_dir / "N490_line_to_MST_length_fit.png"
    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    print("\nLinear fit: edges per node (diagnostic only)")
    print("------------------------------------------------")
    print(f"Slope:                 {edge_fit['slope']:.6f} edges/node/kV")
    print(f"Intercept:             {edge_fit['intercept']:.6f}")
    print(f"R-squared:             {edge_fit['r_squared']:.6f}")
    print(f"RMSE:                  {edge_fit['rmse']:.6f}")
    print(
        f"Diagnostic extrapolation at {extrapolate_to_kv:g} kV: "
        f"{edge_fit['extrapolated_value']:.4f} (NOT used)"
    )

    print("\nLinear fit: total length / MST")
    print("--------------------------------")
    print(f"Slope:                 {length_fit['slope']:.6f} ratio/kV")
    print(f"Intercept:             {length_fit['intercept']:.6f}")
    print(f"R-squared:             {length_fit['r_squared']:.6f}")
    print(f"RMSE:                  {length_fit['rmse']:.6f}")
    print(
        f"Predicted at {extrapolate_to_kv:g} kV:    "
        f"{length_fit['extrapolated_value']:.4f}"
    )
    print(f"\nSaved figure:\n  {figure_path}")

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

    # Collapse parallel circuits once and use this same simple-graph edge set
    # consistently throughout the analysis.
    simple_lines, simple_graph_diagnostics = build_simple_graph_lines(lines)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n")
    print("=" * 72)
    print("N490 simple-graph reduction by voltage")
    print("=" * 72)
    print(simple_graph_diagnostics.to_string(index=False))

    simple_graph_diagnostics.to_pickle(
        OUTPUT_DIR / "N490_simple_graph_reduction.pkl"
    )
    simple_graph_diagnostics.to_csv(
        OUTPUT_DIR / "N490_simple_graph_reduction.csv",
        index=False,
    )

    statistics = calculate_line_bus_statistics(
        buses=buses,
        lines=simple_lines,
    )

    length_statistics = calculate_line_to_mst_length_ratio(
        lines=simple_lines,
    )
    
    # -------------------------------------------------------------
    # Nodal-degree distributions
    # -------------------------------------------------------------
    nodal_degrees = calculate_nodal_degrees(
        buses=buses,
        lines=simple_lines,
    )

    degree_distribution = calculate_degree_distribution(
        nodal_degrees
    )
    
    
    print("\n")
    print("=" * 72)
    print("N490 nodal-degree distribution by voltage")
    print("=" * 72)
    
    print(
        degree_distribution
        .to_string(index=False)
    )
    
    nodal_degrees.to_pickle(
        OUTPUT_DIR
        / "N490_nodal_degrees.pkl"
    )
    
    degree_distribution.to_pickle(
        OUTPUT_DIR
        / "N490_nodal_degree_distribution.pkl"
    )
    
    degree_distribution.to_csv(
        OUTPUT_DIR
        / "N490_nodal_degree_distribution.csv",
        index=False,
    )
    
    print(
        "\nSaved:"
        f"\n  {OUTPUT_DIR / 'N490_nodal_degrees.pkl'}"
        f"\n  {OUTPUT_DIR / 'N490_nodal_degree_distribution.pkl'}"
        f"\n  {OUTPUT_DIR / 'N490_nodal_degree_distribution.csv'}"
    )
    
    plot_degree_distribution(
        degree_distribution=degree_distribution,
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
    length_statistics.to_csv(
        OUTPUT_DIR
        / "N490_line_mst_length_statistics.csv",
        index=False,
    )
    
    print(
        "\nSaved:"
        f"\n  {OUTPUT_DIR / 'N490_line_mst_length_statistics.pkl'}"
        f"\n  {OUTPUT_DIR / 'N490_line_mst_length_statistics.csv'}"
    )
    
    edge_fit, length_fit = plot_graph_statistics_fits(
        statistics=statistics,
        length_statistics=length_statistics,
        extrapolate_to_kv=132,
    )


    fit_summary = pd.DataFrame(
        [
            {
                "statistic": "edges_per_node",
                "slope": edge_fit["slope"],
                "intercept": edge_fit["intercept"],
                "r_squared": edge_fit["r_squared"],
                "rmse": edge_fit["rmse"],
                "extrapolate_to_kv": edge_fit["extrapolate_to_kv"],
                "extrapolated_value": edge_fit["extrapolated_value"],
                "use_132_extrapolation": False,
            },
            {
                "statistic": "total_line_length_over_mst",
                "slope": length_fit["slope"],
                "intercept": length_fit["intercept"],
                "r_squared": length_fit["r_squared"],
                "rmse": length_fit["rmse"],
                "extrapolate_to_kv": length_fit["extrapolate_to_kv"],
                "extrapolated_value": length_fit["extrapolated_value"],
                "use_132_extrapolation": True,
            },
        ]
    )

    fit_summary.to_pickle(
        OUTPUT_DIR / "N490_graph_statistic_voltage_fits.pkl"
    )
    fit_summary.to_csv(
        OUTPUT_DIR / "N490_graph_statistic_voltage_fits.csv",
        index=False,
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
        OUTPUT_DIR / "N490_line_bus_statistics.pkl"
    )
    statistics.to_csv(
        OUTPUT_DIR / "N490_line_bus_statistics.csv",
        index=False,
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