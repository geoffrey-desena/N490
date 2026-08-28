# -*- coding: utf-8 -*-
"""
N490 same-voltage line-crossing statistics.

For each N490 voltage network, this script:

1. Constructs the SIMPLE-GRAPH representation of the AC line network.
2. Builds each line geometry from the full N490 ``lon`` / ``lat`` path.
3. Identifies proper geometric crossings between lines of the SAME voltage.
4. Counts:
       - total simple-graph lines
       - distinct crossing pairs
       - lines that cross at least one other line
       - percentage of lines that cross at least one other line
5. Saves both summary statistics and detailed crossing information.

A crossing at a common line endpoint does NOT count. The Shapely
``crosses()`` predicate is used specifically so that ordinary network
connections at substations are excluded.

Outputs
-------
N490_same_voltage_line_crossing_summary.csv
N490_same_voltage_line_crossing_summary.pkl

N490_same_voltage_crossing_pairs.csv
N490_same_voltage_crossing_pairs.pkl

N490_lines_with_crossing_status.csv
N490_lines_with_crossing_status.pkl
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from nordic490 import N490


# =====================================================================
# CONFIGURATION
# =====================================================================

N490_OUTPUT_DIR = Path(
    "/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490"
)

VOLTAGE_LEVELS = [
    132,
    220,
    300,
    380,
]

SOURCE_CRS = "EPSG:4326"

# ETRS89 / LAEA Europe.
# Metric CRS suitable for spatial operations across the Nordic region.
TARGET_CRS = "EPSG:3035"


# =====================================================================
# BASIC HELPERS
# =====================================================================

def ensure_output_dir(
    output_dir: Path = N490_OUTPUT_DIR,
) -> Path:
    """Create the output directory if needed."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def load_n490_model(
    year: int = 2018,
) -> N490:
    """Load the Nordic490 model."""

    return N490(
        year=year
    )


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
# BUILD SIMPLE-GRAPH LINE TABLE
# =====================================================================

def build_simple_graph_lines(
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse parallel circuits to one line per unordered bus pair
    within each voltage network.

    The first branch geometry associated with each unique bus pair is
    retained.

    Voltage is included in the duplicate key so that connections at
    different voltage levels remain distinct.
    """

    required = {
        "Vbase",
        "lat",
        "lon",
    }

    missing = required - set(lines.columns)

    if missing:
        raise ValueError(
            "model.line is missing required columns: "
            f"{sorted(missing)}"
        )

    bus0_col, bus1_col = resolve_line_endpoint_columns(
        lines
    )

    work = lines.copy()

    # -------------------------------------------------------------
    # Preserve the original N490 line index explicitly.
    # Do this before any reset_index() calls so we do not depend on
    # what the dataframe index happens to be named.
    # -------------------------------------------------------------

    work["original_line_index"] = work.index

    # -------------------------------------------------------------
    # Standardize voltage.
    # -------------------------------------------------------------

    work["Vbase"] = pd.to_numeric(
        work["Vbase"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # N490 endpoint IDs may be stored as floats even though they
    # represent integer bus IDs.
    # -------------------------------------------------------------

    work[bus0_col] = pd.to_numeric(
        work[bus0_col],
        errors="coerce",
    )

    work[bus1_col] = pd.to_numeric(
        work[bus1_col],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "Vbase",
            bus0_col,
            bus1_col,
        ]
    ).copy()

    work[bus0_col] = (
        work[bus0_col]
        .astype(int)
    )

    work[bus1_col] = (
        work[bus1_col]
        .astype(int)
    )

    # -------------------------------------------------------------
    # Canonical unordered bus pair.
    # -------------------------------------------------------------

    endpoint_array = np.sort(
        work[
            [
                bus0_col,
                bus1_col,
            ]
        ].to_numpy(),
        axis=1,
    )

    work["bus_i"] = endpoint_array[:, 0]
    work["bus_j"] = endpoint_array[:, 1]

    # Remove self-loops.
    work = work.loc[
        work["bus_i"]
        != work["bus_j"]
    ].copy()

    # -------------------------------------------------------------
    # Simple graph:
    # one row per voltage + unordered bus pair.
    #
    # Keep the first N490 branch geometry for any parallel set.
    # -------------------------------------------------------------

    simple = (
        work
        .drop_duplicates(
            subset=[
                "Vbase",
                "bus_i",
                "bus_j",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # Stable identifier for this analysis.
    simple["line_id"] = np.arange(
        len(simple),
        dtype=int,
    )

    return simple

# =====================================================================
# BUILD LINE GEOMETRIES
# =====================================================================

def build_line_geometries(
    lines: pd.DataFrame,
    source_crs: str = SOURCE_CRS,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Construct full N490 line geometries from the stored lon/lat paths.

    The complete intermediate geometry of each line is retained rather
    than replacing the line with a straight terminal-to-terminal segment.
    """

    rows = []

    for _, row in lines.iterrows():

        lats = row["lat"]
        lons = row["lon"]

        if (
            lats is None
            or lons is None
        ):
            continue

        points = [
            (
                float(lon),
                float(lat),
            )
            for lat, lon in zip(
                lats,
                lons,
            )
            if (
                pd.notna(lat)
                and pd.notna(lon)
            )
        ]

        if len(points) < 2:
            continue

        geometry = LineString(
            points
        )

        rows.append(
            {
                "line_id": int(
                    row["line_id"]
                ),
                "original_line_index": row[
                    "original_line_index"
                ],
                "Vbase": int(
                    round(
                        float(
                            row["Vbase"]
                        )
                    )
                ),
                "bus_i": int(
                    row["bus_i"]
                ),
                "bus_j": int(
                    row["bus_j"]
                ),
                "geometry": geometry,
            }
        )

    if not rows:

        raise ValueError(
            "Could not construct any valid N490 line geometries."
        )

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=source_crs,
    )

    return gdf.to_crs(
        target_crs
    )


# =====================================================================
# CROSSING DETECTION
# =====================================================================

def find_same_voltage_crossings(
    lines_gdf: gpd.GeoDataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Identify proper crossings between lines of the same voltage.

    Returns
    -------
    crossing_pairs:
        One row per unique pair of crossing lines.

    line_status:
        One row per line, including:
            crosses_another
            n_crossings
    """

    crossing_rows = []

    line_status = (
        lines_gdf[
            [
                "line_id",
                "original_line_index",
                "Vbase",
                "bus_i",
                "bus_j",
            ]
        ]
        .copy()
    )

    line_status[
        "n_crossings"
    ] = 0

    # -------------------------------------------------------------
    # Analyze each voltage independently.
    # -------------------------------------------------------------

    for voltage_kv in voltage_levels:

        group = lines_gdf.loc[
            lines_gdf["Vbase"]
            == int(voltage_kv)
        ].copy()

        if group.empty:
            continue

        group = group.reset_index(
            drop=True
        )

        # Spatial index only reduces the number of expensive exact
        # geometry comparisons.
        sindex = group.sindex

        seen_pairs = set()

        for local_i, row_i in group.iterrows():

            geom_i = row_i.geometry
            line_i = int(
                row_i["line_id"]
            )

            # Bounding-box candidates.
            candidate_positions = list(
                sindex.intersection(
                    geom_i.bounds
                )
            )

            for local_j in candidate_positions:

                if local_j == local_i:
                    continue

                row_j = group.iloc[
                    local_j
                ]

                line_j = int(
                    row_j["line_id"]
                )

                # -------------------------------------------------
                # Canonical pair prevents double-counting:
                #
                # A-B == B-A
                # -------------------------------------------------

                pair = tuple(
                    sorted(
                        (
                            line_i,
                            line_j,
                        )
                    )
                )

                if pair in seen_pairs:
                    continue

                seen_pairs.add(
                    pair
                )

                geom_j = row_j.geometry

                # -------------------------------------------------
                # Proper crossings only.
                #
                # This excludes:
                # - shared terminal buses
                # - endpoint touching
                # - parallel/overlapping segments
                # -------------------------------------------------

                if not geom_i.crosses(
                    geom_j
                ):
                    continue

                intersection = (
                    geom_i.intersection(
                        geom_j
                    )
                )

                crossing_rows.append(
                    {
                        "Vbase": int(
                            voltage_kv
                        ),
                        "line_id_1": pair[0],
                        "line_id_2": pair[1],
                        "bus_i_1": int(
                            row_i["bus_i"]
                        )
                        if line_i == pair[0]
                        else int(
                            row_j["bus_i"]
                        ),
                        "bus_j_1": int(
                            row_i["bus_j"]
                        )
                        if line_i == pair[0]
                        else int(
                            row_j["bus_j"]
                        ),
                        "bus_i_2": int(
                            row_j["bus_i"]
                        )
                        if line_j == pair[1]
                        else int(
                            row_i["bus_i"]
                        ),
                        "bus_j_2": int(
                            row_j["bus_j"]
                        )
                        if line_j == pair[1]
                        else int(
                            row_i["bus_j"]
                        ),
                        "intersection_type": (
                            intersection.geom_type
                        ),
                    }
                )

                # Increment crossing count for both lines.
                line_status.loc[
                    line_status["line_id"]
                    == line_i,
                    "n_crossings",
                ] += 1

                line_status.loc[
                    line_status["line_id"]
                    == line_j,
                    "n_crossings",
                ] += 1

    crossing_pairs = pd.DataFrame(
        crossing_rows
    )

    line_status[
        "crosses_another"
    ] = (
        line_status[
            "n_crossings"
        ]
        > 0
    )

    return (
        crossing_pairs,
        line_status,
    )


# =====================================================================
# SUMMARIZE
# =====================================================================

def summarize_crossings(
    lines_gdf: gpd.GeoDataFrame,
    crossing_pairs: pd.DataFrame,
    line_status: pd.DataFrame,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> pd.DataFrame:
    """
    Summarize same-voltage line crossings by voltage network.
    """

    rows = []

    for voltage_kv in voltage_levels:

        voltage_lines = line_status.loc[
            line_status["Vbase"]
            == int(voltage_kv)
        ].copy()

        n_lines = len(
            voltage_lines
        )

        if n_lines == 0:
            continue

        n_lines_crossing = int(
            voltage_lines[
                "crosses_another"
            ].sum()
        )

        if crossing_pairs.empty:

            n_crossing_pairs = 0

        else:

            n_crossing_pairs = int(
                (
                    crossing_pairs[
                        "Vbase"
                    ]
                    == int(
                        voltage_kv
                    )
                ).sum()
            )

        percent_lines_crossing = (
            100.0
            * n_lines_crossing
            / n_lines
        )

        rows.append(
            {
                "Vbase": int(
                    voltage_kv
                ),
                "n_lines": int(
                    n_lines
                ),
                "n_crossing_pairs": (
                    n_crossing_pairs
                ),
                "n_lines_crossing": (
                    n_lines_crossing
                ),
                "percent_lines_crossing": float(
                    percent_lines_crossing
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# ANALYSIS
# =====================================================================

def analyze_same_voltage_line_crossings(
    model: N490,
    output_dir: Path = N490_OUTPUT_DIR,
    voltage_levels: list[int] = VOLTAGE_LEVELS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run and save the full same-voltage line-crossing analysis.
    """

    output_dir = ensure_output_dir(
        output_dir
    )

    lines = (
        model.line.copy()
    )

    # -------------------------------------------------------------
    # Complete network -> simple graph.
    # -------------------------------------------------------------

    simple_lines = (
        build_simple_graph_lines(
            lines
        )
    )

    # -------------------------------------------------------------
    # Full line geometries.
    # -------------------------------------------------------------

    lines_gdf = (
        build_line_geometries(
            simple_lines
        )
    )

    # -------------------------------------------------------------
    # Crossing detection.
    # -------------------------------------------------------------

    crossing_pairs, line_status = (
        find_same_voltage_crossings(
            lines_gdf=lines_gdf,
            voltage_levels=voltage_levels,
        )
    )

    # -------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------

    summary = summarize_crossings(
        lines_gdf=lines_gdf,
        crossing_pairs=crossing_pairs,
        line_status=line_status,
        voltage_levels=voltage_levels,
    )

    # -------------------------------------------------------------
    # Save summary.
    # -------------------------------------------------------------

    summary.to_csv(
        output_dir
        / "N490_same_voltage_line_crossing_summary.csv",
        index=False,
    )

    summary.to_pickle(
        output_dir
        / "N490_same_voltage_line_crossing_summary.pkl"
    )

    # -------------------------------------------------------------
    # Save crossing pairs.
    # -------------------------------------------------------------

    crossing_pairs.to_csv(
        output_dir
        / "N490_same_voltage_crossing_pairs.csv",
        index=False,
    )

    crossing_pairs.to_pickle(
        output_dir
        / "N490_same_voltage_crossing_pairs.pkl"
    )

    # -------------------------------------------------------------
    # Save individual line status.
    # -------------------------------------------------------------

    line_status.to_csv(
        output_dir
        / "N490_lines_with_crossing_status.csv",
        index=False,
    )

    line_status.to_pickle(
        output_dir
        / "N490_lines_with_crossing_status.pkl"
    )

    return (
        summary,
        crossing_pairs,
        line_status,
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """Run N490 same-voltage line-crossing analysis."""

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    output_dir = ensure_output_dir(
        N490_OUTPUT_DIR
    )

    print(
        "Loading N490 model..."
    )

    model = load_n490_model(
        year=2018
    )

    print(
        "\nCalculating same-voltage "
        "line crossings..."
    )

    (
        summary,
        crossing_pairs,
        line_status,
    ) = analyze_same_voltage_line_crossings(
        model=model,
        output_dir=output_dir,
        voltage_levels=VOLTAGE_LEVELS,
    )

    print("\n")
    print("=" * 95)
    print(
        "N490 SAME-VOLTAGE LINE CROSSINGS"
    )
    print("=" * 95)

    print(
        summary.to_string(
            index=False,
            formatters={
                "percent_lines_crossing":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    print("\n")
    print("=" * 95)
    print(
        "CROSSING PAIRS"
    )
    print("=" * 95)

    if crossing_pairs.empty:

        print(
            "No same-voltage crossings found."
        )

    else:

        print(
            crossing_pairs.to_string(
                index=False
            )
        )

    print(
        "\nSaved outputs to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()