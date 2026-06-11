# -*- coding: utf-8 -*-
"""
Inspect and export N490 DC-link locations.

This script inspects the N490 ``Data/*.pkl`` topology files, especially
``bus.pkl`` and ``link.pkl``, to determine whether DC-link endpoint bus
locations can be recovered directly from the N490 model.

The main workflow is:

1. Load the N490 bus and link topology tables.
2. Print diagnostic summaries of the bus, link, and optional topology tables.
3. Join ``link.bus0`` and ``link.bus1`` to ``bus.pkl`` to recover endpoint
   coordinates, bidding zones, voltage levels, and bus metadata.
4. Add endpoint-distance diagnostics comparing bus endpoint coordinates with
   any geometry arrays stored directly in ``link.pkl``.
5. Save a tabular endpoint summary to ``Data/diagnostics``.
6. Plot internal DC links whose two endpoint buses both exist in the N490
   modeled network.
7. Build a GeoDataFrame summarizing DC-link locations:
   - internal links are represented as ``LineString`` geometries between the
     two Nordic endpoint buses;
   - external links are represented as a single ``Point`` geometry at the
     Nordic-side endpoint bus, because only the modeled external-grid
     connection location is needed in the synthetic Nordic-grid workflow.
8. Save the resulting GeoDataFrame to the Nordic-grid raw N490 directory as
   pickle, GeoPackage, and CSV-with-WKT outputs.

Run from the N490 root directory, where the ``Data/`` folder is available:

    python link_locs.py

Expected inputs
---------------
Data/bus.pkl
Data/link.pkl

Optional inputs
---------------
Data/line.pkl
Data/trafo.pkl
Data/gen.pkl
Data/farms.pkl

Diagnostic outputs
------------------
Data/diagnostics/link_locations_from_bus_endpoints.csv
Data/diagnostics/link_table_columns.csv
Data/diagnostics/bus_table_columns.csv
Data/diagnostics/link_location_preview.png

Nordic-grid outputs
-------------------
/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490/n490_dc_link_locations.pkl
/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490/n490_dc_link_locations.gpkg
/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490/n490_dc_link_locations.csv

Notes
-----
The exported ``n490_dc_link_locations`` GeoDataFrame uses ``EPSG:4326``. It
contains mixed geometry types: ``LineString`` for internal links and ``Point``
for external-link Nordic endpoints. For later placement of synthetic-model
external grids, filter to:

    gdf[gdf["link_type"] == "external"]

Those rows contain the Nordic bus location for each external DC connection.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString


DATA_DIR = Path("Data")
OUT_DIR = DATA_DIR / "diagnostics"

NORDIC_GRID_N490_RAW_DIR = Path(
    "/Users/geoffreydesena/Documents/nordic-grid/data/raw/n490"
)

NORDIC_BIDZ = {
    "SE1",
    "SE2",
    "SE3",
    "SE4",
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "FI",
    "DK2",
}


def print_section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def load_pickle(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / name

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    obj = pd.read_pickle(path)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame.")

    return obj


def safe_nunique(series: pd.Series) -> int | None:
    """
    Count unique values in a Series, tolerating unhashable objects such as dicts,
    lists, and numpy arrays.

    Returns None if even string conversion fails, which should be rare.
    """
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        try:
            return int(series.dropna().map(repr).nunique())
        except Exception:
            return None


def describe_dataframe(name: str, df: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    print_section(f"{name}: shape and columns")
    print(f"shape: {df.shape}")
    print(list(df.columns))

    print_section(f"{name}: dtypes")
    print(df.dtypes.to_string())

    print_section(f"{name}: head")
    print(df.head(20).to_string())

    rows = []

    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "n_unique": safe_nunique(df[col]),
                "example_non_null": repr(df[col].dropna().iloc[0])
                if df[col].notna().any()
                else None,
            }
        )

    cols = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols.to_csv(out_dir / f"{name}_columns.csv", index=False)


def value_is_sequence(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def sequence_first(value):
    if value_is_sequence(value) and len(value) > 0:
        return value[0]
    return np.nan


def sequence_last(value):
    if value_is_sequence(value) and len(value) > 0:
        return value[-1]
    return np.nan


def sequence_len(value) -> int:
    if value_is_sequence(value):
        return len(value)
    return 0


def infer_coordinate_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Infer likely x/y and lon/lat coordinate columns in a table.
    """
    candidates = {
        "x": ["x", "X"],
        "y": ["y", "Y"],
        "lon": ["lon", "longitude", "Longitude", "lng"],
        "lat": ["lat", "latitude", "Latitude"],
    }

    found = {}

    for key, possible in candidates.items():
        found[key] = next((col for col in possible if col in df.columns), None)

    return found


def build_link_endpoint_table(bus: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    Join link bus0/bus1 endpoint IDs to the bus coordinate table.

    Returns one row per DC link, with:
      - link metadata
      - bus0/bus1 endpoint IDs
      - bus endpoint coordinates
      - optional link geometry endpoints if link.x/link.y/link.lat/link.lon exist
    """
    required_link_cols = {"bus0", "bus1"}
    missing = required_link_cols - set(link.columns)

    if missing:
        raise ValueError(
            "link.pkl does not have the expected endpoint bus columns: "
            f"{sorted(missing)}"
        )

    bus_coords = infer_coordinate_columns(bus)
    link_coords = infer_coordinate_columns(link)

    endpoint_rows = []

    for link_id, row in link.iterrows():
        bus0 = row.get("bus0", np.nan)
        bus1 = row.get("bus1", np.nan)

        bus0_exists = bus0 in bus.index
        bus1_exists = bus1 in bus.index

        out = {
            "link_id": link_id,
            "name": row.get("name", np.nan),
            "area0": row.get("area0", np.nan),
            "area1": row.get("area1", np.nan),
            "bus0": bus0,
            "bus1": bus1,
            "bus0_exists": bool(bus0_exists),
            "bus1_exists": bool(bus1_exists),
            "uc": row.get("uc", np.nan),
            "status": row.get("status", np.nan),
            "Pmax": row.get("Pmax", np.nan),
            "Cap": row.get("Cap", np.nan),
        }

        for side, bus_id, exists in [
            ("bus0", bus0, bus0_exists),
            ("bus1", bus1, bus1_exists),
        ]:
            if exists:
                bus_row = bus.loc[bus_id]

                for coord_name, col in bus_coords.items():
                    if col is not None:
                        out[f"{side}_{coord_name}"] = bus_row[col]

                out[f"{side}_name"] = bus_row.get("name", np.nan)
                out[f"{side}_bidz"] = bus_row.get("bidz", np.nan)
                out[f"{side}_area"] = bus_row.get("area", np.nan)
                out[f"{side}_country"] = bus_row.get("country", np.nan)
                out[f"{side}_Vbase"] = bus_row.get("Vbase", np.nan)
            else:
                for coord_name in ["x", "y", "lon", "lat"]:
                    out[f"{side}_{coord_name}"] = np.nan

                out[f"{side}_name"] = np.nan
                out[f"{side}_bidz"] = np.nan
                out[f"{side}_area"] = np.nan
                out[f"{side}_country"] = np.nan
                out[f"{side}_Vbase"] = np.nan

        # If the link table itself has geometry arrays, extract first/last point.
        for coord_name, col in link_coords.items():
            if col is None:
                continue

            value = row[col]

            out[f"link_{coord_name}_n"] = sequence_len(value)
            out[f"link_{coord_name}_first"] = sequence_first(value)
            out[f"link_{coord_name}_last"] = sequence_last(value)

        endpoint_rows.append(out)

    return pd.DataFrame(endpoint_rows)


def add_distance_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple endpoint consistency checks.

    These compare bus endpoint coordinates to the first/last points stored in
    link.x/link.y if available.
    """
    out = df.copy()

    required = {
        "bus0_x",
        "bus0_y",
        "bus1_x",
        "bus1_y",
        "link_x_first",
        "link_y_first",
        "link_x_last",
        "link_y_last",
    }

    if required.issubset(out.columns):
        out["bus0_to_link_first_m"] = np.sqrt(
            (out["bus0_x"] - out["link_x_first"]) ** 2
            + (out["bus0_y"] - out["link_y_first"]) ** 2
        )

        out["bus1_to_link_last_m"] = np.sqrt(
            (out["bus1_x"] - out["link_x_last"]) ** 2
            + (out["bus1_y"] - out["link_y_last"]) ** 2
        )

        out["bus0_to_link_last_m"] = np.sqrt(
            (out["bus0_x"] - out["link_x_last"]) ** 2
            + (out["bus0_y"] - out["link_y_last"]) ** 2
        )

        out["bus1_to_link_first_m"] = np.sqrt(
            (out["bus1_x"] - out["link_x_first"]) ** 2
            + (out["bus1_y"] - out["link_y_first"]) ** 2
        )

        out["endpoint_order"] = np.where(
            out["bus0_to_link_first_m"] + out["bus1_to_link_last_m"]
            <= out["bus0_to_link_last_m"] + out["bus1_to_link_first_m"],
            "bus0-first_bus1-last",
            "bus0-last_bus1-first",
        )

    if {"bus0_x", "bus0_y", "bus1_x", "bus1_y"}.issubset(out.columns):
        out["endpoint_distance_m"] = np.sqrt(
            (out["bus0_x"] - out["bus1_x"]) ** 2
            + (out["bus0_y"] - out["bus1_y"]) ** 2
        )

        out["endpoint_distance_km"] = out["endpoint_distance_m"] / 1000.0

    return out


def print_link_summary(link_locs: pd.DataFrame) -> None:
    print_section("DC link endpoint summary")
    preferred_cols = [
        "link_id",
        "name",
        "area0",
        "area1",
        "bus0",
        "bus0_name",
        "bus0_bidz",
        "bus0_Vbase",
        "bus0_x",
        "bus0_y",
        "bus0_lon",
        "bus0_lat",
        "bus1",
        "bus1_name",
        "bus1_bidz",
        "bus1_Vbase",
        "bus1_x",
        "bus1_y",
        "bus1_lon",
        "bus1_lat",
        "endpoint_distance_km",
        "endpoint_order",
    ]

    cols = [c for c in preferred_cols if c in link_locs.columns]
    print(link_locs[cols].to_string(index=False))

    print_section("Links by area pair")
    if {"area0", "area1"}.issubset(link_locs.columns):
        by_pair = (
            link_locs.assign(area_pair=link_locs["area0"].astype(str) + "-" + link_locs["area1"].astype(str))
            .groupby("area_pair")
            .size()
            .sort_values(ascending=False)
        )
        print(by_pair.to_string())

    if {"bus0_exists", "bus1_exists"}.issubset(link_locs.columns):
        print_section("Missing endpoint bus references")
        missing = link_locs.loc[
            ~link_locs["bus0_exists"] | ~link_locs["bus1_exists"],
            ["link_id", "name", "area0", "area1", "bus0", "bus1", "bus0_exists", "bus1_exists"],
        ]
        if missing.empty:
            print("All link endpoint buses exist in bus.pkl.")
        else:
            print(missing.to_string(index=False))


def plot_link_locations(
    bus: pd.DataFrame,
    link_locs: pd.DataFrame,
    out_path: Path = OUT_DIR / "link_location_preview.png",
) -> None:
    """
    Quick diagnostic plot in N490 projected x/y coordinates.
    """
    if not {"x", "y"}.issubset(bus.columns):
        print("Skipping plot: bus table does not contain x/y columns.")
        return

    if not {"bus0_x", "bus0_y", "bus1_x", "bus1_y"}.issubset(link_locs.columns):
        print("Skipping plot: link endpoint table does not contain endpoint x/y columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        bus["x"],
        bus["y"],
        s=3,
        alpha=0.25,
        label="N490 buses",
    )

    for _, row in link_locs.iterrows():
        if pd.notna(row["bus0_x"]) and pd.notna(row["bus1_x"]):
            ax.plot(
                [row["bus0_x"], row["bus1_x"]],
                [row["bus0_y"], row["bus1_y"]],
                linewidth=1.5,
                alpha=0.8,
            )

            ax.scatter(
                [row["bus0_x"], row["bus1_x"]],
                [row["bus0_y"], row["bus1_y"]],
                s=25,
            )

            label = str(row.get("name", row["link_id"]))
            x_mid = 0.5 * (row["bus0_x"] + row["bus1_x"])
            y_mid = 0.5 * (row["bus0_y"] + row["bus1_y"])
            ax.text(x_mid, y_mid, label, fontsize=7)

    ax.set_title("N490 DC-link endpoint locations from bus0/bus1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def inspect_optional_topology_tables(data_dir: Path = DATA_DIR) -> None:
    """
    Print very brief summaries of other topology tables if present.
    """
    for name in ["line.pkl", "trafo.pkl", "gen.pkl", "farms.pkl"]:
        path = data_dir / name

        if not path.exists():
            print(f"Optional file missing: {path}")
            continue

        df = pd.read_pickle(path)

        print_section(f"Optional table: {name}")
        print(f"shape: {df.shape}")
        print(list(df.columns))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Loading N490 topology pickles")
    bus = load_pickle("bus.pkl")
    link = load_pickle("link.pkl")

    describe_dataframe("bus_table", bus)
    describe_dataframe("link_table", link)

    inspect_optional_topology_tables()

    print_section("Raw link table")
    print(link.to_string())

    link_locs = build_link_endpoint_table(bus=bus, link=link)
    link_locs = add_distance_diagnostics(link_locs)

    print_link_summary(link_locs)

    out_csv = OUT_DIR / "link_locations_from_bus_endpoints.csv"
    link_locs.to_csv(out_csv, index=False)
    print(f"\nSaved link endpoint table: {out_csv}")

    plot_link_locations(bus=bus, link_locs=link_locs)
    
    dc_link_gdf = build_dc_link_location_gdf(link_locs)
    
    print_dc_link_location_gdf_summary(dc_link_gdf)
    
    save_dc_link_location_gdf(
        dc_link_gdf,
        output_dir=NORDIC_GRID_N490_RAW_DIR,
    )

    print_section("Interpretation")
    print(
        "If bus0_x/bus0_y and bus1_x/bus1_y are populated, then the DC-link "
        "locations can be recovered directly from link.pkl + bus.pkl.\n"
        "If link_x_first/link_y_first and link_x_last/link_y_last are also "
        "populated, compare the endpoint_order and distance diagnostics to "
        "confirm whether link.x/link.y follows bus0->bus1 or the reverse order."
    )
    
def _point_from_row(row: pd.Series, side: str) -> Point | None:
    """
    Create a WGS84 point from bus endpoint lon/lat columns.

    side should be 'bus0' or 'bus1'.
    """
    lon_col = f"{side}_lon"
    lat_col = f"{side}_lat"

    if lon_col not in row.index or lat_col not in row.index:
        return None

    lon = row[lon_col]
    lat = row[lat_col]

    if pd.isna(lon) or pd.isna(lat):
        return None

    return Point(float(lon), float(lat))


def _xy_point_from_row(row: pd.Series, side: str) -> Point | None:
    """
    Create a projected-coordinate point from bus endpoint x/y columns.
    """
    x_col = f"{side}_x"
    y_col = f"{side}_y"

    if x_col not in row.index or y_col not in row.index:
        return None

    x = row[x_col]
    y = row[y_col]

    if pd.isna(x) or pd.isna(y):
        return None

    return Point(float(x), float(y))


def _is_nordic_endpoint(row: pd.Series, side: str) -> bool:
    """
    Decide whether a link endpoint is inside the Nordic modeled system.

    Preference order:
      1. endpoint bus exists in bus.pkl
      2. endpoint bid zone is in NORDIC_BIDZ
      3. endpoint area is in NORDIC_BIDZ
    """
    exists_col = f"{side}_exists"
    bidz_col = f"{side}_bidz"
    area_col = f"{side}_area"

    if exists_col in row.index and bool(row.get(exists_col, False)):
        return True

    if bidz_col in row.index and row.get(bidz_col) in NORDIC_BIDZ:
        return True

    if area_col in row.index and row.get(area_col) in NORDIC_BIDZ:
        return True

    return False


def _link_type_from_row(row: pd.Series) -> str:
    """
    Classify link as internal or external based on endpoint bus availability.
    """
    bus0_exists = bool(row.get("bus0_exists", False))
    bus1_exists = bool(row.get("bus1_exists", False))

    if bus0_exists and bus1_exists:
        return "internal"
    if bus0_exists or bus1_exists:
        return "external"

    return "unresolved"


def build_dc_link_location_gdf(link_locs: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame summarizing N490 DC-link modeled endpoint locations.

    Internal links:
        one row per link, geometry is a LineString between both Nordic endpoint
        buses.

    External links:
        one row per link, geometry is the Nordic endpoint bus only. This is the
        location where an external-grid connection should be represented in the
        synthetic model.

    The GeoDataFrame CRS is EPSG:4326.
    """
    rows = []

    for _, row in link_locs.iterrows():
        link_type = _link_type_from_row(row)

        bus0_is_nordic = _is_nordic_endpoint(row, "bus0")
        bus1_is_nordic = _is_nordic_endpoint(row, "bus1")

        p0 = _point_from_row(row, "bus0")
        p1 = _point_from_row(row, "bus1")

        p0_xy = _xy_point_from_row(row, "bus0")
        p1_xy = _xy_point_from_row(row, "bus1")

        if link_type == "internal":
            if p0 is None or p1 is None:
                print(
                    f"WARNING: skipping internal link with missing geometry: "
                    f"{row.get('link_id')}"
                )
                continue

            geometry = LineString([p0, p1])
            geometry_type = "line_between_nordic_endpoints"

            nordic_bus = None
            nordic_side = "both"
            nordic_area = f"{row.get('area0')}-{row.get('area1')}"
            nordic_bidz = f"{row.get('bus0_bidz')}-{row.get('bus1_bidz')}"

            nordic_bus_x = None
            nordic_bus_y = None
            nordic_bus_lon = None
            nordic_bus_lat = None

        elif link_type == "external":
            if bus0_is_nordic and p0 is not None:
                geometry = p0
                geometry_type = "nordic_endpoint_point"
                nordic_side = "bus0"
                nordic_bus = row.get("bus0")
                nordic_area = row.get("area0")
                nordic_bidz = row.get("bus0_bidz")
                nordic_bus_x = row.get("bus0_x")
                nordic_bus_y = row.get("bus0_y")
                nordic_bus_lon = row.get("bus0_lon")
                nordic_bus_lat = row.get("bus0_lat")

            elif bus1_is_nordic and p1 is not None:
                geometry = p1
                geometry_type = "nordic_endpoint_point"
                nordic_side = "bus1"
                nordic_bus = row.get("bus1")
                nordic_area = row.get("area1")
                nordic_bidz = row.get("bus1_bidz")
                nordic_bus_x = row.get("bus1_x")
                nordic_bus_y = row.get("bus1_y")
                nordic_bus_lon = row.get("bus1_lon")
                nordic_bus_lat = row.get("bus1_lat")

            else:
                print(
                    f"WARNING: skipping external link without identifiable "
                    f"Nordic endpoint geometry: {row.get('link_id')}"
                )
                continue

        else:
            print(f"WARNING: skipping unresolved link: {row.get('link_id')}")
            continue

        rows.append(
            {
                "link_id": row.get("link_id"),
                "name": row.get("name"),
                "link_type": link_type,
                "geometry_type": geometry_type,
                "area0": row.get("area0"),
                "area1": row.get("area1"),
                "bus0": row.get("bus0"),
                "bus1": row.get("bus1"),
                "bus0_exists": row.get("bus0_exists"),
                "bus1_exists": row.get("bus1_exists"),
                "bus0_name": row.get("bus0_name"),
                "bus1_name": row.get("bus1_name"),
                "bus0_bidz": row.get("bus0_bidz"),
                "bus1_bidz": row.get("bus1_bidz"),
                "bus0_Vbase": row.get("bus0_Vbase"),
                "bus1_Vbase": row.get("bus1_Vbase"),
                "nordic_side": nordic_side,
                "nordic_bus": nordic_bus,
                "nordic_area": nordic_area,
                "nordic_bidz": nordic_bidz,
                "nordic_bus_x": nordic_bus_x,
                "nordic_bus_y": nordic_bus_y,
                "nordic_bus_lon": nordic_bus_lon,
                "nordic_bus_lat": nordic_bus_lat,
                "endpoint_distance_km": row.get("endpoint_distance_km"),
                "endpoint_order": row.get("endpoint_order"),
                "geometry": geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    if not gdf.empty:
        gdf = gdf.sort_values(
            ["link_type", "area0", "area1", "name"],
            na_position="last",
        ).reset_index(drop=True)

    return gdf


def save_dc_link_location_gdf(
    gdf: gpd.GeoDataFrame,
    output_dir: str | Path = NORDIC_GRID_N490_RAW_DIR,
) -> None:
    """
    Save DC-link location GeoDataFrame to the nordic-grid raw N490 directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "n490_dc_link_locations.pkl"
    gpkg_path = output_dir / "n490_dc_link_locations.gpkg"
    csv_path = output_dir / "n490_dc_link_locations.csv"

    gdf.to_pickle(pkl_path)

    # Geopackage is convenient for GIS inspection.
    try:
        gdf.to_file(gpkg_path, layer="n490_dc_link_locations", driver="GPKG")
    except Exception as exc:
        print(f"WARNING: could not write GeoPackage: {exc}")

    # CSV stores geometry as WKT for quick inspection.
    csv_df = gdf.copy()
    csv_df["geometry_wkt"] = csv_df.geometry.to_wkt()
    csv_df = pd.DataFrame(csv_df.drop(columns="geometry"))
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved pickle:     {pkl_path}")
    print(f"Saved GeoPackage: {gpkg_path}")
    print(f"Saved CSV:        {csv_path}")


def print_dc_link_location_gdf_summary(gdf: gpd.GeoDataFrame) -> None:
    print_section("DC link location GeoDataFrame summary")
    print(f"rows: {len(gdf)}")
    print(f"crs:  {gdf.crs}")
    print(gdf.geometry.geom_type.value_counts(dropna=False).to_string())

    cols = [
        "link_id",
        "name",
        "link_type",
        "area0",
        "area1",
        "nordic_side",
        "nordic_bus",
        "nordic_bidz",
        "nordic_bus_lon",
        "nordic_bus_lat",
        "bus0",
        "bus1",
        "bus0_exists",
        "bus1_exists",
    ]

    cols = [c for c in cols if c in gdf.columns]

    print_section("DC link location GeoDataFrame preview")
    if gdf.empty:
        print("EMPTY")
    else:
        print(gdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()