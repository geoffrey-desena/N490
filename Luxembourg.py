from pathlib import Path
import geopandas as gpd
import fiona

gpkg = Path("LUX.gpkg")

print("Exists:", gpkg.exists())
print("Layers:")
layers = fiona.listlayers(gpkg)
for layer in layers:
    print(" -", layer)

for layer in layers:
    print("\n" + "=" * 60)
    print(layer)
    gdf = gpd.read_file(gpkg, layer=layer)

    print("rows:", len(gdf))
    print("crs:", gdf.crs)
    print("geometry types:")
    print(gdf.geometry.geom_type.value_counts(dropna=False))
    print("columns:")
    print(list(gdf.columns))
    print("\nhead:")
    print(gdf.head())
    
import geopandas as gpd
import pandas as pd

# Load substation layer
gdf = gpd.read_file("LUX.gpkg", layer="power_substation_point")

# --------------------------------------------------
# voltages summary
# --------------------------------------------------

print("\n==============================")
print("voltages counts")
print("==============================")

voltages_counts = (
    gdf["voltages"]
    .astype(str)
    .value_counts(dropna=False)
    .sort_index()
)

print(voltages_counts)

# --------------------------------------------------
# max_voltage summary
# --------------------------------------------------

print("\n==============================")
print("max_voltage counts")
print("==============================")

max_voltage_counts = (
    gdf["max_voltage"]
    .astype(str)
    .value_counts(dropna=False)
    .sort_index()
)

print(max_voltage_counts)

# --------------------------------------------------
# Numeric interpretation (kV)
# --------------------------------------------------

gdf["max_voltage_num"] = pd.to_numeric(
    gdf["max_voltage"],
    errors="coerce",
)

gdf["max_voltage_kv"] = gdf["max_voltage_num"] / 1000.0

print("\n==============================")
print("max_voltage_kv counts")
print("==============================")

print(
    gdf["max_voltage_kv"]
    .value_counts(dropna=False)
    .sort_index()
)

# --------------------------------------------------
# Combined summary table
# --------------------------------------------------

summary = pd.DataFrame({
    "voltages_count": (
        gdf["voltages"]
        .astype(str)
        .value_counts()
    ),
    "max_voltage_count": (
        gdf["max_voltage"]
        .astype(str)
        .value_counts()
    ),
})

summary = summary.fillna(0).astype(int)

print("\n==============================")
print("combined summary")
print("==============================")

print(summary.sort_index())