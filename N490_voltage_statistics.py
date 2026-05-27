import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from nordic490 import N490

# -----------------------------
# Load model and overwrite with
# locally saved generator data
# -----------------------------
m = N490(year=2018)

m.gen = pd.read_pickle("Data/gen.pkl")
m.farms = pd.read_pickle("Data/farms.pkl")

bus = m.bus.copy()

# -----------------------------
# Conventional generators
# -----------------------------
gen = m.gen.copy()
gen["source"] = "conventional"

# -----------------------------
# Renewable farms
# -----------------------------
farms = m.farms.copy()
farms["source"] = "renewable"

# -------------------------------------------------
# Identify bus column and capacity column for farms
# -------------------------------------------------
print("\nFarm columns")
print("------------")
print(list(farms.columns))

# Common guesses
farm_bus_col_candidates = ["bus", "Bus", "bus_id"]
farm_cap_col_candidates = ["Pmax", "pmax", "MW", "capacity", "max_p_mw"]

farm_bus_col = None
farm_cap_col = None

for col in farm_bus_col_candidates:
    if col in farms.columns:
        farm_bus_col = col
        break

for col in farm_cap_col_candidates:
    if col in farms.columns:
        farm_cap_col = col
        break

if farm_bus_col is None:
    raise ValueError("Could not identify farm bus column.")

if farm_cap_col is None:
    raise ValueError("Could not identify farm capacity column.")

# Standardize renewable dataframe
farms = farms.rename(
    columns={
        farm_bus_col: "bus",
        farm_cap_col: "Pmax",
    }
)

# -----------------------------
# Combine datasets
# -----------------------------
all_gen = pd.concat(
    [
        gen[["bus", "Pmax", "source"]],
        farms[["bus", "Pmax", "source"]],
    ],
    ignore_index=True,
)

# -----------------------------
# Map bus voltages
# -----------------------------
all_gen["bus_voltage_kv"] = all_gen["bus"].map(bus["Vbase"])

# Clean data
plot_df = all_gen.dropna(subset=["bus_voltage_kv", "Pmax"]).copy()

plot_df["bus_voltage_kv"] = pd.to_numeric(
    plot_df["bus_voltage_kv"],
    errors="coerce",
)

plot_df["Pmax"] = pd.to_numeric(
    plot_df["Pmax"],
    errors="coerce",
)

plot_df = plot_df.dropna(subset=["bus_voltage_kv", "Pmax"])

# -----------------------------
# Linear regression
# -----------------------------
reg = linregress(
    plot_df["bus_voltage_kv"],
    plot_df["Pmax"],
)

print("\nLinear regression")
print("-----------------")
print(f"slope:       {reg.slope:.4f} MW/kV")
print(f"intercept:   {reg.intercept:.4f} MW")
print(f"r:           {reg.rvalue:.4f}")
print(f"r-squared:   {reg.rvalue**2:.4f}")
print(f"p-value:     {reg.pvalue:.4e}")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Plot conventional and renewable separately
for source, marker in [
    ("conventional", "o"),
    ("renewable", "^"),
]:
    subset = plot_df[plot_df["source"] == source]

    ax.scatter(
        subset["bus_voltage_kv"],
        subset["Pmax"],
        s=45,
        alpha=0.7,
        marker=marker,
        label=source,
    )

# Trend line
xline = sorted(plot_df["bus_voltage_kv"].unique())
yline = [
    reg.intercept + reg.slope * x
    for x in xline
]

ax.plot(
    xline,
    yline,
    linewidth=2,
    label=f"Linear fit (r = {reg.rvalue:.2f})",
)

ax.set_xlabel("Bus nominal voltage (kV)")
ax.set_ylabel("Generator capacity Pmax (MW)")
ax.set_title("N490 generation capacity by connected bus voltage")

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Prepare data
# -----------------------------
plot_df = bus[["Vbase", "load_share"]].copy()

plot_df["Vbase"] = pd.to_numeric(plot_df["Vbase"], errors="coerce")
plot_df["load_share"] = pd.to_numeric(plot_df["load_share"], errors="coerce")

# plot_df = plot_df.dropna(subset=["Vbase", "load_share"])

# Optional: remove zero-load buses if you want relationship among load buses only
plot_df = plot_df[plot_df["load_share"] > 0].copy()

# -----------------------------
# Linear regression
# -----------------------------
reg = linregress(
    plot_df["Vbase"],
    plot_df["load_share"],
)

print("\nLinear regression: load_share vs bus voltage")
print("--------------------------------------------")
print(f"slope:       {reg.slope:.8f} load_share/kV")
print(f"intercept:   {reg.intercept:.8f}")
print(f"r:           {reg.rvalue:.4f}")
print(f"r-squared:   {reg.rvalue**2:.4f}")
print(f"p-value:     {reg.pvalue:.4e}")

# -----------------------------
# Summary by voltage
# -----------------------------
summary = (
    plot_df.groupby("Vbase")["load_share"]
    .agg(
        n_buses="size",
        total_load_share="sum",
        mean_load_share="mean",
        median_load_share="median",
        max_load_share="max",
    )
    .reset_index()
    .sort_values("Vbase")
)

print("\nLoad-share summary by voltage")
print("-----------------------------")
print(summary.to_string(index=False))

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    plot_df["Vbase"],
    plot_df["load_share"],
    s=45,
    alpha=0.7,
    marker="o",
)

xline = sorted(plot_df["Vbase"].unique())
yline = [
    reg.intercept + reg.slope * x
    for x in xline
]

ax.plot(
    xline,
    yline,
    linewidth=2,
    label=f"Linear fit (r = {reg.rvalue:.2f})",
)

ax.set_xlabel("Bus nominal voltage (kV)")
ax.set_ylabel("Load share")
ax.set_title("N490 load share by connected bus voltage")

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Load model
# -----------------------------
m = N490(year=2018)

bus = m.bus.copy()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# -----------------------------
# Inspect available columns
# -----------------------------
print("\nBus columns")
print("-----------")
print(list(bus.columns))

# ---------------------------------------------------
# Create synthetic substations from shared coordinates
# ---------------------------------------------------
#
# Assumption:
# buses sharing identical lat/lon belong
# to the same physical substation.
#
# Use rounded coordinates to avoid floating precision
# issues.
# ---------------------------------------------------

bus["lat_r"] = bus["lat"].round(6)
bus["lon_r"] = bus["lon"].round(6)

# Create coordinate key
bus["substation_key"] = (
    bus["lat_r"].astype(str)
    + "_"
    + bus["lon_r"].astype(str)
)

# Assign unique substation IDs
substation_lookup = {
    key: idx
    for idx, key in enumerate(
        sorted(bus["substation_key"].unique())
    )
}

bus["substation_id"] = bus["substation_key"].map(substation_lookup)

# ---------------------------------------------------
# Create substation dataframe
# ---------------------------------------------------

# Voltages to inspect
voltages = [132.0, 220.0, 300.0, 380.0]

# Aggregate representative information
substations = (
    bus.groupby("substation_id")
    .agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        bidz=("bidz", "first"),
        country=("country", "first"),
    )
    .reset_index()
)

# Add boolean voltage columns
for v in voltages:

    buses_at_v = (
        bus[bus["Vbase"] == v]
        .groupby("substation_id")
        .size()
        .index
    )

    substations[f"{int(v)}_kV"] = (
        substations["substation_id"]
        .isin(buses_at_v)
    )

# ---------------------------------------------------
# Print substation summary
# ---------------------------------------------------

print("\nSubstation dataframe sample")
print("---------------------------")
print(substations.head(20))

print("\nNumber of substations by bidding zone")
print("-------------------------------------")

zone_counts = (
    substations.groupby("bidz")
    .size()
    .reset_index(name="n_substations")
    .sort_values("n_substations", ascending=False)
)

print(zone_counts.to_string(index=False))

# ---------------------------------------------------
# Voltage presence summary by bidding zone
# ---------------------------------------------------

summary_rows = []

for zone, grp in substations.groupby("bidz"):

    n_subs = len(grp)

    row = {
        "bidz": zone,
        "n_substations": n_subs,
    }

    for v in voltages:

        col = f"{int(v)}_kV"

        pct = 100 * grp[col].mean()

        row[col] = pct

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values("bidz")

# Format as percentages
summary_fmt = summary.copy()

for v in voltages:
    col = f"{int(v)}_kV"
    summary_fmt[col] = summary_fmt[col].map(lambda x: f"{x:.1f}%")

print("\nPercentage of substations hosting each voltage")
print("------------------------------------------------")
print(summary_fmt.to_string(index=False))

# ---------------------------------------------------
# Optional: inspect multi-voltage substations
# ---------------------------------------------------

substations["n_voltage_levels"] = (
    substations[[f"{int(v)}_kV" for v in voltages]]
    .sum(axis=1)
)

print("\nDistribution of number of voltage levels per substation")
print("-------------------------------------------------------")

print(
    substations["n_voltage_levels"]
    .value_counts()
    .sort_index()
)


