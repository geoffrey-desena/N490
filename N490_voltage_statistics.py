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