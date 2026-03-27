"""
district_risk_statistics.py
----------------------------
Step 2 of the AI Crop Health pipeline.

Reads Crop_Stress_Risk_Map.tif and the district boundary shapefile,
computes zonal statistics (mean / max / min risk per district), and exports:
  - district_risk_statistics.csv
  - district_crop_risk_map.shp
  - district_crop_stress_heatmap.png   ← used by dashboard "District Heatmap" page

Run AFTER ndvi_calculation.py (which creates Crop_Stress_Risk_Map.tif).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterstats import zonal_stats
from shapely.geometry import box

# ================================================================
# CONFIGURATION  ← edit these three paths only
# ================================================================

DISTRICT_SHP = r"D:/Crop_Health/Crop_Health_Project/data/boundaries/gadm41_IND_2.shp"
OUTPUT_DIR   = r"D:/Crop_Health/outputs"
RISK_RASTER  = os.path.join(OUTPUT_DIR, "Crop_Stress_Risk_Map.tif")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "district_risk_statistics.csv")
OUTPUT_SHP    = os.path.join(OUTPUT_DIR, "district_crop_risk_map.shp")
OUTPUT_HEATMAP= os.path.join(OUTPUT_DIR, "district_crop_stress_heatmap.png")

# ================================================================
# STEP 1: LOAD DATA
# ================================================================

print("\n========== DISTRICT RISK STATISTICS ==========\n")

if not os.path.exists(RISK_RASTER):
    raise FileNotFoundError(
        f"Risk raster not found: {RISK_RASTER}\n"
        "Run ndvi_calculation.py first."
    )

districts = gpd.read_file(DISTRICT_SHP)
print(f"✔ Shapefile loaded: {len(districts)} total districts")

with rasterio.open(RISK_RASTER) as src:
    raster_crs    = src.crs
    raster_bounds = src.bounds
    nodata_val    = src.nodata

# ================================================================
# STEP 2: CLIP TO RASTER EXTENT
# ================================================================

districts = districts.to_crs(raster_crs)

bbox = gpd.GeoDataFrame(
    geometry=[box(*raster_bounds)],
    crs=raster_crs,
)

intersecting = gpd.overlay(districts, bbox, how="intersection")
print(f"✔ Districts intersecting raster extent: {len(intersecting)}")

# ================================================================
# STEP 3: ZONAL STATISTICS
# ================================================================

print("Running zonal statistics…")

stats = zonal_stats(
    intersecting,
    RISK_RASTER,
    stats=["mean", "max", "min"],
    nodata=nodata_val if nodata_val is not None else -9999,
    all_touched=True,
)

intersecting["mean_risk"] = [
    round(s["mean"], 4) if s["mean"] is not None else np.nan
    for s in stats
]
intersecting["max_risk"] = [
    round(s["max"], 4) if s["max"]  is not None else np.nan
    for s in stats
]
intersecting["min_risk"] = [
    round(s["min"], 4) if s["min"]  is not None else np.nan
    for s in stats
]

# Drop rows where all stats are NaN (districts fully outside raster)
intersecting.dropna(subset=["mean_risk", "max_risk"], how="all", inplace=True)

print(f"✔ Zonal statistics computed for {len(intersecting)} districts")

# ================================================================
# STEP 4: EXPORT SHAPEFILE & CSV
# ================================================================

intersecting.to_file(OUTPUT_SHP)
print(f"✔ Shapefile saved → {OUTPUT_SHP}")

df_out = intersecting[["NAME_2", "mean_risk", "max_risk", "min_risk"]].copy()
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✔ CSV saved → {OUTPUT_CSV}")

# ================================================================
# STEP 5: GENERATE DISTRICT HEATMAP PNG
#
#   Matrix layout: districts (rows) × risk metrics (columns)
#   Colour = YlOrRd (white→yellow→red as risk increases)
# ================================================================

print("Generating district heatmap PNG…")

metrics = ["mean_risk", "max_risk", "min_risk"]
col_labels = ["Mean Risk", "Max Risk", "Min Risk"]

# Sort by mean_risk descending so worst districts appear at top
df_heat = (
    df_out[["NAME_2"] + metrics]
    .dropna()
    .sort_values("mean_risk", ascending=False)
    .set_index("NAME_2")
)

arr   = df_heat.values.astype(float)
n_d   = len(df_heat)
n_m   = len(metrics)

fig, ax = plt.subplots(figsize=(max(7, n_m * 2.5), max(6, n_d * 0.55)))
fig.patch.set_facecolor("#0a0f0d")
ax.set_facecolor("#0a0f0d")

vmin = 0
vmax = max(arr.max(), 1)   # normalise relative to actual risk range

im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

# Axis labels
ax.set_xticks(range(n_m))
ax.set_xticklabels(col_labels, rotation=0, ha="center",
                   color="#d4f0de", fontsize=11, fontweight="bold")
ax.set_yticks(range(n_d))
ax.set_yticklabels(df_heat.index, color="#d4f0de", fontsize=8)

ax.tick_params(length=0)

# Cell annotations
for i in range(n_d):
    for j in range(n_m):
        v    = arr[i, j]
        norm = (v - vmin) / (vmax - vmin + 1e-10)
        tc   = "black" if norm > 0.55 else "white"
        ax.text(j, i, f"{v:.3f}",
                ha="center", va="center", fontsize=7,
                color=tc, fontweight="bold")

# Grid lines between cells
for j in range(n_m + 1):
    ax.axvline(j - 0.5, color="#0a0f0d", linewidth=1.2)
for i in range(n_d + 1):
    ax.axhline(i - 0.5, color="#0a0f0d", linewidth=0.8)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Risk Score", color="#d4f0de", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#5a8a6a")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#d4f0de")

ax.set_title("District Crop Stress Heatmap",
             color="#d4f0de", fontsize=14, pad=14)

plt.tight_layout()
fig.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches="tight",
            facecolor="#0a0f0d")
plt.close(fig)

print(f"✔ Heatmap PNG saved → {OUTPUT_HEATMAP}")

print("\n========== DISTRICT STATISTICS COMPLETE ==========\n")
print("Top 5 highest-risk districts:")
print(df_out.sort_values("mean_risk", ascending=False).head(5).to_string(index=False))
print()
