"""
ndvi_calculation.py
-------------------
Step 1 of the AI Crop Health pipeline.

Reads Sentinel-2 bands, computes vegetation indices (NDVI, NDWI, SAVI, EVI),
runs K-Means clustering for unsupervised crop health mapping, and exports:
  - MRSAC_Crop_Health_Map.tif     (cluster labels: 0=Stressed, 1=Moderate, 2=Healthy)
  - Crop_Stress_Risk_Map.tif      (0-100 risk score raster — used by dashboard)
  - crop_health_statistics.csv

Run this script BEFORE district_risk_statistics.py and stress_hotspot_detection.py.
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================================================================
# CONFIGURATION  ← edit these two paths only
# ================================================================

RAW_DIR    = r"D:/Crop_Health/Crop_Health_Project/data/raw"   # folder with B02–B08 .jp2
OUTPUT_DIR = r"D:/Crop_Health/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# STEP 1: LOAD SENTINEL-2 BANDS
# ================================================================

print("\n========== NDVI CALCULATION & RISK MAP ==========\n")
print("Loading Sentinel-2 bands…")

band_paths = {
    "blue":  os.path.join(RAW_DIR, "B02.jp2"),
    "green": os.path.join(RAW_DIR, "B03.jp2"),
    "red":   os.path.join(RAW_DIR, "B04.jp2"),
    "nir":   os.path.join(RAW_DIR, "B08.jp2"),
}

bands = {}
meta  = None

for name, path in band_paths.items():
    with rasterio.open(path) as src:
        bands[name] = src.read(1).astype("float32")
        if meta is None:
            meta = src.meta.copy()

blue, green, red, nir = bands["blue"], bands["green"], bands["red"], bands["nir"]
print("✔ All 4 bands loaded")

# ================================================================
# STEP 2: VEGETATION INDICES
# ================================================================

eps = 1e-10

ndvi = (nir - red)   / (nir + red   + eps)          # Normalised Difference Vegetation Index
ndwi = (green - nir) / (green + nir + eps)           # Normalised Difference Water Index
savi = ((nir - red)  / (nir + red + 0.5)) * 1.5     # Soil-Adjusted Vegetation Index
evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + eps)  # Enhanced Vegetation Index

# Clip to physical range
ndvi = np.clip(ndvi, -1, 1)
ndwi = np.clip(ndwi, -1, 1)
savi = np.clip(savi, -1.5, 1.5)
evi  = np.clip(evi,  -1,   1)

print("✔ Vegetation indices calculated: NDVI, NDWI, SAVI, EVI")

# ================================================================
# STEP 3: FEATURE STACK & NORMALISATION
# ================================================================

features = np.dstack((ndvi, ndwi, savi, evi))
rows, cols, n_bands = features.shape

feature_flat = features.reshape(rows * cols, n_bands)
feature_flat = np.nan_to_num(feature_flat)

scaler = StandardScaler()
feature_scaled = scaler.fit_transform(feature_flat)

print("✔ Feature stack normalised for clustering")

# ================================================================
# STEP 4: K-MEANS CLUSTERING
# ================================================================

print("Running K-Means clustering (3 classes)…")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(feature_scaled)

health_clusters = kmeans.labels_.reshape(rows, cols)
print("✔ K-Means clustering completed")

# ================================================================
# STEP 5: LABEL CLUSTERS BY NDVI MEAN (low → stressed, high → healthy)
# ================================================================

cluster_means = {
    cid: np.mean(ndvi[health_clusters == cid])
    for cid in np.unique(health_clusters)
}

sorted_clusters = sorted(cluster_means, key=cluster_means.get)  # ascending NDVI

labeled_health_map = np.zeros_like(health_clusters, dtype=np.uint8)
labeled_health_map[health_clusters == sorted_clusters[0]] = 0   # Stressed
labeled_health_map[health_clusters == sorted_clusters[1]] = 1   # Moderate
labeled_health_map[health_clusters == sorted_clusters[2]] = 2   # Healthy

print("✔ Clusters re-labelled: 0=Stressed · 1=Moderate · 2=Healthy")

# ================================================================
# STEP 6: AREA STATISTICS
# ================================================================

total_pixels   = labeled_health_map.size
stressed_pct   = 100 * np.sum(labeled_health_map == 0) / total_pixels
moderate_pct   = 100 * np.sum(labeled_health_map == 1) / total_pixels
healthy_pct    = 100 * np.sum(labeled_health_map == 2) / total_pixels

print(f"\nCrop Health Area Statistics:")
print(f"  Stressed  : {stressed_pct:.2f}%")
print(f"  Moderate  : {moderate_pct:.2f}%")
print(f"  Healthy   : {healthy_pct:.2f}%")

csv_path = os.path.join(OUTPUT_DIR, "crop_health_statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Health Class", "Percentage Area"])
    writer.writerow(["Stressed", round(stressed_pct, 2)])
    writer.writerow(["Moderate", round(moderate_pct, 2)])
    writer.writerow(["Healthy",  round(healthy_pct,  2)])

print(f"✔ Statistics CSV saved → {csv_path}")

# ================================================================
# STEP 7: EXPORT CLASSIFIED HEALTH MAP GeoTIFF
# ================================================================

health_tif = os.path.join(OUTPUT_DIR, "MRSAC_Crop_Health_Map.tif")
meta.update({"dtype": rasterio.uint8, "count": 1, "nodata": 255})

with rasterio.open(health_tif, "w", **meta) as dst:
    dst.write(labeled_health_map, 1)

print(f"✔ Health-class GeoTIFF saved → {health_tif}")

# ================================================================
# STEP 8: BUILD CROP STRESS RISK MAP (0–100 score)
#
#   risk = 100 × (1 − scaled_NDVI)
#   Clipped to [0, 100].  This TIF is what the dashboard "Satellite
#   Risk Map" page reads and overlays on the interactive Folium map.
# ================================================================

ndvi_min = np.nanpercentile(ndvi, 2)
ndvi_max = np.nanpercentile(ndvi, 98)

ndvi_scaled = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + eps)
ndvi_scaled  = np.clip(ndvi_scaled, 0, 1)

risk_score = ((1 - ndvi_scaled) * 100).astype(np.float32)
risk_score  = np.clip(risk_score, 0, 100)

risk_meta = meta.copy()
risk_meta.update({"dtype": rasterio.float32, "count": 1, "nodata": -9999.0})

risk_tif = os.path.join(OUTPUT_DIR, "Crop_Stress_Risk_Map.tif")
with rasterio.open(risk_tif, "w", **risk_meta) as dst:
    dst.write(risk_score, 1)

print(f"✔ Crop Stress Risk Map (0–100) saved → {risk_tif}")

# ================================================================
# STEP 9: VISUALISATION (saved to disk + shown interactively)
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(ndvi,              cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[0].set_title("NDVI"); plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(labeled_health_map,cmap="RdYlGn", vmin=0,    vmax=2)
axes[1].set_title("AI Crop Health Map (K-Means)")
plt.colorbar(im1, ax=axes[1], ticks=[0, 1, 2])
axes[1].get_images()[0].colorbar.set_ticklabels(["Stressed","Moderate","Healthy"])

im2 = axes[2].imshow(risk_score,        cmap="RdYlGn_r", vmin=0,  vmax=100)
axes[2].set_title("Crop Stress Risk Score (0–100)"); plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "ndvi_overview.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"✔ Overview plot saved → {plot_path}")
plt.show()

print("\n========== ndvi_calculation.py COMPLETE ==========\n")
