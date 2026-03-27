"""
stress_hotspot_detection.py
---------------------------
Step 3 of the AI Crop Health pipeline.

Reads Crop_Stress_Risk_Map.tif (0–100 float), identifies pixels above a
stress threshold, applies DBSCAN spatial clustering, and exports:
  - Crop_Stress_Hotspots.tif   (cluster-ID raster, 0 = non-hotspot,
                                 1…N = cluster IDs, −1 = noise pixels)

The dashboard "Hotspot Detection" page reads this TIF and renders it as
a Folium HeatMap layer on a satellite basemap.

Run AFTER ndvi_calculation.py.
"""

import os
import numpy as np
import rasterio
from sklearn.cluster import DBSCAN

# ================================================================
# CONFIGURATION  ← edit these two paths only
# ================================================================

OUTPUT_DIR = r"D:/Crop_Health/outputs"

risk_path   = os.path.join(OUTPUT_DIR, "Crop_Stress_Risk_Map.tif")
output_path = os.path.join(OUTPUT_DIR, "Crop_Stress_Hotspots.tif")

# ================================================================
# MAIN
# ================================================================

print("\n========== CROP STRESS HOTSPOT DETECTION ==========\n")

if not os.path.exists(risk_path):
    raise FileNotFoundError(
        f"Risk raster not found: {risk_path}\n"
        "Run ndvi_calculation.py first."
    )

# ------------------------------------------------------------------
# Load risk raster
# ------------------------------------------------------------------

with rasterio.open(risk_path) as src:
    risk      = src.read(1).astype("float32")
    transform = src.transform
    meta      = src.meta.copy()
    nodata    = src.nodata if src.nodata is not None else -9999.0

# Mask nodata
risk[risk == nodata] = np.nan

print(f"Raster loaded: shape={risk.shape}  "
      f"min={np.nanmin(risk):.1f}  max={np.nanmax(risk):.1f}")

# ------------------------------------------------------------------
# Select high-stress pixels (threshold = 70 out of 100)
# ------------------------------------------------------------------

THRESHOLD = 70.0
mask = (risk > THRESHOLD) & (~np.isnan(risk))
coords = np.column_stack(np.where(mask))        # row, col indices

print(f"High-stress pixels (risk > {THRESHOLD}): {len(coords):,}")

if len(coords) == 0:
    print("No hotspot pixels found — lowering threshold to 50.")
    THRESHOLD = 50.0
    mask   = (risk > THRESHOLD) & (~np.isnan(risk))
    coords = np.column_stack(np.where(mask))
    print(f"Pixels at threshold {THRESHOLD}: {len(coords):,}")

if len(coords) == 0:
    print("Still no pixels — exporting empty hotspot raster.")
    hotspot_map = np.zeros_like(risk, dtype="int16")
else:
    # ------------------------------------------------------------------
    # DBSCAN spatial clustering
    # eps   = neighbourhood radius in pixels
    # min_samples = min pixels to form a core point
    # ------------------------------------------------------------------
    print("Running DBSCAN clustering…")

    clustering = DBSCAN(eps=5, min_samples=20).fit(coords)
    labels     = clustering.labels_           # −1 = noise

    unique_labels = set(labels)
    n_clusters    = len(unique_labels - {-1})
    n_noise       = np.sum(labels == -1)

    print(f"✔ Clusters found : {n_clusters}")
    print(f"  Noise pixels   : {n_noise:,}")

    # ------------------------------------------------------------------
    # Build output raster
    #   0  = background (non-hotspot)
    #  -1  = DBSCAN noise (isolated high-risk pixels)
    #  1…N = cluster IDs
    # ------------------------------------------------------------------
    hotspot_map = np.zeros(risk.shape, dtype="int16")

    for i, (r, c) in enumerate(coords):
        if labels[i] == -1:
            hotspot_map[r, c] = -1           # noise
        else:
            hotspot_map[r, c] = int(labels[i]) + 1   # 1-based cluster ID

    print(f"  Max cluster ID : {hotspot_map.max()}")

# ------------------------------------------------------------------
# Export — keep exact same georeference as the source risk raster
# ------------------------------------------------------------------

meta.update({
    "dtype":  "int16",
    "count":  1,
    "nodata": -9999,
})

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(hotspot_map, 1)

print(f"\n✔ Hotspot raster exported → {output_path}")
print("\n========== HOTSPOT DETECTION COMPLETE ==========\n")
