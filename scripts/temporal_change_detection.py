"""
temporal_change_detection.py
-----------------------------
Step 4 of the AI Crop Health pipeline.

Reads two Sentinel-2 scenes (date1/ and date2/), computes per-pixel ΔNDVI,
classifies change, and exports:
  - Temporal_Stress_Change_Map.tif   (int8: 0=Severe, 1=Mild, 2=Stable, 3=Improvement)
  - Temporal_NDVI_Change_Raw.tif     (float32: raw ΔNDVI values for smooth map rendering)
  - Temporal_Change_Summary.csv

The dashboard "Temporal Change Analysis" page reads the category TIF and
overlays it on a live Folium map with a diverging colour ramp.
The "Vegetation Time Series" page animates the same layer frame-by-frame.

Run AFTER ndvi_calculation.py.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION  ← edit these paths only
# ================================================================

RAW_DIR    = r"D:/Crop_Health/Crop_Health_Project/data/raw"
OUTPUT_DIR = r"D:/Crop_Health/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

date1_red = os.path.join(RAW_DIR, "date1", "B04.jp2")
date1_nir = os.path.join(RAW_DIR, "date1", "B08.jp2")
date2_red = os.path.join(RAW_DIR, "date2", "B04.jp2")
date2_nir = os.path.join(RAW_DIR, "date2", "B08.jp2")

# ================================================================
# HELPER
# ================================================================

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.meta.copy()

# ================================================================
# STEP 1: LOAD DATE-1 BANDS
# ================================================================

print("\n========== TEMPORAL STRESS ANALYSIS ==========\n")

red1, meta = read_band(date1_red)
nir1, _    = read_band(date1_nir)
print("✔ Date-1 bands loaded")

# ================================================================
# STEP 2: LOAD DATE-2 BANDS
# ================================================================

red2, _ = read_band(date2_red)
nir2, _ = read_band(date2_nir)
print("✔ Date-2 bands loaded")

# ================================================================
# STEP 3: COMPUTE NDVI FOR EACH DATE
# ================================================================

eps   = 1e-10
ndvi1 = (nir1 - red1) / (nir1 + red1 + eps)
ndvi2 = (nir2 - red2) / (nir2 + red2 + eps)

ndvi1 = np.clip(ndvi1, -1, 1)
ndvi2 = np.clip(ndvi2, -1, 1)

print(f"✔ NDVI Date-1: min={ndvi1.min():.3f}  mean={ndvi1.mean():.3f}  max={ndvi1.max():.3f}")
print(f"✔ NDVI Date-2: min={ndvi2.min():.3f}  mean={ndvi2.mean():.3f}  max={ndvi2.max():.3f}")

# ================================================================
# STEP 4: DELTA NDVI  (positive = improvement, negative = decline)
# ================================================================

ndvi_change = ndvi2 - ndvi1   # range ≈ −2 to +2, typically −1 to +1
print(f"✔ ΔNDVI computed: min={ndvi_change.min():.3f}  max={ndvi_change.max():.3f}")

# ================================================================
# STEP 5: CLASSIFY CHANGE INTO 4 CATEGORIES
#
#   0 = Severe Stress Increase   ΔNDVI < −0.20
#   1 = Mild Stress Increase    −0.20 ≤ ΔNDVI < −0.05
#   2 = Stable                  −0.05 ≤ ΔNDVI ≤  0.05
#   3 = Vegetation Improvement   ΔNDVI >  0.05
# ================================================================

change_map = np.full(ndvi_change.shape, 2, dtype="int8")   # default: Stable

change_map[ndvi_change < -0.20]                                   = 0
change_map[(ndvi_change >= -0.20) & (ndvi_change < -0.05)]       = 1
change_map[(ndvi_change >= -0.05) & (ndvi_change <=  0.05)]      = 2
change_map[ndvi_change >  0.05]                                   = 3

print("✔ Change categories created (0=Severe, 1=Mild, 2=Stable, 3=Improvement)")

# ================================================================
# STEP 6: EXPORT CATEGORY RASTER  (int8) — used by dashboard map
# ================================================================

category_path = os.path.join(OUTPUT_DIR, "Temporal_Stress_Change_Map.tif")

meta_cat = meta.copy()
meta_cat.update({"driver": "GTiff", "dtype": "int8", "count": 1, "nodata": -128})

with rasterio.open(category_path, "w", **meta_cat) as dst:
    dst.write(change_map, 1)

print(f"✔ Category GeoTIFF saved → {category_path}")

# ================================================================
# STEP 7: EXPORT RAW ΔNDVI RASTER  (float32)
#         — gives smoother visual when rendered as image overlay
# ================================================================

raw_path = os.path.join(OUTPUT_DIR, "Temporal_NDVI_Change_Raw.tif")

meta_raw = meta.copy()
meta_raw.update({"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": -9999.0})

with rasterio.open(raw_path, "w", **meta_raw) as dst:
    dst.write(ndvi_change.astype("float32"), 1)

print(f"✔ Raw ΔNDVI GeoTIFF saved → {raw_path}")

# ================================================================
# STEP 8: STATISTICS SUMMARY
# ================================================================

unique, counts = np.unique(change_map, return_counts=True)
total_pixels   = change_map.size

category_labels = {
    0: "Severe Stress Increase",
    1: "Mild Stress Increase",
    2: "Stable",
    3: "Vegetation Improvement",
}

print("\n========== TEMPORAL CHANGE STATISTICS ==========")
rows_data = []
for val, cnt in zip(unique, counts):
    pct   = 100 * cnt / total_pixels
    label = category_labels.get(int(val), "Unknown")
    print(f"  {label:<32s}: {cnt:>9,} pixels  ({pct:.2f}%)")
    rows_data.append({
        "Category_ID":   int(val),
        "Category":      label,
        "Pixel_Count":   int(cnt),
        "Percentage":    round(pct, 2),
    })
print("=================================================\n")

df = pd.DataFrame(rows_data)
csv_path = os.path.join(OUTPUT_DIR, "Temporal_Change_Summary.csv")
df.to_csv(csv_path, index=False)
print(f"✔ Summary CSV saved → {csv_path}")

print("\n========== TEMPORAL ANALYSIS COMPLETE ==========\n")
