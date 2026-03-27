"""
supervised_model_comparison.py
--------------------------------
Optional Step 5 of the AI Crop Health pipeline.

Trains four supervised classifiers (Random Forest, XGBoost, SVM, KNN) on
vegetation-index features derived from Sentinel-2 bands, selects the best
model by accuracy, applies it to the full raster, and exports:
  - Supervised_Crop_Health_Map.tif   (uint8: 0=Stressed, 1=Moderate, 2=Healthy)
  - supervised_model_results.csv     (accuracy comparison table)

Labels are generated from NDVI thresholds (no manual annotation required).

Run after ndvi_calculation.py so the raw bands are available.
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION  ← edit these two paths only
# ================================================================

RAW_DIR    = r"D:/Crop_Health/Crop_Health_Project/data/raw"
OUTPUT_DIR = r"D:/Crop_Health/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# STEP 1: LOAD BANDS
# ================================================================

print("\n========== CROP HEALTH AI — SUPERVISED MODEL COMPARISON ==========\n")

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.meta.copy()

blue,  _    = read_band(os.path.join(RAW_DIR, "B02.jp2"))
green, _    = read_band(os.path.join(RAW_DIR, "B03.jp2"))
red,   meta = read_band(os.path.join(RAW_DIR, "B04.jp2"))
nir,   _    = read_band(os.path.join(RAW_DIR, "B08.jp2"))

print("✔ Bands loaded")

# ================================================================
# STEP 2: COMPUTE VEGETATION INDICES
# ================================================================

eps  = 1e-10
ndvi = np.clip((nir - red)   / (nir + red   + eps), -1, 1)
ndwi = np.clip((green - nir) / (green + nir + eps), -1, 1)
savi = np.clip(((nir - red)  / (nir + red + 0.5)) * 1.5, -1.5, 1.5)
evi  = np.clip(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + eps), -1, 1)

print("✔ Vegetation indices calculated")

# ================================================================
# STEP 3: NDVI-BASED LABELS  (no manual labelling needed)
# ================================================================

labels = np.zeros_like(ndvi, dtype="uint8")
labels[ndvi < 0.30]  = 0   # Stressed
labels[(ndvi >= 0.30) & (ndvi < 0.60)] = 1   # Moderate
labels[ndvi >= 0.60]  = 2   # Healthy

# ================================================================
# STEP 4: FEATURE MATRIX
# ================================================================

rows, cols = ndvi.shape
X = np.column_stack([
    ndvi.ravel(), ndwi.ravel(), savi.ravel(), evi.ravel()
])
X = np.nan_to_num(X)
y = labels.ravel().astype("int")

print(f"✔ Feature matrix: {X.shape}  Labels: {np.unique(y)}")

# ================================================================
# STEP 5: STRATIFIED RANDOM SAMPLE (to keep RAM manageable)
# ================================================================

SAMPLE_SIZE = 30_000
np.random.seed(42)
idx = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)

X_sample = X[idx]
y_sample = y[idx]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sample, test_size=0.30, random_state=42, stratify=y_sample
)

print(f"✔ Sample: {len(X_sample):,}  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ================================================================
# STEP 6: TRAIN & COMPARE MODELS
# ================================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    ),
    "SVM":           SVC(kernel="rbf", C=5, gamma="scale", random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

# XGBoost is optional — skip gracefully if not installed
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=120, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42,
        eval_metric="mlogloss", verbosity=0,
    )
except ImportError:
    print("⚠ XGBoost not installed — skipping")

results = {}

print("\n====== MODEL TRAINING & COMPARISON ======")
for name, model in models.items():
    print(f"\nTraining {name}…")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Stressed", "Moderate", "Healthy"]))

# ================================================================
# STEP 7: SELECT BEST MODEL & PREDICT FULL RASTER
# ================================================================

best_name  = max(results, key=results.get)
best_model = models[best_name]
best_acc   = results[best_name]

print(f"\n✔ Best model: {best_name}  (accuracy {best_acc:.4f})")

# Save comparison CSV
results_df = pd.DataFrame(
    [{"Model": n, "Accuracy": round(a, 4)} for n, a in results.items()]
).sort_values("Accuracy", ascending=False)
results_csv = os.path.join(OUTPUT_DIR, "supervised_model_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"✔ Model comparison CSV saved → {results_csv}")

# Chunk-wise prediction on full raster
print("\nPredicting full raster in chunks…")
supervised_map = np.zeros(rows * cols, dtype="uint8")

CHUNK = 500_000
for start in range(0, len(X), CHUNK):
    chunk_raw    = X[start : start + CHUNK]
    chunk_scaled = scaler.transform(chunk_raw)
    supervised_map[start : start + CHUNK] = best_model.predict(chunk_scaled)

supervised_map = supervised_map.reshape(rows, cols)
print("✔ Full raster prediction complete")

# ================================================================
# STEP 8: EXPORT GeoTIFF
# ================================================================

out_tif = os.path.join(OUTPUT_DIR, "Supervised_Crop_Health_Map.tif")

meta.update({"driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": 255})

with rasterio.open(out_tif, "w", **meta) as dst:
    dst.write(supervised_map, 1)

print(f"✔ Supervised map exported → {out_tif}")

# ================================================================
# QUICK STATS
# ================================================================

total  = supervised_map.size
for cls_id, cls_name in enumerate(["Stressed", "Moderate", "Healthy"]):
    cnt = np.sum(supervised_map == cls_id)
    print(f"  {cls_name:10s}: {cnt:>9,}  ({100*cnt/total:.2f}%)")

print("\n========== SUPERVISED MODEL COMPARISON COMPLETE ==========\n")
