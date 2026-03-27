"""
generate_ai_report.py
----------------------
Final step of the AI Crop Health pipeline.

Reads the district statistics CSV and heatmap PNG, then builds a
formatted PDF technical report using ReportLab and exports it to:
  - AI_Crop_Health_Technical_Report.pdf

The dashboard "AI Technical Report" page embeds this PDF in an iframe.

Run LAST — after all other scripts have completed.
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units  import inch
from reportlab.lib        import colors
from reportlab.lib.enums  import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ================================================================
# CONFIGURATION  ← edit output dir if needed
# ================================================================

OUTPUT_DIR = r"D:/Crop_Health/outputs"

STATS_CSV    = os.path.join(OUTPUT_DIR, "district_risk_statistics.csv")
HEATMAP_PNG  = os.path.join(OUTPUT_DIR, "district_crop_stress_heatmap.png")
OUTPUT_PDF   = os.path.join(OUTPUT_DIR, "AI_Crop_Health_Technical_Report.pdf")

# ================================================================
# STYLES
# ================================================================

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=20,
    leading=26,
    spaceAfter=6,
    alignment=TA_CENTER,
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    textColor=colors.HexColor("#555555"),
    alignment=TA_CENTER,
    spaceAfter=20,
)

h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=14,
    textColor=colors.HexColor("#1a4a2e"),
    spaceBefore=18,
    spaceAfter=6,
    borderPad=2,
)

h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=12,
    textColor=colors.HexColor("#2d6a4f"),
    spaceBefore=12,
    spaceAfter=4,
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)

caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8,
    textColor=colors.HexColor("#777777"),
    alignment=TA_CENTER,
    spaceAfter=10,
)

# ================================================================
# HELPER: inline matplotlib chart → ReportLab Image
# ================================================================

def mpl_to_reportlab_image(fig, width=6*inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    ratio = fig.get_size_inches()[1] / fig.get_size_inches()[0]
    return Image(buf, width=width, height=width * ratio)


# ================================================================
# LOAD STATS
# ================================================================

print("\n========== GENERATING AI CROP HEALTH REPORT ==========\n")

df = None
if os.path.exists(STATS_CSV):
    df = pd.read_csv(STATS_CSV)
    df = df.sort_values("mean_risk", ascending=False).reset_index(drop=True)
    print(f"✔ Loaded {len(df)} district records")
else:
    print(f"⚠ Stats CSV not found: {STATS_CSV}")

today = datetime.date.today().strftime("%B %d, %Y")

# ================================================================
# BUILD ELEMENTS
# ================================================================

elements = []

# ── Cover ───────────────────────────────────────────────────────

elements.append(Spacer(1, 0.6*inch))
elements.append(Paragraph("🌾 AI-Driven Crop Health Intelligence System", title_style))
elements.append(Paragraph(
    f"Technical Report · Sentinel-2 Satellite Analysis · {today}", subtitle_style
))
elements.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#2d6a4f")))
elements.append(Spacer(1, 0.2*inch))

# ── Abstract ────────────────────────────────────────────────────

elements.append(Paragraph("Abstract", h1_style))
elements.append(Paragraph(
    "This report presents the findings of an AI-assisted crop health monitoring "
    "framework applied to Sentinel-2 multispectral satellite imagery over Gujarat, India. "
    "Vegetation indices (NDVI, NDWI, SAVI, EVI) were computed and processed through a "
    "machine learning pipeline to detect crop stress, identify spatial hotspot clusters, "
    "quantify temporal NDVI change, and generate district-level agricultural risk statistics "
    "across the monitored region.", body_style
))

# ── Methodology ─────────────────────────────────────────────────

elements.append(Paragraph("Methodology", h1_style))

elements.append(Paragraph("Vegetation Index Computation", h2_style))
elements.append(Paragraph(
    "Bands B02 (Blue), B03 (Green), B04 (Red), and B08 (NIR) from Sentinel-2 Level-2A "
    "surface reflectance products were loaded with Rasterio. Four vegetation indices were "
    "derived: NDVI = (NIR − Red)/(NIR + Red), NDWI = (Green − NIR)/(Green + NIR), "
    "SAVI = [(NIR − Red)/(NIR + Red + 0.5)] × 1.5, and EVI = 2.5 × (NIR − Red)/"
    "(NIR + 6·Red − 7.5·Blue). All indices were clipped to physically valid ranges.",
    body_style
))

elements.append(Paragraph("Crop Stress Risk Score", h2_style))
elements.append(Paragraph(
    "A 0–100 crop stress risk score was derived by inverting and scaling the NDVI values "
    "relative to the 2nd–98th percentile range of the scene. High NDVI (healthy vegetation) "
    "maps to low risk; low NDVI (bare soil / stressed crops) maps to high risk. The resulting "
    "float32 GeoTIFF preserves full spatial resolution for map overlay.", body_style
))

elements.append(Paragraph("Machine Learning Classification", h2_style))
elements.append(Paragraph(
    "Four supervised classifiers were evaluated — Random Forest, XGBoost, SVM (RBF kernel), "
    "and KNN — on a stratified 30 000-pixel sample. Features consisted of the four vegetation "
    "indices. Labels were assigned by NDVI thresholding (< 0.30 = Stressed, 0.30–0.60 = "
    "Moderate, ≥ 0.60 = Healthy). The best-performing model was then applied to the full "
    "raster in 500 000-pixel chunks.", body_style
))

elements.append(Paragraph("Spatial Hotspot Detection", h2_style))
elements.append(Paragraph(
    "Pixels with risk score > 70 were extracted and subjected to DBSCAN clustering "
    "(ε = 5 pixels, min_samples = 20) to delineate contiguous high-stress zones. "
    "Cluster IDs were written to a GeoTIFF and visualised as a weighted heatmap "
    "overlaid on satellite imagery in the dashboard.", body_style
))

elements.append(Paragraph("Temporal Change Detection", h2_style))
elements.append(Paragraph(
    "NDVI was computed independently for two acquisition dates. Per-pixel ΔNDVI "
    "(date-2 minus date-1) was classified into four categories: Severe Stress Increase "
    "(ΔNDVI < −0.20), Mild Stress Increase (−0.20 to −0.05), Stable (±0.05), and "
    "Vegetation Improvement (> 0.05). Both the category raster and raw ΔNDVI float32 "
    "raster are exported for dashboard rendering.", body_style
))

# ── District Statistics ─────────────────────────────────────────

elements.append(PageBreak())
elements.append(Paragraph("District Crop Stress Statistics", h1_style))

if df is not None and not df.empty:
    # Summary KPIs
    n_d        = len(df)
    worst_d    = df.iloc[0]["NAME_2"]
    worst_risk = df.iloc[0]["mean_risk"]
    avg_risk   = df["mean_risk"].mean()

    elements.append(Paragraph(
        f"Zonal statistics were computed for <b>{n_d}</b> districts within the satellite "
        f"scene extent. The highest mean risk score is recorded in <b>{worst_d}</b> "
        f"({worst_risk:.3f}). The regional average is <b>{avg_risk:.3f}</b>.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    # Table
    tbl_cols = ["District", "Mean Risk", "Max Risk"]
    has_min  = "min_risk" in df.columns
    if has_min:
        tbl_cols.append("Min Risk")

    tbl_data = [tbl_cols]
    for _, row in df.iterrows():
        r_row = [row["NAME_2"],
                 f"{row['mean_risk']:.4f}",
                 f"{row['max_risk']:.4f}"]
        if has_min:
            r_row.append(f"{row.get('min_risk', 'N/A'):.4f}"
                         if pd.notna(row.get("min_risk")) else "N/A")
        tbl_data.append(r_row)

    n_cols = len(tbl_cols)
    col_w  = [2.2*inch] + [(6.3 - 2.2) / (n_cols - 1) * inch] * (n_cols - 1)

    tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1a4a2e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  10),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f7f3"), colors.white]),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccddcc")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 4),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.15*inch))

    # Bar chart of top districts
    top_n = min(15, n_d)
    top_df = df.head(top_n)
    bar_colors = [
        "#ff3b3b" if v >= 0.75 else
        "#ff6b35" if v >= 0.50 else
        "#ffcc00" if v >= 0.25 else "#2dff8f"
        for v in top_df["mean_risk"]
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(7, 0.38 * top_n + 1.2))
    fig_bar.patch.set_facecolor("#f8fdf9")
    ax_bar.set_facecolor("#f8fdf9")
    ax_bar.barh(top_df["NAME_2"][::-1], top_df["mean_risk"][::-1],
                color=list(reversed(bar_colors)), edgecolor="none")
    ax_bar.set_xlabel("Mean Risk Score", fontsize=9)
    ax_bar.set_title(f"Top {top_n} Districts by Mean Crop Stress Risk", fontsize=10)
    ax_bar.set_xlim(0, 1.05)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#ccddcc")
    plt.tight_layout()

    elements.append(mpl_to_reportlab_image(fig_bar, width=6.3*inch))
    elements.append(Paragraph(
        f"Figure 1 — Top {top_n} districts ranked by mean crop stress risk index.",
        caption_style
    ))
    plt.close(fig_bar)

else:
    elements.append(Paragraph("District statistics not available.", body_style))

# ── Heatmap ─────────────────────────────────────────────────────

elements.append(PageBreak())
elements.append(Paragraph("District Crop Stress Heatmap", h1_style))
elements.append(Paragraph(
    "The heatmap below shows mean and maximum risk scores across all monitored districts "
    "using a YlOrRd (yellow → red) colour scale. Darker red cells indicate higher stress.",
    body_style
))
elements.append(Spacer(1, 0.1*inch))

if os.path.exists(HEATMAP_PNG):
    elements.append(Image(HEATMAP_PNG, width=6.3*inch, height=4.5*inch))
    elements.append(Paragraph(
        "Figure 2 — District Crop Stress Heatmap (district_crop_stress_heatmap.png).",
        caption_style
    ))
else:
    elements.append(Paragraph(
        "Heatmap image not found — run district_risk_statistics.py to generate it.",
        body_style
    ))

# ── Conclusion ──────────────────────────────────────────────────

elements.append(Paragraph("Conclusion & Recommendations", h1_style))
elements.append(Paragraph(
    "The AI Crop Health Monitoring framework provides automated, scalable agricultural "
    "surveillance by combining Sentinel-2 satellite imagery, multi-index vegetation "
    "analysis, machine learning classification, and interactive geospatial dashboarding. "
    "The following actions are recommended based on current analysis:", body_style
))

recommendations = [
    "Deploy field inspection teams to <b>CRITICAL</b> districts (mean risk ≥ 0.75) "
    "within 72 hours.",
    "Issue soil-moisture monitoring advisories for HIGH-risk districts (0.50–0.74).",
    "Integrate the next available Sentinel-2 pass into the temporal change pipeline "
    "to update ΔNDVI maps.",
    "Consider drought-tolerant crop variety advisories for the upcoming planting season "
    "in the highest-stress districts.",
    "Expand district shapefile coverage to neighbouring states to capture spillover "
    "stress patterns.",
]

for i, rec in enumerate(recommendations, 1):
    elements.append(Paragraph(f"{i}. {rec}", body_style))

elements.append(Spacer(1, 0.2*inch))
elements.append(HRFlowable(width="100%", thickness=0.8,
                            color=colors.HexColor("#2d6a4f")))
elements.append(Spacer(1, 0.05*inch))
elements.append(Paragraph(
    f"Report generated automatically · {today} · "
    "AI Crop Health Monitoring System",
    caption_style
))

# ================================================================
# BUILD PDF
# ================================================================

doc = SimpleDocTemplate(
    OUTPUT_PDF,
    pagesize=(8.5*inch, 11*inch),
    leftMargin=1*inch, rightMargin=1*inch,
    topMargin=0.9*inch, bottomMargin=0.9*inch,
    title="AI Crop Health Technical Report",
    author="AI Crop Health Monitoring System",
)

doc.build(elements)
print(f"\n✔ PDF report generated → {OUTPUT_PDF}")
print("\n========== REPORT GENERATION COMPLETE ==========\n")
