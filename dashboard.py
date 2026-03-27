import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import base64
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import io
from PIL import Image

# Raise PIL decompression bomb limit — needed for large satellite-derived PNGs
Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Crop Health Monitoring Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0e1b13; }
    [data-testid="stSidebarNav"] { display: none; }
    .block-container { padding-top: 1.5rem; }
    .metric-container { background: #101915; border: 1px solid #1e3028; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #d4f0de; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 AI Crop Health Monitoring Dashboard")
st.markdown(
    "AI-driven crop monitoring system using **Sentinel-2** satellite imagery, "
    "vegetation indices, machine learning classification, and spatial analytics."
)

# ---------------------------------------------------
# FILE PATHS  ← update these to match your folder layout
# ---------------------------------------------------

BASE_OUT = r"D:/Crop_Health/outputs"

district_shp   = os.path.join(BASE_OUT, "district_crop_risk_map.shp")
stats_csv      = os.path.join(BASE_OUT, "district_risk_statistics.csv")
heatmap_img    = os.path.join(BASE_OUT, "district_crop_stress_heatmap.png")
report_pdf     = os.path.join(BASE_OUT, "AI_Crop_Health_Technical_Report.pdf")

risk_raster    = os.path.join(BASE_OUT, "Crop_Stress_Risk_Map.tif")
hotspot_raster = os.path.join(BASE_OUT, "Crop_Stress_Hotspots.tif")
temporal_raster= os.path.join(BASE_OUT, "Temporal_Stress_Change_Map.tif")

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------

st.sidebar.title("🌿 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "District Analysis",
        "Interactive Monitoring Map",
        "Satellite Risk Map",
        "Hotspot Detection",
        "Temporal Change Analysis",
        "Vegetation Time Series",
        "District Heatmap",
        "AI Technical Report",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("System: Sentinel-2 · NDVI · ML")
st.sidebar.caption("Region: Gujarat, India")

# ---------------------------------------------------
# HELPER: reproject TIF to EPSG:4326 and return bounds + array
# ---------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raster_4326(path, max_pixels=1500):
    """
    Reads a raster, reprojects to EPSG:4326, downsamples so the longer
    dimension is at most `max_pixels` (default 1500), and returns
    (data_array_float32, [west, south, east, north]).

    Downsampling is critical for large Sentinel-2 TIFs (e.g. 10740×11384)
    which would otherwise require several GB of RAM when converted to RGBA.
    1500×1500 pixels is more than sufficient for a web map overlay.
    """
    with rasterio.open(path) as src:
        src_crs = src.crs
        src_bounds = src.bounds
        src_nodata = src.nodata

        # -- Step 1: calculate full-resolution reprojected dimensions --
        dst_crs = "EPSG:4326"
        full_transform, full_width, full_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src_bounds
        )

        # -- Step 2: compute downsampled dimensions keeping aspect ratio --
        scale = min(max_pixels / max(full_width, full_height), 1.0)
        out_width  = max(1, int(full_width  * scale))
        out_height = max(1, int(full_height * scale))

        # -- Step 3: recalculate transform for the downsampled grid --
        from rasterio.transform import from_bounds as tfrom_bounds
        from rasterio.crs import CRS as RCRS

        # geographic bounds in 4326
        from rasterio.warp import transform_bounds
        west, south, east, north = transform_bounds(
            src_crs, dst_crs, *src_bounds
        )
        out_transform = tfrom_bounds(west, south, east, north,
                                     out_width, out_height)

        # -- Step 4: reproject + resample directly into small output buffer --
        data = np.full((src.count, out_height, out_width),
                       fill_value=np.nan, dtype=np.float32)

        for band_i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_i),
                destination=data[band_i - 1],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=out_transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,   # average = smoother downscale
                src_nodata=src_nodata,
                dst_nodata=np.nan,
            )

    band = data[0]
    bounds = [west, south, east, north]   # [W, S, E, N]
    return band, bounds


def array_to_png_base64(arr, cmap_name, vmin=None, vmax=None, nodata=None):
    """
    Convert a 2-D numpy array to a base64-encoded RGBA PNG for Folium ImageOverlay.

    Works entirely in float32 / uint8 to avoid the ~3.6 GB float64 allocation
    that matplotlib's cmap() would otherwise produce on large arrays.
    """
    arr = arr.astype(np.float32)   # ensure float32 throughout
    if nodata is not None:
        arr[arr == nodata] = np.nan

    nan_mask = np.isnan(arr)

    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))

    # --- Normalise to [0, 1] in float32 ---
    rng = vmax - vmin if vmax != vmin else 1.0
    norm_arr = np.clip((arr - vmin) / rng, 0.0, 1.0).astype(np.float32)
    norm_arr[nan_mask] = 0.0          # will be made transparent below

    # --- Map to uint8 LUT index (0-255) ---
    cmap = plt.get_cmap(cmap_name)
    lut  = (np.array([cmap(i / 255) for i in range(256)],
                     dtype=np.float32) * 255).astype(np.uint8)   # 256 × 4 uint8

    idx = (norm_arr * 255).astype(np.uint8)   # H × W  uint8 indices
    rgba_uint8 = lut[idx]                      # H × W × 4  uint8  ← tiny!

    # Transparent where nodata
    rgba_uint8[nan_mask, 3] = 0

    img = Image.fromarray(rgba_uint8, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def raster_overlay_map(raster_path, cmap_name, vmin=None, vmax=None,
                       zoom_start=8, opacity=0.75, tiles="CartoDB dark_matter",
                       nodata=None):
    """
    Build a Folium map with the given GeoTIFF overlaid as a coloured image layer.
    Returns (folium.Map, band_array, bounds).
    """
    band, bounds = load_raster_4326(raster_path)
    west, south, east, north = bounds

    center_lat = (south + north) / 2
    center_lon = (west  + east)  / 2

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=zoom_start, tiles=tiles)

    # Satellite imagery base layer (toggle)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

    # Raster image overlay
    b64 = array_to_png_base64(band, cmap_name, vmin=vmin, vmax=vmax, nodata=nodata)
    img_url = f"data:image/png;base64,{b64}"

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        name="Risk Layer",
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m, band, bounds


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data(show_spinner=False)
def load_vector_data():
    try:
        stats    = pd.read_csv(stats_csv)
        districts= gpd.read_file(district_shp)
        return stats, districts
    except Exception:
        return None, None

stats, districts = load_vector_data()

# ---------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 System Overview")

    if stats is not None:
        total_districts = len(stats)
        avg_risk        = round(stats["mean_risk"].mean(), 3)
        max_risk        = round(stats["max_risk"].max(), 3)
        max_district    = stats.loc[stats["max_risk"].idxmax(), "NAME_2"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Districts Monitored",  total_districts)
        col2.metric("Average Crop Stress",  avg_risk)
        col3.metric("Maximum Risk Score",   max_risk)
        col4.metric("Highest Risk District", max_district)

        st.divider()
        st.subheader("District Crop Stress Statistics")
        st.dataframe(stats, width="stretch")
    else:
        st.warning("District statistics file not found. Run `district_risk_statistics.py` first.")

# ---------------------------------------------------
# PAGE: DISTRICT ANALYSIS
# ---------------------------------------------------

elif page == "District Analysis":

    st.header("📈 District Crop Stress Analysis")

    if stats is None:
        st.warning("District statistics not found. Run `district_risk_statistics.py` first.")
    else:
        district = st.selectbox("Select District", sorted(stats["NAME_2"].unique()))

        row = stats[stats["NAME_2"] == district]
        if row.empty:
            st.error(f"No data found for {district}.")
        else:
            row = row.iloc[0]
            mean_risk = float(row["mean_risk"])
            max_risk  = float(row["max_risk"])
            min_risk  = float(row["min_risk"]) if "min_risk" in row.index else 0.0

            # ── Auto-detect scale: 0-100 vs 0-1 ──────────────────────────
            # The risk raster may store values as 0-100 (float from NDVI inversion)
            # or normalised 0-1. Detect from the max value across all districts.
            global_max = float(stats["max_risk"].max())
            if global_max > 1.5:
                # 0-100 scale — normalise display to 0-100
                y_max     = 110
                x_max     = global_max * 1.05
                scale_lbl = "(0 – 100 index)"
            else:
                # Already 0-1
                y_max     = 1.15
                x_max     = 1.05
                scale_lbl = "(0 – 1 index)"

            # ── KPI metrics ───────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            threshold = 50 if global_max > 1.5 else 0.5
            c1.metric("Mean Risk Score", f"{mean_risk:.3f}",
                      delta="⚠ High" if mean_risk >= threshold else "✓ Low",
                      delta_color="inverse")
            c2.metric("Max Risk Score", f"{max_risk:.3f}")
            c3.metric("Min Risk Score", f"{min_risk:.3f}")

            st.markdown("---")

            # ── Use tabs instead of columns so neither chart is squashed ──
            tab1, tab2 = st.tabs(["📊 Selected District", "📋 All Districts Ranked"])

            # ── Tab 1: bar chart for selected district ────────────────────
            with tab1:
                fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
                fig_bar.patch.set_facecolor("#1a2620")
                ax_bar.set_facecolor("#1a2620")

                labels_v = ["Mean Risk", "Max Risk", "Min Risk"]
                values_v = [mean_risk, max_risk, min_risk]
                colors_v = ["#f5c518", "#e05252", "#4cde80"]

                x_pos = range(len(labels_v))
                rects = ax_bar.bar(x_pos, values_v,
                                   color=colors_v, width=0.45,
                                   edgecolor="#ffffff", linewidth=0.4,
                                   zorder=3)

                # Grid lines so bars are visible even if short
                ax_bar.yaxis.grid(True, color="#2e4038", linewidth=0.6, zorder=0)
                ax_bar.set_axisbelow(True)
                ax_bar.set_ylim(0, y_max)
                ax_bar.set_xticks(list(x_pos))
                ax_bar.set_xticklabels(labels_v, color="#e8f5ed", fontsize=12)
                ax_bar.set_ylabel(f"Risk Score {scale_lbl}",
                                  color="#e8f5ed", fontsize=10)
                ax_bar.set_title(f"Crop Stress Scores — {district}",
                                 color="#e8f5ed", fontsize=13, pad=12)
                ax_bar.tick_params(axis="y", colors="#a0c8a8", labelsize=9)
                ax_bar.tick_params(axis="x", length=0)
                for spine in ax_bar.spines.values():
                    spine.set_edgecolor("#2e4038")

                # Value labels above each bar
                for rect, val in zip(rects, values_v):
                    label_y = rect.get_height() + y_max * 0.02
                    ax_bar.text(
                        rect.get_x() + rect.get_width() / 2,
                        label_y,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        color="#ffffff", fontsize=12, fontweight="bold"
                    )

                fig_bar.subplots_adjust(
                    left=0.13, right=0.97, top=0.88, bottom=0.12
                )
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # ── Tab 2: all districts ranked ───────────────────────────────
            with tab2:
                ranked  = stats.sort_values("mean_risk", ascending=True).reset_index(drop=True)
                n_d     = len(ranked)

                # Fixed height per row so labels never collapse
                row_px  = 0.36
                fig_h   = max(5.0, n_d * row_px + 1.2)
                left_pad = 0.38   # fixed — wide enough for longest district name

                fig_rank, ax_rank = plt.subplots(figsize=(9, fig_h))
                fig_rank.patch.set_facecolor("#1a2620")
                ax_rank.set_facecolor("#1a2620")

                rank_cols = [
                    "#e05252" if n == district else "#4cde80"
                    for n in ranked["NAME_2"]
                ]
                ax_rank.barh(
                    ranked["NAME_2"], ranked["mean_risk"],
                    color=rank_cols, edgecolor="none", height=0.65, zorder=3
                )
                ax_rank.xaxis.grid(True, color="#2e4038",
                                   linewidth=0.6, zorder=0)
                ax_rank.set_axisbelow(True)
                ax_rank.set_xlim(0, x_max)
                ax_rank.set_xlabel(f"Mean Risk Score {scale_lbl}",
                                   color="#e8f5ed", fontsize=10)
                ax_rank.set_title("All Districts — Mean Risk Ranked",
                                  color="#e8f5ed", fontsize=12, pad=10)
                ax_rank.tick_params(axis="y", colors="#e8f5ed", labelsize=8)
                ax_rank.tick_params(axis="x", colors="#a0c8a8", labelsize=8)
                for spine in ax_rank.spines.values():
                    spine.set_edgecolor("#2e4038")

                # Dashed line + label at selected district's value
                sel_val = float(
                    ranked.loc[ranked["NAME_2"] == district, "mean_risk"].iloc[0]
                )
                ax_rank.axvline(sel_val, color="#f5c518",
                                linewidth=1.5, linestyle="--", alpha=0.9,
                                label=f"{district}: {sel_val:.3f}")
                ax_rank.legend(
                    fontsize=8,
                    facecolor="#1a2620",
                    edgecolor="#2e4038",
                    labelcolor="#e8f5ed",
                    loc="lower right",
                )

                fig_rank.subplots_adjust(
                    left=left_pad, right=0.97, top=0.96, bottom=0.08
                )
                st.pyplot(fig_rank)
                plt.close(fig_rank)

# ---------------------------------------------------
# PAGE: INTERACTIVE MONITORING MAP
# ---------------------------------------------------

elif page == "Interactive Monitoring Map":

    st.header("🛰 Interactive Crop Stress Monitoring Map")

    if districts is not None:
        districts_4326 = districts.to_crs(epsg=4326)

        # Project to a metric CRS for accurate centroid, then convert back
        districts_proj = districts_4326.to_crs(epsg=32643)   # UTM zone 43N covers Gujarat
        centroids_proj = districts_proj.geometry.centroid
        centroids_geo  = centroids_proj.to_crs(epsg=4326)
        center_lat = centroids_geo.y.mean()
        center_lon = centroids_geo.x.mean()

        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=7, tiles="CartoDB dark_matter")

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite", overlay=False, control=True
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

        folium.Choropleth(
            geo_data=districts_4326.__geo_interface__,
            data=stats,
            columns=["NAME_2", "mean_risk"],
            key_on="feature.properties.NAME_2",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.4,
            legend_name="Crop Stress Risk Index",
            name="Choropleth Risk Layer",
        ).add_to(m)

        # Tooltips
        tooltip = folium.GeoJsonTooltip(
            fields=["NAME_2", "mean_risk", "max_risk"],
            aliases=["District:", "Mean Risk:", "Max Risk:"],
            localize=True,
        )
        folium.GeoJson(
            districts_4326.__geo_interface__,
            name="District Boundaries",
            style_function=lambda f: {
                "fillOpacity": 0,
                "color": "#ffffff",
                "weight": 0.5,
            },
            tooltip=tooltip,
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=1100, height=600)
    else:
        st.warning("Shapefile not found. Run `district_risk_statistics.py` first.")

# ---------------------------------------------------
# PAGE: SATELLITE RISK MAP  ← now shown on real map
# ---------------------------------------------------

elif page == "Satellite Risk Map":

    st.header("🌿 Crop Stress Risk Map")

    if not os.path.exists(risk_raster):
        st.warning("Crop Stress Risk Map TIF not found. Run `ndvi_calculation.py` first.")
    else:
        st.info("🗺 Raster overlaid on interactive map — toggle layers in the top-right corner.")

        col_op, col_cmap = st.columns(2)
        opacity = col_op.slider("Layer Opacity", 0.1, 1.0, 0.75, 0.05,
                                key="risk_opacity")
        cmap_choice = col_cmap.selectbox("Colour Map",
                                         ["RdYlGn_r", "hot_r", "YlOrRd", "viridis_r"],
                                         key="risk_cmap")

        with st.spinner("Reprojecting and rendering raster…"):
            m, band, bounds = raster_overlay_map(
                risk_raster,
                cmap_name=cmap_choice,
                vmin=0, vmax=100,
                zoom_start=8,
                opacity=opacity,
                nodata=None,
            )

        st_folium(m, width=1100, height=600)

        st.divider()
        st.subheader("Raster Statistics")
        valid = band[~np.isnan(band)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min Value",  f"{np.nanmin(valid):.2f}")
        c2.metric("Mean Value", f"{np.nanmean(valid):.2f}")
        c3.metric("Max Value",  f"{np.nanmax(valid):.2f}")
        c4.metric("Std Dev",    f"{np.nanstd(valid):.2f}")

        # Colorbar legend
        fig_leg, ax_leg = plt.subplots(figsize=(6, 0.4))
        fig_leg.patch.set_facecolor("#101915")
        ax_leg.set_facecolor("#101915")
        cb = matplotlib.colorbar.ColorbarBase(
            ax_leg,
            cmap=plt.get_cmap(cmap_choice),
            norm=mcolors.Normalize(vmin=0, vmax=100),
            orientation="horizontal",
        )
        cb.set_label("Crop Stress Risk Score", color="#d4f0de")
        ax_leg.tick_params(colors="#5a8a6a")
        fig_leg.subplots_adjust(left=0.05, right=0.95, top=0.7, bottom=0.3)
        st.pyplot(fig_leg)

# ---------------------------------------------------
# PAGE: HOTSPOT DETECTION
# ---------------------------------------------------

elif page == "Hotspot Detection":

    st.header("🔥 Crop Stress Hotspots")

    if not os.path.exists(hotspot_raster):
        st.warning("Hotspot TIF not found. Run `stress_hotspot_detection.py` first.")
    else:
        col_op2, col_r, col_b = st.columns(3)
        opacity_h = col_op2.slider("Heat opacity", 0.1, 1.0, 0.70, 0.05, key="hot_op")
        radius    = col_r.slider("Heat radius",   5,  30, 15,    key="hot_rad")
        blur      = col_b.slider("Heat blur",     5,  40, 20,    key="hot_blur")

        with st.spinner("Loading hotspot raster…"):
            try:
                band_h, bounds_h = load_raster_4326(hotspot_raster)
                west_h, south_h, east_h, north_h = bounds_h

                center_lat_h = (south_h + north_h) / 2
                center_lon_h = (west_h  + east_h)  / 2

                m_h = folium.Map(
                    location=[center_lat_h, center_lon_h],
                    zoom_start=8,
                    tiles="CartoDB dark_matter",
                )

                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                          "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri", name="Satellite",
                    overlay=False, control=True,
                ).add_to(m_h)
                folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m_h)

                # Build heatmap point list
                rows_idx, cols_idx = np.where(
                    (~np.isnan(band_h)) & (band_h > 0)
                )

                if len(rows_idx) == 0:
                    st.warning(
                        "No hotspot pixels (value > 0) found in the raster. "
                        "Check that `stress_hotspot_detection.py` ran successfully "
                        "and that the risk threshold is not too high."
                    )
                else:
                    h_h, w_h = band_h.shape
                    # Convert pixel indices → geographic coordinates
                    lats_h = north_h + rows_idx * (south_h - north_h) / h_h
                    lons_h = west_h  + cols_idx * (east_h  - west_h)  / w_h

                    intensities_h = band_h[rows_idx, cols_idx].astype(float)
                    max_int_h     = intensities_h.max()
                    if max_int_h <= 0:
                        max_int_h = 1.0

                    heat_data_h = [
                        [float(la), float(lo), float(v) / max_int_h]
                        for la, lo, v in zip(lats_h, lons_h, intensities_h)
                    ]

                    # NOTE: gradient keys MUST be strings for Leaflet.heat
                    HeatMap(
                        heat_data_h,
                        radius=radius,
                        blur=blur,
                        min_opacity=0.3,
                        gradient={
                            "0.3": "#ffff00",
                            "0.6": "#ff6b35",
                            "1.0": "#ff0000",
                        },
                        name="Hotspot Heatmap",
                    ).add_to(m_h)

                    folium.LayerControl(collapsed=False).add_to(m_h)
                    st_folium(m_h, width=1100, height=600, key="hotspot_map")

                    n_clusters = len(np.unique(intensities_h.astype(int)))
                    st.success(
                        f"✅ **{len(heat_data_h):,}** hotspot pixels detected across "
                        f"**{n_clusters}** intensity clusters."
                    )

                    # Cluster intensity bar chart
                    st.markdown("---")
                    st.subheader("Hotspot Intensity Distribution")
                    fig_hs, ax_hs = plt.subplots(figsize=(8, 3))
                    fig_hs.patch.set_facecolor("#101915")
                    ax_hs.set_facecolor("#101915")
                    ax_hs.hist(
                        intensities_h / max_int_h, bins=40,
                        color="#ff6b35", edgecolor="none", alpha=0.85
                    )
                    ax_hs.set_xlabel("Normalised Hotspot Intensity", color="#d4f0de")
                    ax_hs.set_ylabel("Pixel Count", color="#d4f0de")
                    ax_hs.set_title("Hotspot Pixel Intensity Histogram", color="#d4f0de")
                    ax_hs.tick_params(colors="#5a8a6a")
                    for spine in ax_hs.spines.values():
                        spine.set_edgecolor("#1e3028")
                    fig_hs.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18)
                    st.pyplot(fig_hs)
                    plt.close(fig_hs)

            except Exception as e:
                st.error(f"Error rendering hotspot map: {e}")
                st.exception(e)

# ---------------------------------------------------
# PAGE: TEMPORAL CHANGE ANALYSIS  ← now shown on real map
# ---------------------------------------------------

elif page == "Temporal Change Analysis":

    st.header("📈 Temporal Vegetation Change Analysis")

    if not os.path.exists(temporal_raster):
        st.warning("Temporal Change TIF not found. Run `temporal_change_detection.py` first.")
    else:
        st.info(
            "🗺 NDVI change map overlaid on interactive satellite basemap. "
            "Blue = vegetation improvement · Red = stress increase."
        )

        col_op3, col_cm3 = st.columns(2)
        opacity_t = col_op3.slider("Layer Opacity", 0.1, 1.0, 0.75, 0.05,
                                   key="temp_opacity")
        cmap_t    = col_cm3.selectbox("Colour Map",
                                      ["RdBu", "coolwarm", "bwr", "PiYG"],
                                      key="temp_cmap")

        with st.spinner("Reprojecting temporal change raster…"):
            m, band_t, bounds_t = raster_overlay_map(
                temporal_raster,
                cmap_name=cmap_t,
                vmin=0, vmax=3,          # categories 0,1,2,3
                zoom_start=8,
                opacity=opacity_t,
                nodata=None,
            )

        st_folium(m, width=1100, height=600)

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Change Category Legend")
            legend_df = pd.DataFrame({
                "Value": [0, 1, 2, 3],
                "Category": [
                    "Severe Stress Increase (ΔNDVI < −0.20)",
                    "Mild Stress Increase  (−0.20 ≤ ΔNDVI < −0.05)",
                    "Stable                (−0.05 ≤ ΔNDVI ≤ 0.05)",
                    "Vegetation Improvement (ΔNDVI > 0.05)",
                ],
                "Colour": ["🔴 Red", "🟠 Orange", "🟡 Yellow", "🔵 Blue"],
            })
            st.dataframe(legend_df, hide_index=True)

        with col_r:
            st.subheader("Pixel Distribution")
            unique_vals, counts = np.unique(
                band_t[~np.isnan(band_t)].astype(int), return_counts=True
            )
            cat_map = {0: "Severe Decline", 1: "Mild Decline",
                       2: "Stable", 3: "Improvement"}
            labels  = [cat_map.get(v, str(v)) for v in unique_vals]
            colors  = ["#ff3b3b", "#ff6b35", "#ffcc00", "#2dff8f"]
            total_p = counts.sum()

            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            fig_pie.patch.set_facecolor("#101915")
            ax_pie.set_facecolor("#101915")
            wedges, texts, autotexts = ax_pie.pie(
                counts, labels=labels, autopct="%1.1f%%",
                colors=colors[:len(unique_vals)],
                textprops={"color": "#d4f0de", "fontsize": 9},
                startangle=140,
            )
            ax_pie.set_title("Change Area Distribution", color="#d4f0de")
            fig_pie.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
            st.pyplot(fig_pie)

# ---------------------------------------------------
# PAGE: VEGETATION TIME SERIES
# ---------------------------------------------------

elif page == "Vegetation Time Series":

    st.header("🛰 Vegetation Change Time-Lapse")

    if not os.path.exists(temporal_raster):
        st.warning("Temporal TIF not found. Run `temporal_change_detection.py` first.")
    else:
        st.info(
            "Each frame progressively reveals the NDVI change raster — "
            "simulating a temporal scan from north to south over the study region."
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        steps     = col_s1.slider("Animation Frames", 4, 12, 8,   key="ts_steps")
        cmap_ts   = col_s2.selectbox("Colour Map",
                                     ["coolwarm", "RdBu", "bwr", "PiYG"],
                                     key="ts_cmap")
        delay_s   = col_s3.slider("Frame Delay (s)", 0.2, 2.0, 0.6, 0.1, key="ts_delay")

        play = st.button("▶ Play Vegetation Time-Lapse", key="ts_play_btn")

        # Load raster once (cached)
        with st.spinner("Loading temporal raster…"):
            band_v, bounds_v = load_raster_4326(temporal_raster)

        west_v, south_v, east_v, north_v = bounds_v
        h_v, w_v = band_v.shape

        # Pre-compute global vmin/vmax so colour scale is consistent across frames
        v_valid = band_v[~np.isnan(band_v)]
        vmin_v  = float(np.nanmin(v_valid)) if len(v_valid) else 0.0
        vmax_v  = float(np.nanmax(v_valid)) if len(v_valid) else 3.0

        # Placeholders that stay fixed on the page
        frame_label   = st.empty()
        img_slot      = st.empty()          # single image slot — no key conflicts
        progress_bar  = st.progress(0)

        def render_frame(band_slice, frame_num, total):
            """Render one animation frame as a matplotlib figure → st.image bytes."""
            fig_f, ax_f = plt.subplots(figsize=(10, 6))
            fig_f.patch.set_facecolor("#0a0f0d")
            ax_f.set_facecolor("#0a0f0d")

            # Build display array — NaN rows appear transparent / background colour
            display = np.where(np.isnan(band_slice), np.nan, band_slice)

            im_f = ax_f.imshow(
                display,
                cmap=cmap_ts,
                vmin=vmin_v, vmax=vmax_v,
                extent=[west_v, east_v, south_v, north_v],
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            ax_f.set_xlabel("Longitude", color="#d4f0de", fontsize=9)
            ax_f.set_ylabel("Latitude",  color="#d4f0de", fontsize=9)
            ax_f.set_title(
                f"NDVI Change Map — Frame {frame_num}/{total}  "
                f"({int(100 * frame_num / total)}% revealed)",
                color="#d4f0de", fontsize=11
            )
            ax_f.tick_params(colors="#5a8a6a", labelsize=8)
            for spine in ax_f.spines.values():
                spine.set_edgecolor("#1e3028")

            cbar = fig_f.colorbar(im_f, ax=ax_f, fraction=0.025, pad=0.02)
            cbar.set_label("Change Category", color="#d4f0de", fontsize=9)
            cbar.ax.tick_params(colors="#5a8a6a")
            cbar.set_ticks([vmin_v, vmax_v])
            cbar.set_ticklabels(["Decline", "Improvement"])

            # Add a legend annotation
            cat_text = (
                "0: Severe Stress  1: Mild Stress\n"
                "2: Stable  3: Improvement"
            )
            ax_f.text(
                0.01, 0.01, cat_text,
                transform=ax_f.transAxes,
                fontsize=7, color="#d4f0de",
                verticalalignment="bottom",
                bbox=dict(facecolor="#101915", alpha=0.7,
                          edgecolor="#1e3028", boxstyle="round,pad=0.3"),
            )

            fig_f.subplots_adjust(left=0.08, right=0.94, top=0.91, bottom=0.09)

            buf_f = io.BytesIO()
            fig_f.savefig(buf_f, format="png", dpi=120,
                          facecolor="#0a0f0d", bbox_inches="tight")
            buf_f.seek(0)
            plt.close(fig_f)
            return buf_f

        if play:
            for i in range(1, steps + 1):
                # Reveal top i/steps rows; rest stays NaN (transparent)
                cut         = max(1, int(h_v * i / steps))
                slice_arr   = np.full_like(band_v, np.nan)
                slice_arr[:cut, :] = band_v[:cut, :]

                frame_label.markdown(
                    f"**Frame {i}/{steps}** — "
                    f"{int(100 * i / steps)}% of study area revealed"
                )
                buf_frame = render_frame(slice_arr, i, steps)
                img_slot.image(buf_frame, use_container_width=True)
                progress_bar.progress(i / steps)
                time.sleep(delay_s)

            frame_label.success(
                "✅ Animation complete — full NDVI change map displayed."
            )
        else:
            # Static full-frame view before play is pressed
            frame_label.markdown(
                "Full NDVI change map shown below. "
                "Press **▶ Play** to animate frame by frame."
            )
            buf_static = render_frame(band_v, steps, steps)
            img_slot.image(buf_static, use_container_width=True)
            progress_bar.progress(1.0)

# ---------------------------------------------------
# PAGE: DISTRICT HEATMAP  ← regenerate from stats CSV
# ---------------------------------------------------

elif page == "District Heatmap":

    st.header("🗺 District Crop Stress Heatmap")

    # Try loading saved image first; if missing, regenerate from stats CSV
    if os.path.exists(heatmap_img):
        # Resize to a safe display resolution before handing to Streamlit
        try:
            with Image.open(heatmap_img) as raw_img:
                # Downsample to max 2000px wide for safe display
                max_w = 2000
                w, h  = raw_img.size
                if w > max_w:
                    ratio    = max_w / w
                    new_size = (max_w, max(1, int(h * ratio)))
                    display_img = raw_img.convert("RGB").resize(
                        new_size, Image.LANCZOS
                    )
                else:
                    display_img = raw_img.convert("RGB")
                buf_disp = io.BytesIO()
                display_img.save(buf_disp, format="PNG")
                buf_disp.seek(0)
            st.image(buf_disp, use_container_width=True,
                     caption="District Crop Stress Heatmap (district_crop_stress_heatmap.png)")
        except Exception as e:
            st.error(f"Could not load heatmap image: {e}")

    elif stats is not None:
        st.info("Heatmap PNG not found — generating from district statistics CSV.")

        with st.spinner("Generating heatmap…"):
            metrics = ["mean_risk", "max_risk"]
            available_metrics = [m for m in metrics if m in stats.columns]

            if available_metrics:
                heat_data = stats.set_index("NAME_2")[available_metrics]
                col_labels = [c.replace("_", " ").title() for c in available_metrics]

                fig_h, ax_h = plt.subplots(
                    figsize=(max(8, len(available_metrics) * 3),
                             max(6, len(heat_data) * 0.45))
                )
                fig_h.patch.set_facecolor("#0a0f0d")
                ax_h.set_facecolor("#0a0f0d")

                arr = heat_data.values.astype(float)
                im = ax_h.imshow(arr, cmap="YlOrRd", aspect="auto",
                                  vmin=0, vmax=arr.max())

                ax_h.set_xticks(range(len(col_labels)))
                ax_h.set_xticklabels(col_labels, rotation=30,
                                      ha="right", color="#d4f0de", fontsize=10)
                ax_h.set_yticks(range(len(heat_data)))
                ax_h.set_yticklabels(heat_data.index, color="#d4f0de", fontsize=8)

                # Annotate cells
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        v = arr[i, j]
                        text_color = "black" if v > arr.max() * 0.5 else "white"
                        ax_h.text(j, i, f"{v:.3f}",
                                  ha="center", va="center",
                                  fontsize=7, color=text_color)

                plt.colorbar(im, ax=ax_h, label="Risk Score", shrink=0.8)
                ax_h.set_title("District Crop Stress Heatmap",
                                color="#d4f0de", fontsize=14, pad=12)
                fig_h.subplots_adjust(left=0.22, right=0.92, top=0.93, bottom=0.1)

                st.pyplot(fig_h)

                # Offer to save
                buf_h = io.BytesIO()
                fig_h.savefig(buf_h, format="png", dpi=150,
                               facecolor="#0a0f0d")
                buf_h.seek(0)
                st.download_button(
                    "⬇ Download Heatmap PNG",
                    data=buf_h,
                    file_name="district_crop_stress_heatmap.png",
                    mime="image/png",
                )
            else:
                st.error("Stats CSV does not contain expected risk columns.")
    else:
        st.warning(
            "Neither the heatmap PNG nor the statistics CSV was found. "
            "Run `district_risk_statistics.py` first."
        )

# ---------------------------------------------------
# PAGE: AI TECHNICAL REPORT
# ---------------------------------------------------

elif page == "AI Technical Report":

    st.header("📄 AI Crop Health Technical Report")

    if os.path.exists(report_pdf):
        with open(report_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            f'width="100%" height="700" type="application/pdf"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)

        with open(report_pdf, "rb") as f:
            st.download_button("⬇ Download PDF Report", f,
                               file_name="AI_Crop_Health_Technical_Report.pdf",
                               mime="application/pdf")
    else:
        st.warning(
            "PDF report not found. Run `generate_ai_report.py` to produce it."
        )
        st.info("Showing inline summary instead:")

        if stats is not None:
            st.subheader("Key Findings")
            top5 = stats.sort_values("mean_risk", ascending=False).head(5)
            st.dataframe(top5, width="stretch")
