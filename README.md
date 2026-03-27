# 🌾 AI Crop Health Monitoring System

An AI-driven crop monitoring platform that uses satellite imagery, vegetation indices, and geospatial analytics to detect crop stress, identify agricultural hotspots, and monitor vegetation health across districts.

This system processes satellite data to generate crop stress risk maps, hotspot detection layers, and district-level analytics, which are visualized through an interactive monitoring dashboard.

---

## 📌 Project Overview

Agriculture monitoring is critical for ensuring crop productivity and food security. Traditional field inspection methods are time-consuming and limited in coverage.

This project leverages remote sensing and artificial intelligence to analyze satellite imagery and automatically detect vegetation stress patterns.

The system:
- Processes satellite vegetation data
- Detects crop stress regions
- Identifies high-risk agricultural zones
- Generates district-level statistics
- Produces automated analytical reports
- Displays results through an interactive dashboard

---

## 🚀 Key Features

### 🌱 Satellite Crop Stress Detection
Analyzes satellite vegetation data to identify areas experiencing potential crop stress.

### 🔥 Hotspot Detection
Detects agricultural zones with high stress concentration using spatial analysis.

### 🗺 District-Level Risk Analysis
Calculates crop stress statistics for each district using zonal statistics.

### 📈 Vegetation Change Monitoring
Tracks changes in vegetation health using temporal analysis.

### 📊 Interactive Monitoring Dashboard
Provides a real-time visualization interface to explore agricultural health data.

### 📄 Automated AI Report Generation
Automatically generates a technical PDF report summarizing the crop health analysis.

---

## 🛰 System Architecture

```
Satellite Data
      ↓
Vegetation Index Calculation
      ↓
Crop Stress Risk Modeling
      ↓
Hotspot Detection
      ↓
District Zonal Statistics
      ↓
AI Report Generation
      ↓
Interactive Monitoring Dashboard
```

---

## 📊 Dashboard Modules

The dashboard contains several analytical components.

### 📊 System Overview
Displays key performance indicators including:
- Districts monitored
- Average crop stress
- Maximum stress level
- Highest risk district

### 📈 District Analysis
Allows users to select a district and analyze its crop stress levels.

### 🛰 Interactive Monitoring Map
Displays crop stress levels spatially using district boundaries.

### 🌿 Satellite Crop Stress Map
Visualizes the raster-based crop stress risk derived from satellite imagery.

### 🔥 Crop Stress Hotspot Detection
Identifies areas with concentrated vegetation stress using heatmap visualization.

### 📈 Vegetation Change Analysis
Shows vegetation health changes across the monitored region.

### 📊 NDVI Vegetation Time Series
Simulates vegetation health trends over time.

### 🗺 District Heatmap
Displays a spatial heatmap of crop stress levels across districts.

### 📄 AI Technical Report
Automatically generated report summarizing:
- Crop health conditions
- Risk levels
- Spatial patterns

---

## 🛠 Technologies Used

### Programming Language
- Python

### Data Processing
- geopandas
- numpy
- pandas
- rasterio
- rasterstats

### Visualization
- matplotlib
- folium
- streamlit

### Dashboard Framework
- Streamlit

### Geospatial Processing
- GIS raster analysis
- Zonal statistics
- Spatial hotspot detection

---

## 📁 Project Structure

```
Crop_Health_Project
│
├── data
│   └── boundaries
│       └── gadm41_IND_2.shp
│
├── outputs
│   ├── Crop_Stress_Risk_Map.tif
│   ├── Crop_Stress_Hotspots.tif
│   ├── Temporal_Stress_Change_Map.tif
│   ├── district_risk_statistics.csv
│   ├── district_crop_risk_map.shp
│   ├── district_crop_stress_heatmap.png
│   └── AI_Crop_Health_Technical_Report.pdf
│
├── scripts
│   ├── crop_stress_risk_map.py
│   ├── stress_hotspot_detection.py
│   ├── district_risk_statistics.py
│   └── generate_ai_report.py
│
├── dashboard.py
└── README.md
```

---

## ⚙ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Crop_Health_Project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit geopandas rasterio rasterstats folium matplotlib pandas numpy
```

---

## ▶ Running the Dashboard

Start the monitoring dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser:

```
http://localhost:8501
```

---

## 📊 Example Outputs

The system generates the following outputs:
- Crop stress risk map
- Hotspot detection map
- Temporal vegetation change map
- District risk statistics
- AI technical report
- Interactive monitoring dashboard

---

## 🎯 Applications

This system can support:
- Precision agriculture monitoring
- Crop health assessment
- Agricultural risk analysis
- Government agricultural planning
- Climate impact monitoring

---

## 📚 Future Improvements

Possible extensions include:
- Real-time satellite data integration
- Machine learning crop classification
- Multi-season vegetation monitoring
- Farmer advisory alerts
- Web deployment for public access

---

## 👨‍💻 Author

Final Year Engineering Project — AI-Based Crop Health Monitoring System