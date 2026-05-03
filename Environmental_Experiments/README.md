# FrED: Environmental Case Study — Global Flood Events

This folder contains the implementation and results for applying the **FrED** framework to high-dimensional physical systems. This study focuses on attributing 24-hour lead-time weather forecasts to historical flood events by grounding meteorological data in biological and geographic context.

---

## 🌍 Methodology: Environmental Mapping

To move beyond simple numerical similarity, we combine three data streams into a unified **Environmental Knowledge Graph ($\mathcal{G}_{E}$)**. This allows the framework to link atmospheric states to structural ground conditions via **Direct Geospatial Matching**.

### 1. The Nature Layer (Biological)
Using the **iNaturalist** database, we extract verified sightings of flora and fauna within a 10km radius of flood coordinates. This identifies regional biodiversity signatures, allowing the model to compare geographical regions based on shared ecological habitats.

### 2. The Land Cover Layer (Geographic)
Physical terrain context is provided by **ESA WorldCover** satellite data (10-meter resolution). We analyze land composition (urban, water-absorbing vegetation, agricultural zones) within a 10km radius to evaluate how surface types influence flood severity and vulnerability.

### 🖼️ Physical Exemplar & KG Extraction
The framework pairs continuous atmospheric tensors with discrete structural nodes.


## 📂 Folder Structure

*   **`Env_KG_Data/`**: Contains the source data for the Environmental KG, including iNaturalist sightings and ESA WorldCover statistics linked to 1,372 major flood events.
*   **`reanalysis_data/`**: Subsampled ERA5 reanalysis data collected from **WeatherBench 2** used for training and validation.
*   **`train_weather_encoder.py`**: Implementation of the specialized latent encoder designed to capture physical similarities between atmospheric variables.
*   **`weather_autoencoder_baseline.pth`**: The pre-trained weights for the weather encoder.
*   **`final_results/`**: Sample CSV files containing attribution scores and rankings for Pangu-Weather predictions across 237 major events (2018–2022).

---

## 🛠️ Predictive Setup

We utilize **Pangu-Weather** (3D Earth-specific transformer) forecasts via WeatherBench 2. Attribution is performed on regional spatial patches centered on flood coordinates.

### Atmospheric Variable Subsetting
To preserve the primary meteorological signal while reducing overhead, we isolate:
*   **Surface:** 2m temperature and mean sea-level pressure.
*   **Upper-air:** Temperature, specific humidity, geopotential, and U/V wind components at **1000, 850, 500, and 250 hPa**.
  * *Justification:* Captures moisture transport (1000/850), steering flows (500), and upper-level divergence (250).

---

## 📊 Results & Physical Consistency

Our framework is evaluated on its ability to anchor abstract atmospheric data to correct physical locations through two metrics:

### 1. Regional Match Rate (Geographic Precision)
FrED significantly improves regional localization compared to baseline similarity methods.

| Attribution Method | RMR@10 | RMR@20 | RMR@50 | RMR@100 |
| :--- | :--- | :--- | :--- | :--- |
| CNN Latent Engine (Baseline) | 50.9% | 35.6% | 15.2% | 6.8% |
| **FrED (Proposed)** | **65.7%** | **47.8%** | **22.5%** | **12.0%** |

### 2. Ecological Consistency
When exact geographic matches are unavailable, FrED retrieves **physically consistent analogs** by clustering events via shared environmental biomes (e.g., matching South Sudan to Mali based on savanna profiles).

| Query Country | Matched Country | Frequency |
| :--- | :--- | :--- |
| Central African Republic | Sudan | 7 |
| South Sudan | Mali | 7 |
| Paraguay | Chile | 6 |
