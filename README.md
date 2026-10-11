# PoC-DinSAR
Automated Sentinel-1 DInSAR processing pipeline using ESA SNAP GPT and SNAPHU for ground displacement extraction and EGMS-compatible data output.

# Sentinel-1 DInSAR Ground Displacement Monitoring — Proof of Concept

Automated end-to-end processing pipeline for extracting ground and infrastructure surface displacement using **Sentinel-1 SAR** imagery via **Differential Interferometry (DInSAR)**. 

The pipeline runs headless using **ESA SNAP Graph Processing Tool (GPT)** and **SNAPHU**, converting raw satellite interferograms into analysis-ready vector points (`.csv` / `.geojson`) formatted for compatibility with the **European Ground Motion Service (EGMS)** standard.

---
## Purpose & Scope

* **Proof of Concept:** Validating millimetric vertical and Line-of-Sight (LOS) displacement detection over coastal and urban study areas (e.g., Gdynia / Redłowo Cliff, Poland).
* **Headless Automation:** Eliminating SNAP GUI manual interactions by driving processing graphs through parameterized XML templates orchestrated with Python.
* **Downstream Integration:** Generating structured tabular datasets to feed spatial databases (PostgreSQL/PostGIS) and web-based analytical dashboards without intermediate manual steps.

---
