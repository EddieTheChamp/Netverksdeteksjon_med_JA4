# JA4+ Classification Prototype

This repository contains a network traffic classification system based on JA4 and JA4S network fingerprints. It features a hybrid classification pipeline that utilizes a primary deterministic database lookup, with a fallback to a Random Forest machine learning model for unknown traffic.

## Repository Structure

The repository is organized into distinct functional areas:

### 1. `Modeller/` (Core Application & Analysis)
This is the primary directory of the project, containing the active classification system and the tools used to evaluate it.
*   **`prototype/`**: Contains the hybrid classification pipeline and the Graphical User Interface (GUI).
*   **`evaluations/`**: Scripts to independently benchmark different classification models.
*   **`analysis/`**: Utilities to measure system latency, calculate model accuracy, and visualize performance data.
*   **`results/`**: Output directory for generated graphs and evaluation metrics.
*(See `Modeller/README.md` for full details.)*

### 2. `Datasett/`
This directory stores the data used to train the machine learning models and populate the database index for exact matches. It contains the JSON files mapping correlated JA4/JA4S fingerprints to specific network applications.

### 3. `Create Dictionary/`
Contains utility scripts for processing raw network datasets. These tools are used to build and format the structured dictionaries required by the prototype's database lookup stage.

---

## Getting Started

### Requirements
Ensure you have the necessary dependencies installed:
```powershell
pip install -r requirements.txt
```
*(Note: For FoxIO or packet capture features, Wireshark/Tshark must be installed and added to your system's PATH).*

### Running the Application

**To launch the Graphical User Interface:**
```powershell
python Modeller/prototype/app_gui.py
```

**To run the pipeline directly via the command line:**
```powershell
python Modeller/prototype/pipeline_model/pipeline.py --ja4 "t13d1516h2_8a2d1d4d_8a2d1d4d" --ja4s "t130200_1301_8a2d1d4d"
```