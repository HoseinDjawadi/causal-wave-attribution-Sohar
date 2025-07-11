# Beyond Prediction: A Causal Attribution Framework for Operational Risk in Complex Marine Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code and processed data for the paper: **"Beyond Prediction: A Causal Attribution Framework for Operational Risk in Complex Marine Environments"**, submitted to *Science of The Total Environment*.

### Abstract

Accurate prediction of nearshore wave conditions is a critical challenge in complex ocean systems where unobserved, non-stationary dynamics render forecasting unreliable. This research presents a paradigm shift from prediction to diagnosis, introducing a causal-inferential workflow to extract robust insights from imperfect data. Focusing on Sohar Port, a region of complex hydrodynamics, we first use the PCMCI+ algorithm to infer the system’s underlying causal structure. A systematic modeling experiment then reveals a consistent failure to generalize to unseen data (R²<0.05), a result we interpret not as a model limitation, but as a powerful diagnostic of significant unobserved confounding. Despite the system's unpredictability, formal sensitivity analysis validates the specific causal link from local wind speed to nearshore wave height as exceptionally robust. This validated link serves as the cornerstone for our primary contribution: a Causal Attribution Decision Support System (DSS). This novel tool leverages robust causal knowledge to provide real-time, actionable intelligence on the physical drivers of hazardous wave events.

### Causal Diagnostic Workflow

This study introduces an end-to-end workflow to move from raw data to robust, actionable causal insights.

![Workflow Diagram](figures/workflow_diagram.png)  <!-- You should create a simple workflow diagram for this -->

### Repository Structure

-   **/data**: Contains the final processed datasets used for analysis. A guide to obtaining the raw public data is included.
-   **/notebooks**: Jupyter notebooks to reproduce the entire analysis, from data processing to figure generation.
-   **/src**: Reusable Python functions for data processing, modeling, and plotting.
-   **/figures**: High-resolution versions of all figures presented in the manuscript.

### Installation & Usage

To reproduce the results, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/causal-wave-attribution-sohar.git](https://github.com/YOUR_USERNAME/causal-wave-attribution-sohar.git)
    cd causal-wave-attribution-sohar
    ```

2.  **Create the Conda environment:**
    All required packages are listed in the `environment.yml` file. Create the conda environment using:
    ```bash
    conda env create -f environment.yml
    conda activate causal-waves
    ```

3.  **Run the analysis:**
    The Jupyter notebooks in the `/notebooks` directory are numbered in the order they should be run to reproduce the analysis and generate the figures for the paper.

    -   `1_data_preprocessing.ipynb`: Loads raw data and generates the processed datasets.
    -   `2_causal_discovery.ipynb`: Runs the PCMCI+ algorithm and generates the causal graph.
    -   `3_predictive_modeling.ipynb`: Trains and evaluates the MLR, XGBoost, and CGINN models.
    -   `4_sensitivity_analysis_and_dss.ipynb`: Performs the formal sensitivity analysis and demonstrates the DSS on a case study.

### Data Availability

The final processed datasets (`dss_ready_standardized_dataset_v2.csv` and `dss_event_summary.csv`) are available in the `/data/processed` directory.

The raw data used in this study are from multiple sources:
-   **ERA5 and CMEMS Reanalysis:** These are publicly available. A guide on how to download the specific variables and time ranges used is provided in `/data/raw/README.md`.
-   **In-situ Buoy Data:** The nearshore buoy data used for this study are proprietary and cannot be shared directly. Please contact the corresponding author for inquiries.

### Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{Nikoo2025,
  title   = {Beyond Prediction: A Causal Attribution Framework for Operational Risk in Complex Marine Environments},
  author  = {Nikoo, Mohammad Reza and Etri, Talal and Seyed-Djawadi, Mohamad Hosein and Nazari, Rouzbeh and Al-Rawas, Ghazi},
  journal = {Science of The Total Environment},
  year    = {2025},
  note    = {Under Review}
}