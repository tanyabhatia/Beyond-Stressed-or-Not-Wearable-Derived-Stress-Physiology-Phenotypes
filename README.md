## Overview

This codebase implements a three-step pipeline:

1. **Step 1** (`Step1.py`): BSON-to-Parquet conversion and per-modality data extraction
2. **Step 2** (`Step2.py`): Feature engineering and aggregation (sleep, HR, HRV, SRI)
3. **Step 3** (`Step3.ipynb`): Complete stress-physiology analysis pipeline
   - Sleep phenotyping and visualization
   - Physiological feature aggregation  
   - Stress-centered physiology extraction
   - PCA-based dimensionality reduction
   - Gaussian Mixture Modeling clustering (Recovery vs Activation modes)
   - Leave-one-subject-out and bootstrap validation
   - Negative control analysis (100 Monte Carlo draws)
   - Mixed-effects regression with per-participant phenotyping

## Data Requirements

- **LifeSnaps BSON dump**: `fitbit.bson` (raw MongoDB export)
- **Stress surveys**: CSV with columns `user_id`, `submitdate`, `stai_stress`
- **Directory structure**:
  ```
  ./data/lifesnaps/
  ├── raw/
  │   └── fitbit.bson
  └── processed/
      ├── PreProcess/
      │   ├── Sleep/<participant_id>/Sleep.parquet
      │   ├── HR/<participant_id>/HR.parquet
      │   └── HRV/<participant_id>/HRV.parquet
      └── PostProcess/PostDataset/
          ├── Sleep_features.parquet
          ├── HR_features.parquet
          ├── HRV_features.parquet
          └── ... (other modalities)
  ```

## Quick Start

### 1. Raw Data Extraction
```bash
python Step1.py
```
Converts BSON to Parquet and extracts per-modality time series.

### 2. Feature Engineering
```bash
python Step2.py
```
Computes daily sleep metrics, aggregates physiological features, and calculates Sleep Regularity Index.

### 3. Stress-Physiology Analysis
```bash
python Step3.py
```
Runs complete analysis pipeline: sleep phenotyping → physiological aggregation → stress-centered extraction → PCA+GMM clustering → validation → mixed-effects modeling.

## References

Lautman, Z., Shah, K. U., Shipley, H., Poe, J., Songphatanayothin, T., & Snyder, M. P. (2026).
Beyond 'Stressed' or 'Not': Wearable-Derived Stress-Physiology Phenotypes. *In-prep*.

LifeSnaps dataset: Yfantidou, S., *et al.* (2022). LifeSnaps, a 4-month multi-modal dataset. *Scientific Data*, 9, 663. https://doi.org/10.5281/zenodo.6826682

