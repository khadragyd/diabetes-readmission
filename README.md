# Uncertainty-aware explainable prediction of 30-day readmission in patients with diabetes

Analysis code for:

> Khadragy, S. *Beyond Point Predictions: An Uncertainty-Aware Explainable AI Framework for Trustworthy 30-Day Hospital Readmission Prediction in Patients With Diabetes Using Conformal Prediction and SHAP.*

`run_analysis.py` reproduces every table and figure in the manuscript from the public data file in a single run:

| Step | Output |
|---|---|
| Cohort construction (first encounter per patient; death/hospice excluded) | `results/run_metadata.json` (cohort flow), `results/table4_characteristics.csv` |
| Stratified 60/20/20 split (training / conformal calibration / test) | `results/run_metadata.json` |
| Hyperparameter tuning of logistic regression, Random Forest, XGBoost and LightGBM (5-fold stratified CV, AUROC) | `results/run_metadata.json` |
| Cross-validated sigmoid calibration of XGBoost | `results/table5_performance.csv` |
| Test-set discrimination, calibration and 1,000-resample bootstrap CIs | `results/table5_performance.csv`, `results/fig3_discrimination.png` |
| Classification threshold (Youden's index on the conformal-calibration set) | `results/run_metadata.json` |
| Label-conditional (Mondrian) split conformal prediction at 80/90/95/99% | `results/table6_conformal.csv`, `results/fig4_conformal_tradeoff.png` |
| SHAP global, patient-level and uncertainty-group analyses | `results/fig5_shap_beeswarm.png`, `results/fig6_shap_waterfalls.png`, `results/fig7_shap_by_group.png`, `results/shap_*.csv` |

## Data

Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository,
<https://doi.org/10.24432/C5230J> (Clore, Cios, DeShazo & Strack, 2014; licensed CC BY 4.0).
A copy of `diabetic_data.csv` and `IDS_mapping.csv` is included in `data/` unchanged.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py --data data/diabetic_data.csv --out results
```

Runtime is about 13 minutes on a 2-core machine. The random seed is fixed (`SEED = 42`); the results in `results/` were produced with the package versions in `requirements.txt` (Python 3.13). Different library versions can change the last decimal places.

## Licence

Code: MIT. Data: CC BY 4.0 (original authors).
