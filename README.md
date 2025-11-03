# Ejection Fraction Quantification from Ungated Chest CT by AI

This repository contains the code and resources for the paper “**Ejection fraction quantification from ungated chest CT by AI**.”

> **TL;DR**: We provide an interpretable AI model that estimate left ventricular ejection fraction from non-contrast, ungated chest CT scans. This README explains the study context and shows how to develop and evaluate the model in Python.


## Background & Objectives

**Backgroud**: Left ventricular ejection fraction (LVEF) is commonly assessed through specilized imaging techniques.

**Objective**: Develop an AI model that predicts LVEF from static, non-contrast, un-electrocardiographic-gated chest CT scans. 

## Key Results (from the paper)

1. **External evaluation with PET LVEF**: AI-derived CT LVEF showed a strong correlation with gated position emission tomography (PET) LVEF (r = 0.84), reaching an area under the curve (AUC) of 0.96 and a negative predictive value of 95% for identifying reduced LVEF (< 40%). The hazard ratios (HRs) of heart failure, cardiovascular death, and all-cause death were similar between PET and AI CT LVEF.

2. **External evaluation with Echocardiographic LVEF**: AI CT LVEF showed a moderate correlation with echocardiographic LVEF (r = 0.72), reaching an AUC of 0.91 and a negative predictive value of 96% for identifying reduced LVEF.

3. **Risk stratification in NLST**: Reduced LVEF (AI CT) was asscoiated with a 13-fold increase in the risk of cardiovascular death and a 4-fold incrase for all-cause death in the National Lung Screening Trial (NLST).

**Conclusion**: Opportunistic AI LVEF from chest CT scans provides not only a quantitative measure of cardiac function but also serves as a powerful predictor of adverse outcomes. 

## Why this matters

- **Clinical impact**: Accurately estimates LVEF from ungated chest CT scans which would enhance cardiovascular screening, enabling early identification of systolic dysfunction in populations undergoing chest CT for non‑cardiac indications (e.g., smokers, oncology surveillance, trauma), without incurring additional costs, radiation, or contrast exposure. .

- **Robust validation**: Externally evaluated with PET and echocardiographic LVEF.

- **Transparency & usability**: Explains major features considered in prediction, supporting physician trust and adoption.   


## Repository Layout

 - `xgboost.py` — training/utility code for the XGBoost‑based model used in the study.
 - `shap.py` — feature interpretation globally and locally.
 - `visualization.py` — plotting function.

### Note

The code used to perform the analyzes described in the article depends also on the following open source repositories:

The cLSTM code is publicly available under a Creative Commons BY-NC license at https://doi.org/10.5281/zenodo.10632288, the TotalSegmentator is available at https://github.com/wasserth/TotalSegmentator, and the XGBoost Python implementation is available at https://xgboost.readthedocs.io/en/stable/python/. 

## Graphical abstract

![Central_figure.png](Figure1.jpeg)



## License

Code: see LICENSE in this repository.

Note that some referenced components are available under their own licenses:

cLSTM (CC BY‑NC): https://doi.org/10.5281/zenodo.10632288

TotalSegmentator: https://github.com/wasserth/TotalSegmentator

XGBoost (Apache‑2.0): https://xgboost.readthedocs.io/en/stable/python/

## Contact 

For questions, please open an issue or contact Dr. Piotr Slomka.