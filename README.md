# MammoAssist

MammoAssist: Context-Aware Multimodal Breast Ultrasound Lesion Classification and Clinical Triage is a multimodal CAD project for breast cancer screening. It combines ultrasound lesion segmentation, image-derived features, electronic health record (EHR) variables, and a safety-focused triage layer to support classification and escalation decisions.

## Project Summary

The system follows a three-stage pipeline:

1. Lesion segmentation using a fine-tuned Segment Anything Model (SAM) backbone.
2. Multimodal feature extraction from the segmented lesion and the patient record.
3. Fusion-based classification and rule-based triage for Routine, Follow-up, or Biopsy recommendations.

The manuscript evaluates the pipeline on 252 histopathologically confirmed breast ultrasound cases, with a 201-case train/validation split and a held-out 51-case test split. The work also includes BI-RADS-excluded ablations to simulate resource-constrained settings where expert assessment is unavailable.

## What Is Included

- [mammoassist-classification.ipynb](mammoassist-classification.ipynb) - end-to-end multimodal classification, ablation, calibration, decision-curve, and triage analysis.
- [mammoassist-segmentation.ipynb](mammoassist-segmentation.ipynb) - lesion segmentation workflow and feature preparation.
- [mammoassist-ehr.ipynb](mammoassist-ehr.ipynb) - EHR preprocessing and tabular modeling.
- [statistical_analysis.py](statistical_analysis.py) - statistical utilities used for confidence intervals and hypothesis testing.
- [global_splits.json](global_splits.json) - precomputed train/validation/test split assignment used to prevent leakage.
- [README.md](README.md) - project overview and usage notes.

## Method Overview

- Segmentation: SAM is fine-tuned for breast ultrasound lesion delineation.
- Image features: the classification notebook builds a 1044-dimensional image representation from deep encoder features plus handcrafted morphology and intensity descriptors.
- EHR features: the clinical branch uses 57 variables with BI-RADS included, and a 56-feature variant without BI-RADS for ablation.
- Fusion strategies: concatenation, gated fusion, bilinear fusion, and cross-attention are compared.
- Clinical decision support: a conservative triage engine flags cases as Routine, Follow-up, or Biopsy and highlights discordant model/BI-RADS cases for review.

## Reported Evaluation

The manuscript reports strong performance across segmentation, classification, and triage. Highlights include:

- Segmentation Dice score of about 0.852.
- EHR-only classification AUC of about 0.927 on the held-out test set.
- BI-RADS >= 4a baseline AUC of about 0.930.
- Cross-attention fusion remaining competitive in BI-RADS-excluded settings.
- A triage policy designed to avoid missed malignancies.

## Reproducing the Work

Open the notebooks in Jupyter or VS Code and run them in the intended order:

1. Segmentation notebook to prepare lesion masks and image features.
2. EHR notebook to load and process tabular clinical data.
3. Classification notebook to run fusion models, ablations, plots, and triage analysis.

The notebooks expect the dataset and model artifacts referenced inside the notebook cells. Paths were written for the Kaggle environment used during development, so you may need to update them for a local run.

## Notes

- Large datasets, generated figures, and intermediate experimental outputs are intentionally kept out of version control.
- The repository is organized around the paper workflow, so the classification notebook contains the most complete experiment logic and reporting.
