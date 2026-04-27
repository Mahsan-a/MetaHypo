# MetaHypo
MetaHypo: Personalized Next-Day Hypoglycemia Prediction via Few-Shot Meta-Learning

This repository contains the implementation of **MetaHypo**, the personalized meta-learning framework presented in Chapter 3 of the thesis *"Population-Level Fairness and Individual-Level Personalization for Hypoglycemia Prediction in Type 1 Diabetes"*.

MetaHypo frames next-day hypoglycemia prediction as a patient-as-task MAML problem. A shared spectral-temporal encoder, pretrained on five large CGM cohorts, is frozen. A lightweight classifier head is meta-trained so that five inner-loop gradient steps on a patient's 120-day support set produce a well-calibrated, personalized predictor. A prevalence-aware decision threshold derived from the same support set replaces the standard fixed threshold of 0.5.

---

## Repository Structure

```
metahypo/
├── src/
│   ├── config.py          # Hyperparameters 
│   ├── models.py          # PretrainingModel and PretrainedModel
│   ├── losses.py          # FocalLoss
│   ├── metahypo.py        # MAML meta-training and per-patient adaptation
│   └── evaluate.py        # Per-patient metrics and threshold calibration
├── train.py               # Full cross-validation training script
├── data/                  # Dataset files available upon request
└── README.md
```

---

The datasets supported by this framework include:

| Dataset | N patients | Ages | Monitoring duration |
|---|---|---|---|
| T1DEXI | 491 | 18–70 | 28 days |
| T1DEXIP | 227 | 12–17 | 28 days |
| CL3 | 168 | ≤14 | 6–8 months |
| CL5 | 100 | 6–13 | 16–20 weeks |
| CITY | 149 | 14–25 | 26 weeks |
| PEDAP | 98 | 2–6 | 26–32 weeks |
| AIDET1D | 82 | ≥65 | 54 weeks |
---

## Reproducing Results

## Quickstart

```bash
# Install dependencies
pip install torch numpy scikit-learn

# Prepare CGM data (computes CWT scalograms from raw glucose CSVs)
python src/data_utils.py --data/cgm_csvs path as data_dir and --data/scalograms as out_dir 

# Run full 5-fold cross-validation
python scripts/train.py
```

Results are written to `outputs/fold_{k}/` for each fold.

---

## Method Summary

Three training phases:

1. **Pre-training** — Spectral-temporal CNN encoder trained on pooled patient-days with focal loss. The `cnn_dense`, `basic_processor`, and classifier head are trained jointly. All are frozen after convergence.

2. **Meta-training** — MAML outer loop over training patient tasks. Inner loop: 5 SGD steps with focal loss + L2 anchor toward pretrained head. Outer loop: query focal loss + 0.2 × support focal loss (anti-memorization). Early stopping on held-out validation tasks.

3. **Personalization** — 5 inner-loop adaptation steps on the patient's first 150 support days. Decision threshold calibrated as the (1 − π̂ⱼ)-quantile of support-set predicted probabilities, where π̂ⱼ is the empirical hypoglycemia prevalence.

