"""
evaluate.py
Per-patient threshold calibration and metric computation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict


def calibrate_threshold(model, support_dataset, batch_size: int,
                        device: torch.device,
                        clip_low: float = 0.05,
                        clip_high: float = 0.95) -> float:
    """
    Set τ_j as the (1 − π̂_j)-quantile of support-set predicted probabilities.
    π̂_j is the empirical hypoglycemia prevalence over the support days.
    No future labels are used — π̂_j is computed from the same support set
    used for adaptation, so there is no data leakage.
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(support_dataset, batch_size=batch_size, shuffle=False):
            p = torch.sigmoid(
                model(batch['spectrogram'].to(device),
                      batch['basic_features'].to(device))
            ).squeeze(-1).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch['labels'].cpu().numpy().tolist())

    probs  = np.array(probs)
    labels = np.array(labels)
    pi_hat = labels.mean()

    if pi_hat <= 0.0 or pi_hat >= 1.0:
        return 0.5

    return float(np.clip(np.quantile(probs, 1.0 - pi_hat), clip_low, clip_high))


def compute_patient_metrics(probs: np.ndarray, labels: np.ndarray,
                             threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy    = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float('nan')

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy':    accuracy,
        'auc':         auc,
        'n_pos':       int(labels.sum()),
        'n_total':     len(labels),
        'threshold':   threshold,
    }


def predict_query(model, query_dataset, batch_size: int,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(query_dataset, batch_size=batch_size, shuffle=False):
            p = torch.sigmoid(
                model(batch['spectrogram'].to(device),
                      batch['basic_features'].to(device))
            ).squeeze(-1).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch['labels'].cpu().numpy().tolist())
    return np.array(probs), np.array(labels)
