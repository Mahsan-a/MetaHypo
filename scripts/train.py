"""
train.py
Full 5-fold stratified cross-validation for MetaHypo.

Expects:
    data/all_samples.pkl   — list of (pid, date, features_dict) tuples

Writes per-fold results to:
    outputs/fold_{k}/metrics.pkl
"""

import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.config import config
from src.models import PretrainedModel
from src.metahypo import (build_task_pool, group_by_pid,
                           meta_train, personalize_and_predict)

os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open('data/all_samples.pkl', 'rb') as f:
    all_samples = pickle.load(f)

participant_hypo_freq = compute_participant_hypo_frequency(all_samples)
hypo_participants, participant_labels = compute_participant_labels(all_samples)
participant_ids = np.array(hypo_participants)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

sgkf         = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                                random_state=config.RANDOM_SEED)
fold_results = []

for fold_idx, (train_val_idx, test_idx) in enumerate(
        sgkf.split(participant_ids, participant_labels)):

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")

    train_val_ids    = participant_ids[train_val_idx]
    test_ids         = participant_ids[test_idx]
    train_val_labels = participant_labels[train_val_idx]

    inner_cv = StratifiedKFold(n_splits=config.VAL_FOLDS, shuffle=True,
                                random_state=config.RANDOM_SEED)
    train_sub, val_sub = next(inner_cv.split(train_val_ids, train_val_labels))
    train_ids = train_val_ids[train_sub]
    val_ids   = train_val_ids[val_sub]

    train_samples, val_samples, test_samples = split_samples_by_participants(
        all_samples, train_ids, val_ids, test_ids
    )

    # ── Phase 1: pre-training ────────────────────────────────────────────
    print("\nPhase 1 — pre-training")
    encoder_sd, cnn_dense_sd, basic_proc_sd, classifier_sd = pretrain_encoder_once(
        config, device, train_samples, val_samples, fold_idx=fold_idx + 1
    )

    anchor_params = {n: v.clone().detach()
                     for n, v in classifier_sd.items()}

    # ── build meta-model from pretrained initialization ──────────────────
    meta_model = PretrainedModel(
        config               = config,
        encoder_state_dict   = encoder_sd,
        cnn_dense_state_dict = cnn_dense_sd,
        basic_processor_state_dict = basic_proc_sd,
        classifier_state_dict      = classifier_sd,
        freeze_encoder       = True,
    ).to(device)

    # ── Phase 2: meta-training ────────────────────────────────────────────
    print("\nPhase 2 — meta-training")
    train_tasks = build_task_pool(group_by_pid(train_samples), config)
    val_tasks   = build_task_pool(group_by_pid(val_samples),   config)

    meta_model = meta_train(meta_model, train_tasks, val_tasks,
                             anchor_params, config, device)

    # ── Phase 3: per-patient personalization ─────────────────────────────
    print("\nPhase 3 — per-patient adaptation")

    test_by_pid = group_by_pid(test_samples)
    fold_metrics = {}

    for pid, pid_days in test_by_pid.items():
        if len(pid_days) <= config.SUPPORT_DAYS:
            continue

        sup = pid_days[:config.SUPPORT_DAYS]
        qry = pid_days[config.SUPPORT_DAYS:]

        pos_in_sup = sum(1 for s in sup
                         if extract_hypoglycemia_label(
                             s[2]['next_day_cgm_labels_original']) == 1)
        if pos_in_sup < config.MIN_POS_SUPPORT:
            continue

        result = personalize_and_predict(
            meta_model, sup, qry, anchor_params, config, device
        )
        fold_metrics[pid] = result

    # ── summarize fold ────────────────────────────────────────────────────
    sens = [v['sensitivity'] for v in fold_metrics.values()]
    spec = [v['specificity'] for v in fold_metrics.values()]
    auc  = [v['auc']         for v in fold_metrics.values() if not np.isnan(v['auc'])]

    print(f"\n  Fold {fold_idx + 1} summary  ({len(fold_metrics)} patients)")
    print(f"  Sensitivity  {np.mean(sens):.3f} ± {np.std(sens):.3f}")
    print(f"  Specificity  {np.mean(spec):.3f} ± {np.std(spec):.3f}")
    print(f"  AUC          {np.mean(auc):.3f} ± {np.std(auc):.3f}")

    out_dir = f'outputs/fold_{fold_idx + 1}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(fold_metrics, f)

    fold_results.append({'fold': fold_idx + 1, 'metrics': fold_metrics})

# ---------------------------------------------------------------------------
# Aggregate across folds
# ---------------------------------------------------------------------------

all_sens = [v['sensitivity'] for r in fold_results for v in r['metrics'].values()]
all_spec = [v['specificity'] for r in fold_results for v in r['metrics'].values()]
all_auc  = [v['auc']         for r in fold_results for v in r['metrics'].values()
            if not np.isnan(v['auc'])]

print(f"\n{'='*60}")
print("5-fold aggregate")
print(f"  N patients   {len(all_sens)}")
print(f"  Sensitivity  {np.mean(all_sens):.3f} ± {np.std(all_sens):.3f}")
print(f"  Specificity  {np.mean(all_spec):.3f} ± {np.std(all_spec):.3f}")
print(f"  AUC          {np.mean(all_auc):.3f} ± {np.std(all_auc):.3f}")

with open('outputs/all_fold_results.pkl', 'wb') as f:
    pickle.dump(fold_results, f)
