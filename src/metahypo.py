"""
metahypo.py
MAML meta-training and per-patient adaptation for MetaHypo.

Phase 2: meta_train()
    Outer loop — FOMAML gradient accumulation across patient tasks.
    Inner loop — 5 SGD steps with focal loss + L2 anchor toward ψ₀.

Phase 3: personalize_and_predict()
    Per-patient adaptation from ψ* followed by prevalence-aware
    threshold calibration and query-set prediction.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any

from src.config import Config
from src.losses import FocalLoss
from src.evaluate import calibrate_threshold, predict_query, compute_patient_metrics


# ---------------------------------------------------------------------------
# Task pool construction
# ---------------------------------------------------------------------------

def _get_label(sample) -> int:
    return extract_hypoglycemia_label(sample[2]['next_day_cgm_labels_original'])


def build_task_pool(samples_by_pid: Dict[str, List],
                    config: Config) -> List[Tuple]:
    """
    For each patient with >= SUPPORT_DAYS total days and >= MIN_POS_SUPPORT
    positive labels in the first SUPPORT_DAYS days, create:
        (support_dataset, query_dataset, pos_indices, neg_indices)
    """
    pool, skipped = [], 0

    for pid, days in samples_by_pid.items():
        if len(days) <= config.SUPPORT_DAYS:
            skipped += 1
            continue

        sup = days[:config.SUPPORT_DAYS]
        qry = days[config.SUPPORT_DAYS:]

        pos = [i for i, s in enumerate(sup) if _get_label(s) == 1]
        neg = [i for i, s in enumerate(sup) if _get_label(s) == 0]

        if len(pos) < config.MIN_POS_SUPPORT:
            skipped += 1
            continue

        pool.append((BuildDataset(sup, config), BuildDataset(qry, config), pos, neg))

    print(f"  build_task_pool: {len(pool)} eligible, {skipped} skipped.")
    return pool


def group_by_pid(samples: List) -> Dict[str, List]:
    out = {}
    for s in samples:
        out.setdefault(s[0], []).append(s)
    return out


# ---------------------------------------------------------------------------
# Positive-focused mini-batch
# ---------------------------------------------------------------------------

def _positive_batch(dataset, pos_idx: List[int], neg_idx: List[int],
                    m: int, k_pos: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(k_pos, m, len(pos_idx)) if pos_idx else 0

    chosen_pos = np.random.choice(pos_idx, size=k,   replace=(len(pos_idx) < k))   if k > 0      else np.array([], dtype=int)
    chosen_neg = np.random.choice(neg_idx, size=m-k, replace=(len(neg_idx) < m-k)) if m - k > 0  else np.array([], dtype=int)

    idx   = np.random.permutation(np.concatenate([chosen_pos, chosen_neg]).astype(int))
    items = [dataset[int(i)] for i in idx]

    specs  = torch.stack([it['spectrogram']    for it in items]).to(device)
    basic  = torch.stack([it['basic_features'] for it in items]).to(device)
    labels = torch.tensor([it['labels'].item() for it in items], dtype=torch.float32).to(device)
    return specs, basic, labels


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------

def _inner_loop(fast_model: nn.Module,
                sup_dataset,
                pos_idx: List[int],
                neg_idx: List[int],
                focal: FocalLoss,
                anchor_params: Dict[str, torch.Tensor],
                config: Config,
                device: torch.device):
    """
    K SGD steps on the support set.
    Loss = FocalLoss + λ · ‖ψ − ψ₀‖²
    Only classifier parameters are updated; the encoder is frozen.
    """
    opt = torch.optim.SGD(fast_model.classifier.parameters(), lr=config.INNER_LR)
    fast_model.train()

    for _ in range(config.INNER_STEPS):
        specs, basic, labels = _positive_batch(
            sup_dataset, pos_idx, neg_idx, config.MINIBATCH_SIZE, config.K_POSITIVE, device
        )
        opt.zero_grad()
        loss = focal(fast_model(specs, basic), labels)

        anchor = sum(
            ((p - anchor_params[n].to(device)) ** 2).sum()
            for n, p in fast_model.classifier.named_parameters()
        )
        (loss + config.ANCHOR_WEIGHT * anchor).backward()
        opt.step()


# ---------------------------------------------------------------------------
# Dataset loss (outer-loop evaluation)
# ---------------------------------------------------------------------------

def _dataset_loss(model: nn.Module, dataset, focal: FocalLoss,
                  batch_size: int, device: torch.device) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, n = torch.tensor(0.0, device=device), 0
    model.train()
    for batch in loader:
        specs  = batch['spectrogram'].to(device)
        basic  = batch['basic_features'].to(device)
        labels = batch['labels'].to(device).float()
        total  = total + focal(model(specs, basic), labels) * len(labels)
        n     += len(labels)
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# One meta-training epoch
# ---------------------------------------------------------------------------

def _meta_epoch(meta_model: nn.Module,
                task_pool: List[Tuple],
                focal: FocalLoss,
                meta_opt: torch.optim.Optimizer,
                anchor_params: Dict[str, torch.Tensor],
                config: Config,
                device: torch.device) -> float:

    n_tasks  = len(task_pool)
    t_idx    = np.random.choice(n_tasks, size=config.TASK_BATCH,
                                replace=(n_tasks < config.TASK_BATCH))
    meta_opt.zero_grad()
    epoch_loss = 0.0

    for i in t_idx:
        sup_ds, qry_ds, pos_idx, neg_idx = task_pool[int(i)]

        fast = copy.deepcopy(meta_model).to(device)
        _inner_loop(fast, sup_ds, pos_idx, neg_idx, focal, anchor_params, config, device)

        for p in fast.classifier.parameters():
            p.grad = None

        qry_loss = _dataset_loss(fast, qry_ds, focal, config.MINIBATCH_SIZE, device)
        sup_loss = _dataset_loss(fast, sup_ds, focal, config.MINIBATCH_SIZE, device)
        task_loss = qry_loss + config.ANTI_MEMO_WEIGHT * sup_loss
        task_loss.backward()

        for p_m, p_f in zip(meta_model.classifier.parameters(),
                             fast.classifier.parameters()):
            if p_f.grad is not None:
                if p_m.grad is None:
                    p_m.grad = p_f.grad.clone()
                else:
                    p_m.grad.add_(p_f.grad)

        epoch_loss += task_loss.item()
        del fast

    for p in meta_model.classifier.parameters():
        if p.grad is not None:
            p.grad.div_(config.TASK_BATCH)

    torch.nn.utils.clip_grad_norm_(meta_model.classifier.parameters(), config.GRAD_CLIP)
    meta_opt.step()
    return epoch_loss / config.TASK_BATCH


# ---------------------------------------------------------------------------
# Validation task-pool loss (early stopping)
# ---------------------------------------------------------------------------

def _val_loss(meta_model: nn.Module, task_pool: List[Tuple],
              focal: FocalLoss, anchor_params: Dict[str, torch.Tensor],
              config: Config, device: torch.device) -> float:
    if not task_pool:
        return float('inf')

    idx = np.random.choice(len(task_pool),
                           size=min(len(task_pool), config.TASK_BATCH),
                           replace=False)
    total = 0.0
    for i in idx:
        sup_ds, qry_ds, pos_idx, neg_idx = task_pool[int(i)]
        fast = copy.deepcopy(meta_model).to(device)
        _inner_loop(fast, sup_ds, pos_idx, neg_idx, focal, anchor_params, config, device)
        fast.eval()
        with torch.no_grad():
            q = _dataset_loss(fast, qry_ds, focal, config.MINIBATCH_SIZE, device)
            s = _dataset_loss(fast, sup_ds, focal, config.MINIBATCH_SIZE, device)
        total += (q + config.ANTI_MEMO_WEIGHT * s).item()
        del fast
    return total / len(idx)


# ---------------------------------------------------------------------------
# Public API — meta_train
# ---------------------------------------------------------------------------

def meta_train(meta_model: nn.Module,
               train_tasks: List[Tuple],
               val_tasks:   List[Tuple],
               anchor_params: Dict[str, torch.Tensor],
               config: Config,
               device: torch.device) -> nn.Module:
    """
    Run MAML meta-training with early stopping on validation task loss.
    Returns the meta-model restored to the best checkpoint.
    """
    focal     = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)
    meta_opt  = torch.optim.Adam(meta_model.classifier.parameters(),
                                  lr=config.META_LR, weight_decay=config.META_WD)
    best_val  = float('inf')
    best_ckpt = copy.deepcopy(meta_model.classifier.state_dict())
    patience  = 0

    for epoch in range(1, config.META_EPOCHS + 1):
        tr = _meta_epoch(meta_model, train_tasks, focal, meta_opt,
                         anchor_params, config, device)
        vl = _val_loss(meta_model, val_tasks, focal, anchor_params, config, device)

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{config.META_EPOCHS}  "
                  f"train {tr:.4f}  val {vl:.4f}")

        if vl < best_val:
            best_val  = vl
            best_ckpt = copy.deepcopy(meta_model.classifier.state_dict())
            patience  = 0
        else:
            patience += 1
            if patience >= config.META_PATIENCE:
                print(f"  early stop at epoch {epoch}  best val {best_val:.4f}")
                break

    meta_model.classifier.load_state_dict(best_ckpt)
    return meta_model


# ---------------------------------------------------------------------------
# Public API — personalize_and_predict
# ---------------------------------------------------------------------------

def personalize_and_predict(meta_model: nn.Module,
                             support_samples: List,
                             query_samples:   List,
                             anchor_params: Dict[str, torch.Tensor],
                             config: Config,
                             device: torch.device) -> Dict[str, Any]:
    """
    1. Clone meta-model as ψ*.
    2. Run INNER_STEPS adaptation on patient support set.
    3. Calibrate decision threshold from support-set prevalence.
    4. Predict on query set and return per-patient metrics.
    """
    sup_ds = BuildDataset(support_samples, config)
    qry_ds = BuildDataset(query_samples,   config)

    pos_idx = [i for i, s in enumerate(support_samples) if _get_label(s) == 1]
    neg_idx = [i for i, s in enumerate(support_samples) if _get_label(s) == 0]

    focal         = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)
    patient_model = copy.deepcopy(meta_model).to(device)

    _inner_loop(patient_model, sup_ds, pos_idx, neg_idx,
                focal, anchor_params, config, device)

    tau   = calibrate_threshold(patient_model, sup_ds, config.BATCH_SIZE,
                                device, config.THRESHOLD_CLIP_LOW,
                                config.THRESHOLD_CLIP_HIGH)
    probs, labels = predict_query(patient_model, qry_ds, config.BATCH_SIZE, device)
    metrics = compute_patient_metrics(probs, labels, tau)
    return {'probs': probs, 'labels': labels, **metrics}
