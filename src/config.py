"""
config.py
All hyperparameters and dataset constants for MetaHypo.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── architecture ──────────────────────────────────────────────────────
    CNN_FILTERS: List[int]      = field(default_factory=lambda: [32, 64, 128])
    SPECTROGRAM_SHAPE: tuple    = (96, 288)
    BASIC_FEATURES_DIM: int     = 138
    CNN_DENSE_DIMS: List[int]   = field(default_factory=lambda: [512, 256])
    BASIC_PROC_DIMS: List[int]  = field(default_factory=lambda: [64, 16])
    CLASSIFIER_DIMS: List[int]  = field(default_factory=lambda: [272, 256, 64, 1])
    DROPOUT_ENCODER: float      = 0.5
    DROPOUT_CLASSIFIER: float   = 0.7

    # ── pre-training ──────────────────────────────────────────────────────
    PRETRAIN_LR: float          = 1e-4
    PRETRAIN_WD: float          = 1e-4
    PRETRAIN_BATCH: int         = 64
    PRETRAIN_EPOCHS: int        = 100
    PRETRAIN_PATIENCE: int      = 20
    FOCAL_ALPHA: float          = 0.75
    FOCAL_GAMMA: float          = 2.0

    # ── fine-tuning baseline ──────────────────────────────────────────────
    FINETUNE_LR: float          = 1e-5
    FINETUNE_WD: float          = 1e-4
    FINETUNE_EPOCHS: int        = 50
    FINETUNE_ANCHOR: float      = 0.05

    # ── meta-learning ─────────────────────────────────────────────────────
    META_LR: float              = 0.001
    META_WD: float              = 0.0
    TASK_BATCH: int             = 16
    META_EPOCHS: int            = 200
    META_PATIENCE: int          = 15
    ANTI_MEMO_WEIGHT: float     = 0.2

    INNER_LR: float             = 0.005
    INNER_STEPS: int            = 5
    ANCHOR_WEIGHT: float        = 0.05
    MINIBATCH_SIZE: int         = 16
    K_POSITIVE: int             = 4

    # ── task eligibility ──────────────────────────────────────────────────
    SUPPORT_DAYS: int           = 150
    MIN_POS_SUPPORT: int        = 5

    # ── threshold calibration ─────────────────────────────────────────────
    THRESHOLD_CLIP_LOW: float   = 0.05
    THRESHOLD_CLIP_HIGH: float  = 0.95

    # ── cross-validation ──────────────────────────────────────────────────
    N_FOLDS: int                = 5
    VAL_FOLDS: int              = 5
    RANDOM_SEED: int            = 42

    # ── misc ──────────────────────────────────────────────────────────────
    BATCH_SIZE: int             = 64
    GRAD_CLIP: float            = 1.0


config = Config()
