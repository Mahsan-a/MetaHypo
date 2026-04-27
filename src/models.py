"""
models.py
PretrainingModel  — joint pretraining on pooled patient-days.
PretrainedModel   — frozen encoder + trainable classifier head for
                    fine-tuning and MAML meta-learning.
"""

import torch
import torch.nn as nn
from src.config import Config


# ---------------------------------------------------------------------------

class SharedCNNEncoder(nn.Module):
    """Three-block temporal CNN. Accepts (B, 1, 96, 288) CWT spectrograms."""

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32,  kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


# ---------------------------------------------------------------------------

class PretrainingModel(nn.Module):
    """
    Full model used during pre-training.
    All submodule state dicts are extracted and transferred to PretrainedModel
    after convergence.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = SharedCNNEncoder(config)

        self.cnn_dense = nn.Sequential(
            nn.Linear(128 * 9 * 9, 512), nn.ReLU(),
            nn.Linear(512, 256),         nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
        )
        self.basic_processor = nn.Sequential(
            nn.Linear(config.BASIC_FEATURES_DIM, 64), nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
            nn.Linear(64, 16),                         nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
        )
        self.classifier = nn.Sequential(
            nn.Linear(272, 256), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLASSIFIER),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spectrogram: torch.Tensor,
                basic_features: torch.Tensor) -> torch.Tensor:
        enc      = self.cnn_dense(self.encoder(spectrogram))
        basic    = self.basic_processor(basic_features)
        return self.classifier(torch.cat([enc, basic], dim=1))


# ---------------------------------------------------------------------------

class PretrainedModel(nn.Module):
    """
    Frozen encoder + trainable classifier head.
    Used for both the FT baseline and MAML meta-learning.

    freeze_encoder=True locks encoder, cnn_dense, and basic_processor.
    Only self.classifier parameters receive gradients.
    """

    def __init__(self,
                 config: Config,
                 encoder_state_dict,
                 cnn_dense_state_dict,
                 basic_processor_state_dict,
                 classifier_state_dict,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = SharedCNNEncoder(config)
        self.encoder.load_state_dict(encoder_state_dict)

        self.cnn_dense = nn.Sequential(
            nn.Linear(128 * 9 * 9, 512), nn.ReLU(),
            nn.Linear(512, 256),         nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
        )
        self.cnn_dense.load_state_dict(cnn_dense_state_dict)

        self.basic_processor = nn.Sequential(
            nn.Linear(config.BASIC_FEATURES_DIM, 64), nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
            nn.Linear(64, 16),                         nn.ReLU(),
            nn.Dropout(config.DROPOUT_ENCODER),
        )
        self.basic_processor.load_state_dict(basic_processor_state_dict)

        self.classifier = nn.Sequential(
            nn.Linear(272, 256), nn.ReLU(),
            nn.Dropout(config.DROPOUT_CLASSIFIER),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )
        if classifier_state_dict is not None:
            self.classifier.load_state_dict(classifier_state_dict)

        if freeze_encoder:
            for m in [self.encoder, self.cnn_dense, self.basic_processor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, spectrogram: torch.Tensor,
                basic_features: torch.Tensor) -> torch.Tensor:
        enc   = self.cnn_dense(self.encoder(spectrogram))
        basic = self.basic_processor(basic_features)
        return self.classifier(torch.cat([enc, basic], dim=1))
