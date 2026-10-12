"""
Retinal disease classifiers: DenseNet121, EfficientNet-B0, and Weighted Ensemble.
Optimized for 5-class DR severity grading and binary Glaucoma screening.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(architecture="densenet121", num_classes=5, pretrained=True, dropout=0.3):
    """
    Constructs fine-tuned transfer learning backbone with custom classifier head.
    Supported architectures: 'densenet121', 'efficientnet_b0'
    """
    arch = architecture.lower().replace("-", "_")

    if "densenet121" in arch:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, num_classes)
        )
        model.architecture_name = "densenet121"

    elif "efficientnet_b0" in arch or "efficientnet" in arch:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, num_classes)
        )
        model.architecture_name = "efficientnet_b0"

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


class RetinalEnsembleModel(nn.Module):
    """
    Hybrid Ensemble combining DenseNet121 and EfficientNet-B0.
    Produces weighted fused probabilities to surpass single-model benchmark accuracies.
    """
    def __init__(self, model_densenet, model_efficientnet, weights=(0.5, 0.5)):
        super().__init__()
        self.model_densenet = model_densenet
        self.model_efficientnet = model_efficientnet
        self.weights = weights

    def forward(self, x):
        logits1 = self.model_densenet(x)
        logits2 = self.model_efficientnet(x)

        prob1 = torch.softmax(logits1, dim=-1)
        prob2 = torch.softmax(logits2, dim=-1)

        fused_prob = self.weights[0] * prob1 + self.weights[1] * prob2
        # Convert back to pseudo-logits for compatibility with standard loss/eval pipelines
        log_prob = torch.log(torch.clamp(fused_prob, min=1e-7))
        return log_prob


class MultiTaskRetinalClassifier(nn.Module):
    """
    Multi-task deep classifier that simultaneously predicts:
    1. 5-class DR severity grading (0: No DR -> 4: Proliferative)
    2. Binary Referable DR screening (0: Non-referable, 1: Referable)
    Sharing feature representations regularizes training and surpasses single-task baselines.
    """
    def __init__(self, architecture="densenet121", pretrained=True, dropout=0.3):
        super().__init__()
        arch = architecture.lower().replace("-", "_")
        self.architecture = arch

        if "densenet121" in arch:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
            self.features = base.features
            num_features = base.classifier.in_features
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif "efficientnet" in arch:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            num_features = base.classifier[1].in_features
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.shared_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
        )

        # Multi-class 5-grade head
        self.grade_head = nn.Sequential(
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, 5)
        )

        # Binary referable DR head
        self.referable_head = nn.Sequential(
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.pooling(feat)
        feat = torch.flatten(feat, 1)
        shared = self.shared_fc(feat)

        grade_logits = self.grade_head(shared)
        referable_logits = self.referable_head(shared)

        return {
            "grade_logits": grade_logits,
            "referable_logits": referable_logits,
        }

