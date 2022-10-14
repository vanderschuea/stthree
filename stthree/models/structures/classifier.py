import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, layer_classes, backbone, n_classes, apply_to, dropout=None):
        super().__init__()
        self.backbone = backbone
        self.fc = layer_classes.fc(backbone.output_size, n_classes, bias=False)
        self.dropout = dropout
        self.apply_to = apply_to

    def forward(self, nx):
        nx = self.backbone(nx[self.apply_to])
        nx = F.adaptive_avg_pool2d(nx, (1, 1)).squeeze_(-1).squeeze_(-1)
        if self.dropout is not None:
            nx = F.dropout(nx, p=self.dropout)
        nx = self.fc(nx)
        return {
            "classification": nx
        }

