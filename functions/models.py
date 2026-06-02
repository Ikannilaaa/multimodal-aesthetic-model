# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualOnlyNetwork(nn.Module):
    def __init__(self, visual_dim):
        super(VisualOnlyNetwork, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, visual):
        if visual.dim() > 2:
            visual = visual.view(visual.size(0), -1)

        return self.fusion(visual)

class TextOnlyNetwork(nn.Module):
    def __init__(self, text_dim):
        super(TextOnlyNetwork, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, text):
        if text.dim() > 2:
            text = text.view(text.size(0), -1)

        return self.fusion(text)

class AestheticFusionNetwork(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super(AestheticFusionNetwork, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, visual, text):
        if visual.dim() > 2:
            visual = visual.view(visual.size(0), -1)
        if text.dim() > 2:
            text = text.view(text.size(0), -1)

        combined = torch.cat((visual, text), dim=1)
        return self.fusion(combined)