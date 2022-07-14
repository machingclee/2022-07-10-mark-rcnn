import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from . import config

class MLPDetector(nn.Module):
    def __init__(self):
        super(MLPDetector, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, config.n_classes)
        self.bbox_pred = nn.Linear(4096, config.n_classes * 4)

    def forward(self, roi_pooling):
        x = roi_pooling.reshape((-1, 512 * 7 * 7))  # [128, 512*7*7]
        x = self.mlp_head(x)                        # [128, 4096]
        scores_logits = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores_logits, deltas
            


class MaskDetector(nn.Module):
    def __init__(self):
        super(MaskDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ConvTranspose2d(512, 512, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, config.n_classes, 1, 1, 0)
        )
        

    def forward(self, mask_pooling):
        return self.net(mask_pooling)