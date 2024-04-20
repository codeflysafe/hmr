import torch
import torch.nn as nn

from backbone import build_backbone
from head import build_head

class HMR(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

    def forward(self, images):
        '''
        images: shape, [b, c, h, w]
        '''
        image_features = self.backbone(images)
        out = self.head(image_features)
        return out
    
