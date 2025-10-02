import torch

from dataclasses import dataclass, replace

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor
