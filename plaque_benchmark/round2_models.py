"""3D architectures with the benchmark's logits dictionary interface."""
from torch import nn


class Logits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return {"logits": self.model(image)}


def build(model, classes, params):
    if model == "swin_unetr_v2":
        from monai.networks.nets import SwinUNETR
        return Logits(SwinUNETR(in_channels=1, out_channels=classes, use_v2=True, **params))
    if model == "mednext":
        from .mednext_v1.create_mednext_v1 import create_mednext_v1
        return Logits(create_mednext_v1(1, classes, params["size"], kernel_size=params["kernel_size"],
                                        deep_supervision=False))
    raise ValueError(f"Unknown 3D architecture: {model}")
