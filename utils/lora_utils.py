import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeScaling(nn.Module):
    """Wraps a PEFT LoRA-injected linear layer with learned input/output scalings.

    The base weight is applied with a per-feature input scaling (`scale_in`) and output scaling
    (`scale_out`); the LoRA delta is added unscaled. Only the scalings (and the LoRA params) are
    trained — the frozen base weight is reused.
    """

    def __init__(self, linear_layer, init_scale=0.01):
        super().__init__()
        self.linear = linear_layer
        self.scale_in = nn.Parameter(
            torch.ones(linear_layer.in_features)
            + torch.randn(linear_layer.in_features) * init_scale
        )
        self.scale_out = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, input):
        W = self.linear.base_layer.weight
        b = self.linear.base_layer.bias

        # Base weight with scaling — bias excluded from scaling.
        base_out = self.scale_out * F.linear(input * self.scale_in, W, None)
        if b is not None:
            base_out = base_out + b

        # Isolated LoRA delta (unscaled).
        base_only = F.linear(input, W, b)
        lora_delta = self.linear(input) - base_only

        return base_out + lora_delta
