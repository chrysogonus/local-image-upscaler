"""Real-ESRGAN RRDBNet generator.

Vendored deliberately. The upstream ``realesrgan`` package pulls in ``basicsr``,
which is unmaintained and fails to import against current torchvision releases.
The generator is small and stable, so carrying it here keeps the CUDA engine's
dependency footprint to torch alone.

Architecture and the official weights are BSD-3-Clause (Xintao Wang et al.,
https://github.com/xinntao/Real-ESRGAN). Layer names match the published
checkpoints exactly; renaming anything here breaks weight loading.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

RESIDUAL_SCALE = 0.2


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * RESIDUAL_SCALE + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * RESIDUAL_SCALE + x


class RRDBNet(nn.Module):
    """The x4 generator shared by RealESRGAN_x4plus and its 6-block anime variant."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB(num_feat, num_grow_ch) for _ in range(num_block)))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_generator(weights_path, device, dtype) -> RRDBNet:
    """Build the generator implied by a checkpoint and load it in eval mode.

    ``num_block`` is inferred from the checkpoint rather than hardcoded so the
    23-block photo model and the 6-block anime model share this path.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    else:
        state = checkpoint

    block_indices = {
        int(name.split(".")[1])
        for name in state
        if name.startswith("body.") and name.count(".") > 1
    }
    if not block_indices:
        raise ValueError(f"{weights_path} does not look like an RRDBNet checkpoint")

    model = RRDBNet(num_block=max(block_indices) + 1)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device=device, dtype=dtype)
