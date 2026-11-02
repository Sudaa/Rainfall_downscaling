"""Small U-Net that predicts a correction (residual) on top of bilinear interpolation.

This mirrors CorrDiff's first step, a regression network that predicts the
mean. Training runs on a free GPU (Colab); this file only needs CPU PyTorch locally.
"""

# TODO (U-Net step):
# - class UNet(torch.nn.Module): 2-3 down/up levels, skip connections.
#   Input channels: interpolated precipitation, water vapour, u850, v850.
#   Output: 1 channel, the residual in log(1 + rain) space.
# - class UNetDownscaler: follows models.base.Downscaler
#     fit: build PairedPatches, train with MSE on the residual, save a checkpoint
#     predict: bilinear + predicted residual, then undo the transforms to mm/day
