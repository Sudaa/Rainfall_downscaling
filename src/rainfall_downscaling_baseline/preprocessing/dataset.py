"""PyTorch Dataset of paired (coarse, fine) patches for training the U-Net.

Needs PyTorch: add the CPU build with uv when you reach the U-Net step.
"""

# TODO (U-Net step):
# - class PairedPatches(torch.utils.data.Dataset)
#     __init__(coarse, fine, patch_size, factor): store arrays already transformed
#     __len__: number of days
#     __getitem__(i): pick a random crop location on the coarse grid, cut the
#         matching patch_size*factor crop from the fine grid, return tensors
# - def make_loader(dataset, batch_size, shuffle) -> torch.utils.data.DataLoader
