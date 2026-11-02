"""Baseline rainfall downscaling: ERA5 (0.25 deg) to CHIRPS (0.05 deg) over Thailand.

Subpackages, in pipeline order:

- data:          download, load, crop and align the raw datasets
- preprocessing: transforms, train/test split, paired samples for training
- models:        bilinear, quantile mapping and U-Net downscalers
- evaluation:    metrics, scoring of all models, figures
"""
