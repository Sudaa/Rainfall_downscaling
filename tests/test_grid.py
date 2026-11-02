"""Tests to write for data/grid.py and data/era5.py (Day 1):

- crop keeps only points inside the region, for both latitude orders
- check_alignment passes on a synthetic coarse/fine pair and fails when shifted
- to_daily: 24 hourly values of 0.001 m give 24 mm for that day
- loaded data has no unexpected NaNs over land
"""
