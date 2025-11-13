import pandas as pd
import numpy as np
import os

"""
Prepare full Wytham Woods region CSV for spatial mapping
Uses the same scaler as insitu_period training to ensure comparable predictions
"""

S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
            'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
            'B12_SWI2']

BASE_DIR = "/maps/ys611/MAGIC/data/raw/wytham"
PROCESSED_DIR = "/maps/ys611/MAGIC/data/processed/rtm/wytham"

# Input: Full aggregated CSV (all pixels in Wytham Woods)
CSV_INPUT = os.path.join(BASE_DIR, "csv_preprocessed_data", "rasters_sentinel2_2018.csv")

# Load full region data
print("Loading full region data...")
df_full = pd.read_csv(CSV_INPUT)
print(f"Initial data shape: {df_full.shape}")

# Filter to dates of interest (in-situ measurement period)
dates = ['2018.06.26', '2018.06.29', '2018.07.06', '2018.07.11']
df_full = df_full[df_full["date"].isin(dates)]
print(f"After date filtering: {df_full.shape}")
print(f"Dates: {df_full['date'].unique()}")

# Load the scaler from insitu_period training
# IMPORTANT: Use the SAME scaler to ensure predictions are comparable
SCALER_DIR = os.path.join(PROCESSED_DIR, "insitu_period")
MEAN = np.load(os.path.join(SCALER_DIR, 'train_x_mean.npy'))
SCALE = np.load(os.path.join(SCALER_DIR, 'train_x_scale.npy'))

print(f"\nUsing scaler from: {SCALER_DIR}")
print(f"MEAN shape: {MEAN.shape}")
print(f"SCALE shape: {SCALE.shape}")

# Standardize using SAME scaler as training
print("\nStandardizing spectral bands...")
factor = 10000.0
df_full[S2_BANDS] = df_full[S2_BANDS] / factor
df_full[S2_BANDS] = (df_full[S2_BANDS] - MEAN) / SCALE

# Verify standardization
print("\nStandardized data statistics:")
print(df_full[S2_BANDS].describe())

# Save for inference
SAVE_DIR = os.path.join(PROCESSED_DIR, "full_region_mapping")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

SAVE_PATH = os.path.join(SAVE_DIR, "full_region_for_mapping.csv")
df_full.to_csv(SAVE_PATH, index=False)

print(f"\n✅ Saved full region CSV to: {SAVE_PATH}")
print(f"Total pixels: {len(df_full):,}")
print(f"\nPixels per date:")
print(df_full.groupby('date').size())
print(f"\nColumns: {df_full.columns.tolist()}")
print(f"\nSample of first few rows:")
print(df_full.head())

print("\n" + "="*80)
print("NEXT STEPS:")
print("1. Run model inference on this CSV")
print("2. Use analysis_rtm_spatial_maps.py to create spatial visualizations")
print("="*80)

