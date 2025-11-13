#!/bin/bash

# Script to run model inference on full Wytham Woods region for spatial mapping
# This script runs inference on the prepared full_region_for_mapping.csv

# Set paths
CHECKPOINT_PATH="saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1016_194432/models/model_best.pth"
PYTHON_PATH="/maps/ys611/miniconda3/envs/mres/bin/python"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please verify the checkpoint path and update this script"
    exit 1
fi

# Check if full region CSV exists
FULL_REGION_CSV="/maps/ys611/MAGIC/data/processed/rtm/wytham/full_region_mapping/full_region_for_mapping.csv"
if [ ! -f "$FULL_REGION_CSV" ]; then
    echo "Error: Full region CSV not found at $FULL_REGION_CSV"
    echo "Please run: /maps/ys611/miniconda3/envs/mres/bin/python datasets/preprocessing/07_wytham_csv_full_region_prepare.py"
    exit 1
fi

echo "=========================================="
echo "Running inference on full Wytham region"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $FULL_REGION_CSV"
echo "Python: $PYTHON_PATH"
echo ""

# Run inference with --full_region flag
$PYTHON_PATH test_phys_rtm_smpl.py \
    --resume "$CHECKPOINT_PATH" \
    --full_region

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
echo "Output saved to:"
echo "  ${CHECKPOINT_PATH%.pth}_testset_analyzer_full_region.csv"
echo ""
echo "Next steps:"
echo "1. Run spatial mapping visualization:"
echo "   $PYTHON_PATH visuals/analysis_rtm_spatial_maps.py"
echo ""

