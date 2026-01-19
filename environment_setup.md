# PILA Conda Environment Setup

This document provides instructions for recreating the `pila` conda environment.

## Files Included

1. **`environment.yml`** - Portable environment file (no machine-specific paths)
2. **`setup_environment.sh`** - Automated setup script
3. **`environment_setup.md`** - This setup guide

## Method 1: Automated (Recommended)

```bash
./setup_environment.sh
```

## Method 2: Manual

```bash
conda env create -f environment.yml
conda activate pila
```

## Verification

```bash
conda activate pila
python -c "import pandas, numpy, torch, geopandas; print('Environment setup successful!')"
```

## Notes

- The environment file is portable and contains no machine-specific paths.
- If you need a CPU-only install, replace CUDA packages with `cpuonly` in `environment.yml`.
