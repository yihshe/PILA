# MRES Conda Environment Setup

This document provides instructions for recreating the `mres` conda environment on another machine.

## Files Included

1. **`mres_environment.yml`** - Portable environment file (no machine-specific paths)
2. **`setup_mres_environment.sh`** - Automated setup script
3. **`mres_environment_setup.md`** - This setup guide

## Method 1: Using Conda Environment File (Recommended)

### Prerequisites
- Anaconda or Miniconda installed on the target machine
- Internet connection for downloading packages

### Steps
1. Copy the `mres_environment.yml` file to the target machine
2. Create the environment:
   ```bash
   conda env create -f mres_environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate mres
   ```

**Note**: This environment file is portable and contains no machine-specific paths, making it safe to use on any system.

## Method 2: Manual Recreation (Fallback)

If Method 1 fails, you can manually recreate the environment:

1. Create a new environment with Python 3.10:
   ```bash
   conda create -n mres python=3.10
   conda activate mres
   ```

2. Install key packages:
   ```bash
   # Core scientific computing
   conda install numpy pandas scipy scikit-learn matplotlib seaborn
   
   # Geospatial packages
   conda install -c conda-forge geopandas rasterio fiona shapely pyproj
   
   # Machine learning
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Additional packages
   conda install jupyter notebook ipython tqdm
   ```



## Verification

After setting up the environment, verify it works by running:

```bash
conda activate mres
python -c "import pandas, numpy, torch, geopandas; print('Environment setup successful!')"
```

## Environment Details

- **Python Version**: 3.10
- **Key Channels**: anaconda, plotly, pytorch, conda-forge, defaults
- **Main Packages**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, geopandas, rasterio, pytorch, jupyter

## Important: Portable Environment

This environment file is portable and contains no machine-specific paths, making it safe to use on any system. The file has been cleaned to remove:
- Machine-specific paths (like `/maps/ys611/miniconda3/envs/mres`)
- Build strings that may not be available on other platforms

## Troubleshooting

### Common Issues

1. **Channel conflicts**: If you encounter channel conflicts, try:
   ```bash
   conda config --add channels conda-forge
   conda config --add channels pytorch
   ```

2. **CUDA issues**: If PyTorch CUDA doesn't work, install CPU version:
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. **Package conflicts**: If specific packages fail to install, try installing them individually:
   ```bash
   conda install package_name -c conda-forge
   ```

### Platform Differences

- **Linux**: Use the provided files directly
- **Windows**: May need to adjust some package versions
- **macOS**: May need different CUDA packages or CPU-only versions

## Notes

- The environment was created on a Linux system with CUDA support
- Some packages may have platform-specific dependencies
- If you encounter issues, try creating the environment step by step using Method 3

## File Sizes
- `mres_environment.yml`: ~6.8 KB
- `mres_conda_list.txt`: ~6.0 KB  
- `mres_requirements.txt`: ~1.5 KB

Created on: $(date)
Source machine: $(hostname)
