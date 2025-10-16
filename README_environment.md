# MRES Environment Setup

## Quick Start

To recreate the `mres` conda environment on another machine:

### Option 1: Automated (Recommended)
```bash
./setup_mres_environment.sh
```

### Option 2: Manual
```bash
conda env create -f mres_environment.yml
conda activate mres
```

## Essential Files

- **`mres_environment.yml`** - Main environment file (portable, no machine-specific paths)
- **`setup_mres_environment.sh`** - Automated setup script
- **`mres_environment_setup.md`** - Detailed documentation

## Important Notes

- The environment file is portable and contains no machine-specific paths
- Safe to use on any system (Linux, Windows, macOS)
- See `mres_environment_setup.md` for detailed instructions and troubleshooting
