#!/bin/bash

# Script to recreate the mres conda environment
# Usage: ./setup_mres_environment.sh

echo "Setting up MRES conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment file exists
if [ -f "mres_environment.yml" ]; then
    ENV_FILE="mres_environment.yml"
    echo "Using portable environment file"
else
    echo "Error: mres_environment.yml not found in current directory"
    exit 1
fi

echo "Using environment file: $ENV_FILE"

# Remove existing environment if it exists
echo "Removing existing mres environment (if it exists)..."
conda env remove -n mres -y 2>/dev/null || true

# Create new environment
echo "Creating new mres environment from $ENV_FILE..."
conda env create -f $ENV_FILE

# Check if creation was successful
if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate mres"
    echo ""
    echo "To verify the installation, run:"
    echo "  conda activate mres"
    echo "  python -c \"import pandas, numpy, torch, geopandas; print('Environment setup successful!')\""
else
    echo "❌ Environment creation failed!"
    echo ""
    echo "Trying alternative method with conda list file..."
    
    if [ -f "mres_conda_list.txt" ]; then
        conda create --name mres --file mres_conda_list.txt
        if [ $? -eq 0 ]; then
            echo "✅ Environment created successfully using conda list file!"
        else
            echo "❌ Both methods failed. Please check the setup guide: mres_environment_setup.md"
            exit 1
        fi
    else
        echo "❌ mres_conda_list.txt not found. Please check the setup guide: mres_environment_setup.md"
        exit 1
    fi
fi

echo ""
echo "Environment setup complete!"
