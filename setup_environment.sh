#!/bin/bash

# Script to recreate the PILA conda environment
# Usage: ./setup_environment.sh

echo "Setting up PILA conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment file exists
if [ -f "environment.yml" ]; then
    ENV_FILE="environment.yml"
    echo "Using portable environment file"
else
    echo "Error: environment.yml not found in current directory"
    exit 1
fi

echo "Using environment file: $ENV_FILE"

# Remove existing environment if it exists
echo "Removing existing pila environment (if it exists)..."
conda env remove -n pila -y 2>/dev/null || true

# Create new environment
echo "Creating new pila environment from $ENV_FILE..."
conda env create -f $ENV_FILE

# Check if creation was successful
if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate pila"
    echo ""
    echo "To verify the installation, run:"
    echo "  conda activate pila"
    echo "  python -c \"import pandas, numpy, torch, geopandas; print('Environment setup successful!')\""
else
    echo "❌ Environment creation failed!"
    echo "Please check the setup guide: environment_setup.md"
    exit 1
fi

echo ""
echo "Environment setup complete!"
