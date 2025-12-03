# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv
from pathlib import Path

# %%
# Set the log file path
# Example: '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_202135/log/info.log'
BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_202135'
# BASE_PATH = '/maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1016_181644_klp0_edge1'
LOG_PATH = os.path.join(BASE_PATH, 'log/info.log')

# Derive the models directory from the log path
# Log path structure: saved/{model_type}/{model_name}/{timestamp}/log/info.log
# Models directory: saved/{model_type}/{model_name}/{timestamp}/models/
log_path_obj = Path(LOG_PATH).resolve()
SAVE_PATH = Path(BASE_PATH) / 'models' / 'plots'

# Create plots directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def parse_log_file(log_path):
    """
    Parse the training log file and extract per-epoch metrics.
    
    Args:
        log_path: Path to the info.log file
        
    Returns:
        List of dictionaries, each containing metrics for one epoch
    """
    epochs_data = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for the epoch summary section
        # Pattern: "Validation Epoch: X Rec Loss: Y KL Loss: Z"
        # We use this line only as a marker - all metrics are in the detailed entries below
        val_match = re.search(r'Validation Epoch:\s+(\d+)', line)
        
        if val_match:
            # Initialize empty epoch_data - we'll populate it from the detailed entries
            epoch_data = {}
            
            # Parse the following lines for detailed metrics
            i += 1
            while i < len(lines):
                metric_line = lines[i].strip()
                
                # Check if we've reached the next epoch or end of summary
                if 'Train Epoch:' in metric_line or 'Validation Epoch:' in metric_line:
                    break
                
                # Parse metric lines like "     loss           : 24.17151870727539"
                # The line should have the pattern: "metric_name : value" after the log prefix
                # Look for lines that contain " - trainer - INFO - " followed by metric pattern
                if ' - trainer - INFO - ' in metric_line:
                    # Extract the part after the log prefix
                    log_prefix = ' - trainer - INFO - '
                    if log_prefix in metric_line:
                        metric_part = metric_line.split(log_prefix, 1)[1]
                        # Match pattern: "metric_name : value" where metric_name is alphanumeric with underscores
                        metric_match = re.search(r'(\w+)\s*:\s*([\d.]+)', metric_part)
                        if metric_match:
                            metric_name = metric_match.group(1).strip()
                            metric_value = float(metric_match.group(2))
                            # Keep epoch as integer, others as float
                            if metric_name == 'epoch':
                                epoch_data[metric_name] = int(metric_value)
                            else:
                                epoch_data[metric_name] = metric_value
                
                i += 1
            
            # Only append if we found at least the epoch number
            if 'epoch' in epoch_data:
                epochs_data.append(epoch_data)
        else:
            i += 1
    
    return epochs_data

# %%
# Parse the log file and extract epoch data
print(f"Parsing log file: {LOG_PATH}")
epochs_data = parse_log_file(LOG_PATH)

if not epochs_data:
    print("Error: No epoch data found in the log file!")
else:
    print(f"Found {len(epochs_data)} epochs")

# Save to CSV
CSV_PATH = os.path.join(SAVE_PATH, 'training_metrics.csv')

if epochs_data:
    # Get all unique keys from all epochs
    all_keys = set()
    for epoch_data in epochs_data:
        all_keys.update(epoch_data.keys())
    
    # Define column order (epoch first, then alphabetical)
    ordered_keys = ['epoch'] + sorted([k for k in all_keys if k != 'epoch'])
    
    # Write to CSV
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(epochs_data)
    
    print(f"Saved {len(epochs_data)} epochs to {CSV_PATH}")
else:
    print("Warning: No epoch data to save!")

# %%
# Load the CSV data
df = pd.read_csv(CSV_PATH)

# Display available metrics
print("Available metrics:")
print(df.columns.tolist())

# %%
# Plot individual metrics
# Specify which metrics to plot
# METRICS_TO_PLOT = ['loss', 'rec_loss', 'kl_loss', 'unmix_loss', 'syn_data_loss', 'least_act_loss', 'val_loss', 'val_kl_loss'] #HVAE
# METRICS_TO_PLOT = ['loss', 'rec_loss', 'ortho_penalty', 'edge_penalty', 'val_rec_loss', 'val_residual_loss'] #PILA
METRICS_TO_PLOT = ['loss'] #SMPL
# Or plot all available metrics (except epoch):
# METRICS_TO_PLOT = [col for col in df.columns if col != 'epoch']

# Filter to only include metrics that exist in the dataframe
available_metrics = [m for m in METRICS_TO_PLOT if m in df.columns]
missing_metrics = [m for m in METRICS_TO_PLOT if m not in df.columns]

if missing_metrics:
    print(f"Warning: The following metrics were not found in the data: {missing_metrics}")

# Plot each metric individually
for metric in available_metrics:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(df['epoch'], df[metric], marker='o', markersize=4, linewidth=2)
    fontsize = 32
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=fontsize)
    ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.grid(True, alpha=0.3)
    
    # Save individual plot
    plot_filename = f'convergence_plot_{metric}.png'
    plot_path = os.path.join(SAVE_PATH, plot_filename)
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.show()

# %%
