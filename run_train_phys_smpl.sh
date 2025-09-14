#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use the conda environment's Python directly
PYTHON_CMD="/maps-priv/maps/ys611/miniconda3/envs/mres/bin/python"
echo "Using Python: $PYTHON_CMD"

#---------------WYTHAM DATA-----------------
# Simplified PhysVAE Framework
# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham.json

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 1.0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.5

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.1 --epochs_pretrain 20

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.1 --epochs_pretrain 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.01 --epochs_pretrain 20

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.01 --epochs_pretrain 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --epochs_pretrain 20

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --epochs_pretrain 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --epochs_pretrain 0 --r_init 1.0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --epochs_pretrain 0 --r_init 1.0 --tau_init 1.0


#---------------CAPACITY CONTROL EXPERIMENTS-----------------
# Test capacity control with sensible defaults
$PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria_capacity.json

# Test capacity control with higher capacity target
$PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria_capacity.json --use_capacity_control true --C_max 10.0 

# Test capacity control with custom parameters
$PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria_capacity.json --use_capacity_control true --C_max 3.0 

#---------------ABLATION STUDIES-----------------
# Pure auto-encoder (no KL term):
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false

# # VAE with custom beta:
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --beta_max 0.05

# Coordinated annealing schedules (all 30 epochs):
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --kl_warmup_epochs 30 --tau_warmup_epochs 30 --r_warmup_epochs 30

# Custom pretraining:
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --epochs_pretrain 10

# Control gradient flow for bias correction:
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --detach_x_P_for_bias false

# Alternative: Test the trained model
# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0906_155302/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0906_155302/models/model_best.pth \
#         # --insitu

# #---------------ALL EXPERIMENTS-----------------
# # Run Wytham RTM experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_A_wytham.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_B_wytham.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_wytham.json

# # Run Austria RTM experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_A_austria.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_B_austria.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_austria.json

# # Run Mogi experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_A.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_B.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_C.json