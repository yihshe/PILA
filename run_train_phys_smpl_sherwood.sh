#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use the conda environment's Python directly
PYTHON_CMD="/maps-priv/maps/ys611/miniconda3/envs/mres/bin/python"
echo "Using Python: $PYTHON_CMD"

#---------------WYTHAM DATA-----------------
# Simplified PhysVAE Framework
# Train AE_RTM_C (encoder being replaced with RTM + correction layer)

# $PYTHON_CMD -m pdb train_phys.py --config configs/phys/AE_RTM_C_austria.json

# $PYTHON_CMD -m pdb train_phys.py --config configs/phys/AE_Mogi_C.json

# $PYTHON_CMD -m train_phys --config configs/phys/AE_RTM_C_wytham.json 

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


#---------------ORIGINAL EXPERIMENTS (NO CAPACITY CONTROL)-----------------
# Original VAE with standard KL loss

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 1e-3

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.01 --edge_penalty_weight 1e-3

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 1e-3 --dim_z_aux 4 --residual_rank 4

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 0 --dim_z_aux 4 --residual_rank 4

# # Original VAE with different beta values
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.01

#---------------CAPACITY CONTROL EXPERIMENTS-----------------
# Test capacity control with sensible defaults
# $PYTHON_CMD -m pdb train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_austria.json --use_capacity_control true --C_max 7.0 --C_gamma 10.0 --beta_aux 0.5


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

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1013_164113_klp0_edge10_kla1_rank8_delta_penalty0/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1013_164113_klp0_edge10_kla1_rank8_delta_penalty0/models/model_best.pth    

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1013_155232_klp0_edge10_rank8_delta_penalty0/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1013_155232_klp0_edge10_rank8_delta_penalty0/models/model_best.pth  

# $PYTHON_CMD -m test_phys_mogi \
#         --config saved/mogi/PHYS_VAE_MOGI_C/1016_134316_absweight0.1_xlnvar-9/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C/1016_134316_absweight0.1_xlnvar-9/models/model_best.pth  

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1013_223315_kl0_edge1_LAIu3/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1013_223315_kl0_edge1_LAIu3/models/model_best.pth    \
#         --insitu

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1014_004110_kl0_edge1_LAIu3/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1014_004110_kl0_edge1_LAIu3/models/model_best.pth  \
#         # --insitu

# $PYTHON_CMD -m test_phys_rtm \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_143150/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_143150/models/model_best.pth  


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

# # #---------------EXPERIMENTS LoRA Inversion Mogi-----------------
# # # Mogi main results
# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --dim_z_aux 4 --residual_rank 4

# # # Mogi PhysVAE baseline
# # $PYTHON_CMD -m train_phys --config configs/phys/AE_Mogi_C.json

# # # Mogi Physics only baseline
# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_B.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 

# # # Mogi residual rank experiments
# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --dim_z_aux 1 --residual_rank 1

# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --dim_z_aux 2 --residual_rank 2

# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --dim_z_aux 6 --residual_rank 6

# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --dim_z_aux 8 --residual_rank 8

# #---------------EXPERIMENTS LoRA Inversion RTM-----------------
# # RTM main results
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --dim_z_aux 2 --residual_rank 2

# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --dim_z_aux 2 --residual_rank 2

# # RTM PhysVAE baseline
# $PYTHON_CMD -m train_phys --config configs/phys/AE_RTM_C_austria.json

# $PYTHON_CMD -m train_phys --config configs/phys/AE_RTM_C_wytham.json

# # RTM Physics only baseline
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_B_austria.json --use_kl_term_z_phy false --edge_penalty_weight 1.0

# # $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_B_wytham.json --use_kl_term_z_phy false --edge_penalty_weight 1.0

# # RTM prior KL term experiments
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term_z_phy true --beta_max_z_phy 1.0 --dim_z_aux 2 --residual_rank 2

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term_z_phy true --beta_max_z_phy 0.1 --dim_z_aux 2 --residual_rank 2

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term_z_phy true --beta_max_z_phy 0.01 --dim_z_aux 2 --residual_rank 2

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term_z_phy true --beta_max_z_phy 0.01 --edge_penalty_weight 1.0 --dim_z_aux 2 --residual_rank 2

# #---------------EVALUATION LoRA Inversion Mogi-----------------
# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_B_SMPL/1016_192709/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_B_SMPL/1016_192709/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi \
#         --config saved/mogi/PHYS_VAE_MOGI_C/1016_192308/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C/1016_192308/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_191930/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_191930/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193002/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193002/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193329/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193329/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193707/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_193707/models/model_best.pth \

# $PYTHON_CMD -m test_phys_mogi_smpl \
#         --config saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_194050/models/config.json \
#         --resume saved/mogi/PHYS_VAE_MOGI_C_SMPL/1016_194050/models/model_best.pth \

#---------------EVALUATION LoRA Inversion RTM-----------------
# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_B_AUSTRIA_SMPL/1016_225547/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_B_AUSTRIA_SMPL/1016_225547/models/model_best.pth \

$PYTHON_CMD -m test_phys_rtm_smpl \
        --config saved/rtm/PHYS_VAE_RTM_B_AUSTRIA_SMPL/1027_194202/models/config.json \
        --resume saved/rtm/PHYS_VAE_RTM_B_AUSTRIA_SMPL/1027_194202/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_B_WYTHAM_SMPL/1017_131732/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_B_WYTHAM_SMPL/1017_131732/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_B_WYTHAM_SMPL/1017_131732/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_B_WYTHAM_SMPL/1017_131732/models/model_best.pth \
#         --insitu

# $PYTHON_CMD -m test_phys_rtm \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_202135/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA/1016_202135/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1016_181644/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1016_181644/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_010247/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_010247/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_023422/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_023422/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_042352/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_042352/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_063322/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/1017_063322/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm \
#         --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1023_224414/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1023_224414/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm \
#         --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1023_224414/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1023_224414/models/model_best.pth \
#         --insitu

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1023_234650/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1023_234650/models/model_best.pth \

# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1023_234650/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/1023_234650/models/model_best.pth \
#         --insitu

#/maps/ys611/MAGIC/saved_archived/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/0923_023534_kl0_edge1_rank4/models

# $PYTHON_CMD -m test_phys_rtm \
#         --config /maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1017_033515/models/config.json \
#         --resume /maps/ys611/MAGIC/saved/rtm/PHYS_VAE_RTM_C_WYTHAM/1017_033515/models/model_best.pth \
#         --full_region
