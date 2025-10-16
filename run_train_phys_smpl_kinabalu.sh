#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use the conda environment's Python directly
PYTHON_CMD="/maps-priv/maps/ys611/miniconda3/envs/mres/bin/python"
echo "Using Python: $PYTHON_CMD"



# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham_v3.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria_v2.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false

# $PYTHON_CMD -m train_phys_smpl --config configs/phys/AE_RTM_C.json 

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham_v2.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_wytham/AE_RTM_C.json 

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham_v3.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false

# $PYTHON_CMD -m train_phys --config configs/phys/AE_Mogi_C.json

$PYTHON_CMD -m train_phys --config configs/phys/AE_RTM_C_austria.json

# $PYTHON_CMD -m pdb train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false --dim_z_aux 4 --residual_rank 4 --use_time_in_residual false

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false --dim_z_aux 1 --residual_rank 1 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false --dim_z_aux 2 --residual_rank 2 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false --dim_z_aux 4 --residual_rank 4 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --edge_penalty_power 2.0 --use_kl_term_z_aux false --dim_z_aux 4 --residual_rank 4 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 1.0 --use_kl_term_z_aux false --dim_z_aux 8 --residual_rank 8 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --use_kl_term_z_aux true --beta_max_z_aux 1.0 --dim_z_aux 4 --residual_rank 4 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_C.json --use_kl_term_z_phy false --edge_penalty_weight 10.0 --use_kl_term_z_aux true --beta_max_z_aux 1.0 --dim_z_aux 8 --residual_rank 8 --coeff_penalty_weight 0 --delta_penalty_weight 0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_B.json --use_kl_term_z_phy false --edge_penalty_weight 1.0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_Mogi_B.json --use_kl_term_z_phy false --edge_penalty_weight 10.0












# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 1.0 --use_ema_prior true --ema_momentum 0.99

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 1.0 --use_ema_prior true --ema_momentum 0.99 --r_init 1.0

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term true --beta_max 0.01 --use_ema_prior true --ema_momentum 0.99

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 10

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 1e-2 --dim_z_aux 4 --residual_rank 4

# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json --use_kl_term false --edge_penalty_weight 0 --dim_z_aux 4 --residual_rank 4

