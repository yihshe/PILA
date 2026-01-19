<<<<<<< HEAD
# PILA: Physics-Informed Low-Rank Augmentation for Interpretable Earth Observation
Yihang She, Andrew Blake, Clement Atzberger, Adriano Gualandi, Srinivasan Keshav

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Keywords: physics-informed machine learning, inverse problems, incomplete physical models, low-rank augmentation, Earth observation
=======
# PILA: Physics-Informed Low Rank Augmentation for Interpretable Earth Observation
Physically meaningful representations are essential for Earth Observation (EO), yet existing physical models are often simplified and incomplete. This leads to discrepancies between simulation and observations that hinder reliable forward model inversion. Common approaches to EO inversion either ignored this incompleteness or relied on case-specific preprocessing. More recent methods use physics-informed autoencoders but depend on auxiliary variables that are difficult to interpret and multiple regularizers that are difficult to balance.
We propose Physics-Informed Low-Rank Augmentation (PILA), a framework that augments incomplete physical models using a learnable low-rank residual to improve flexibility, while remaining close to the governing physics. 

We evaluate PILA on two EO inverse problems involving diverse physical processes: forest radiative transfer inversion from optical remote sensing; and volcanic deformation inversion from Global Navigation Satellite Systems (GNSS) displacement data.
Across different domains, PILA yields more accurate and interpretable physical variables. For forest spectral inversion, it improves the separation of tree species and, compared to ground measurements, reduces prediction errors by 40-71\% relative to the state-of-the-art. For volcanic deformation, PILA's recovery of variables captures a major inflation event at the Akutan volcano in 2008, and estimates source depth, volume change, and displacement patterns that are consistent with prior studies that however required substantial additional preprocessing.
Finally, we analyse the effects of model rank, observability, and physical priors, and suggest that PILA may offer an effective general pathway for inverting incomplete physical models even beyond the domain of Earth Observation.
>>>>>>> 398a13dfbb05e2467ee74dfc639da65ed35fd24c

This repository contains the code for PILA as described in:
https://arxiv.org/abs/2405.18953 

PILA stands for **Physics-Informed Low-Rank Augmentation**.

PILA augments a forward physics model with a low-rank, learnable refinement
to address model incompleteness during inversion.

![Teaser figure](figures/intro_teaser_figure.png)
*We study inverse problems arising from Earth observations of disparate physical processes, to understand the planet, from the surface to the subsurface. Study cases include: the inversion of a forest radiative transfer model---a planet renderer---to estimate biophysical status; and the inversion of a volcanic deformation model---representative of a broad family of geophysical inverse problems---to infer subsurface geophysical activity.*

![PILA method diagram](figures/methods_pila.png)
*PILA inverts physical models of varying incompleteness using a residual of intrinsically low rank. Given an observation X, E_R maps it to a high-dimensional feature R, which is then encoded by E_phy and E_aux into physical variables Z_phy and auxiliary variables Z_aux. Z_phy is decoded by F to produce a physical reconstruction X_F, which is refined by a low-rank residual Δ to yield the final reconstruction X_C. The residual Δ is computed by a mapping C as the product of a scaling factor s, a coefficient matrix A in R^{n x r}, and a residual basis matrix B in R^{d x r}, with rank(Δ) = r << d. Here, s and B are shared parameters applied to all samples, while A is obtained by linearly mapping the concatenation of Z_aux and the physical output X_F, with a stop-gradient operation applied to X_F during backpropagation.*

## Key Components

- **PILA (our method)**: `configs/phys_smpl/`, `train_pila.py`,
  `model/model_phys_smpl.py`, `trainer/trainer_phys_smpl.py`

We study two inversion problems:
- **Mogi inversion** (GNSS)
- **RTM inversion** (Austria and Wytham datasets)

## Environment

```bash
./setup_environment.sh
```

<<<<<<< HEAD
Or manually:

```bash
conda env create -f environment.yml
conda activate pila
```

## Training

PILA (our method):

```bash
python train_pila.py --config configs/phys_smpl/PILA_Mogi_C.json
python train_pila.py --config configs/phys_smpl/PILA_RTM_C_austria.json
python train_pila.py --config configs/phys_smpl/PILA_RTM_C_wytham.json
```

See `run.sh` for additional examples.

## Evaluation

```bash
python test_pila_mogi.py \
  --config saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
  --resume saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth

python test_pila_rtm.py \
  --config saved/rtm/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
  --resume saved/rtm/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth \
  --insitu
```

Equivalent HVAE commands are available in `run.sh`.

## Baseline (HVAE)

HVAE stands for **Hybrid Auto-Encoder** and is used as a baseline for
comparison. For details, see Takeishi and Kalousis (2021):
https://proceedings.neurips.cc/paper_files/paper/2021/file/7ca57a9f85a19a6e4b9a248c1daca185-Paper.pdf

To reproduce the baseline:

```bash
python train_hvae.py --config configs/phys/HVAE_Mogi_C.json
python train_hvae.py --config configs/phys/HVAE_RTM_C_austria.json
python train_hvae.py --config configs/phys/HVAE_RTM_C_wytham.json
```

## Configs and Variants

Configs follow the A/B/C variants:
- **A**: no physics (`no_phy: true`)
- **B**: physics only (`dim_z_aux: 0`)
- **C**: physics + refinement (PILA residual)

PILA configs live in `configs/phys_smpl/`. HVAE configs live in `configs/phys/`.

## Project Structure

```
PILA/
├── base/                 # Base classes
├── configs/              # Experiment configs
│   ├── phys/             # HVAE configs
│   └── phys_smpl/        # PILA configs
├── data/                 # Data (see notes below)
├── data_loader/          # Data loaders
├── datasets/             # Dataset preprocessing utilities
├── figures/              # Figures used in README/paper
├── model/                # PILA/HVAE models, losses, metrics
├── physics/              # Forward physics models (RTM, Mogi)
├── pretrained/           # Pretrained checkpoints (to be updated)
├── trainer/              # PILA/HVAE trainers
├── utils/                # Utilities
└── run.sh                # Example commands
```

## Data and Checkpoints

- `data/` will be updated to include all datasets used in the paper.
- Pretrained checkpoints will be added; key models will be shared via an
  external link (to be provided).

## Adding a New Physics Model

1. Implement the forward model under `physics/your_model/`.
2. Add a new config under `configs/phys_smpl/` (PILA).
3. Update/extend the data loader if your input format changes.
4. Train with `train_pila.py` or `train_hvae.py`.

## Citation

```bibtex
@misc{she2025pilaphysicsinformedlowrank,
      title={PILA: Physics-Informed Low Rank Augmentation for Interpretable Earth Observation},
      author={Yihang She and Andrew Blake and Clement Atzberger and Adriano Gualandi and Srinivasan Keshav},
      year={2025},
      eprint={2405.18953},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.18953},
}
```
=======
## Folder Structure
  ```
PILA/
  │
  ├── train.py - main script to start training
  ├── test_AE_Mogi.py - evaluation of trained models for Mogi
  ├── test_AE_RTM.py - evaluation of trained models for RTM
  ├── test_NN_RTM.py -evaluation of trained models for RTM regressor baseline
  │
  ├── configs/ - holds configuration for training
  │   ├── AE_Mogi_A.json - configuration file for training M_A_Mogi
  │   ├── ...
  │   ├── mogi_paras.json - learnable Mogi parameters with known ranges
  │   ├── rtm_paras.json - learnable RTM parameters with known ranges
  │   └── station_info.json - information of 12 GNSS stations
  │
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - data loading for both Sentinel-2 and GNSS data
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │   ├── processed/ - processed data ready for training and evaluation
  │   └── raw/ - raw data 
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  
  │
  ├── physics/ - Forward physical models
│   ├── rtm/ - PyTorch implementation of RTM model
│   ├── mogi/ - PyTorch implementation of Mogi model
│   ├── rtm_numpy/ - NumPy implementation of RTM model
│   └── dpm/ - DPM model implementation
  │
  ├── pretrained/ - pretrained models for evaluation
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── rtm_unit_test.py - unit test for the PyTorch implementation of RTM
  ```

## Training
TBD

## Evaluation
TBD
>>>>>>> 398a13dfbb05e2467ee74dfc639da65ed35fd24c
