# PILA: Physics-Informed Low Rank Augmentation for Interpretable Earth Observation
Physically meaningful representations are essential for Earth Observation (EO), yet existing physical models are often simplified and incomplete. This leads to discrepancies between simulation and observations that hinder reliable forward model inversion. Common approaches to EO inversion either ignored this incompleteness or relied on case-specific preprocessing. More recent methods use physics-informed autoencoders but depend on auxiliary variables that are difficult to interpret and multiple regularizers that are difficult to balance.
We propose Physics-Informed Low-Rank Augmentation (PILA), a framework that augments incomplete physical models using a learnable low-rank residual to improve flexibility, while remaining close to the governing physics. 

We evaluate PILA on two EO inverse problems involving diverse physical processes: forest radiative transfer inversion from optical remote sensing; and volcanic deformation inversion from Global Navigation Satellite Systems (GNSS) displacement data.
Across different domains, PILA yields more accurate and interpretable physical variables. For forest spectral inversion, it improves the separation of tree species and, compared to ground measurements, reduces prediction errors by 40-71\% relative to the state-of-the-art. For volcanic deformation, PILA's recovery of variables captures a major inflation event at the Akutan volcano in 2008, and estimates source depth, volume change, and displacement patterns that are consistent with prior studies that however required substantial additional preprocessing.
Finally, we analyse the effects of model rank, observability, and physical priors, and suggest that PILA may offer an effective general pathway for inverting incomplete physical models even beyond the domain of Earth Observation.

## Requirements
To install requirements:
```
pip3 install -r requirements.txt
```

## Folder Structure
  ```
MAGIC/
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
