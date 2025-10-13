import numpy as np
import torch

class DPM():
    """
    Dynamic Phenology Model for crop EVI time series simulation.
    
    Implements a time-segmented mixing model for agricultural land use:
    - Spring (DOY 1-170): Mixed wheat + background  
    - Summer (DOY 170-365): Mixed rice + maize + background
    
    Each crop follows a double-logistic phenology model:
    EVI(t) = (M - m) * (S_sos_mat(t) - S_sen_eos(t)) + m
    
    Attributes
    ----------
    time_points : Tensor
        23 time points (DOY) for EVI sampling
    Authors
    ----------
    Bo Han, Nanjing Univeristy. Yihang She, Univeristy of Cambridge. 2025.
    """
    def __init__(self, time_points=None, **kwargs):
        super(DPM, self).__init__()
        """
        Initialize the Dynamic Phenology Model.

        Parameters:
        time_points (Tensor): 23 time points (DOY) for EVI sampling.
        """
        if time_points is None:
            # DOY 1, 17, 33, 49, ..., 353
            self.time_points = torch.arange(1, 354, 16)  # from 1 to 353 with step 16
        else:
            self.time_points = time_points

        # Background parameters for EVI simulation
        # These parameters can be adjusted based on the specific dataset or region
        self.background_params = {
            'base_evi': 0.3,
            'amplitude': 0.15,
            'peak_doy': 130
        }
        
    def phenology_model(self, t, M, m, sos, mat, sen, eos):
        """
        Double logistic phenology model: EVI(t) = (M-m)*(S1(t) - S2(t)) + m
        
        Parameters:
            t: time points tensor [time_points]
            M, m, sos, mat, sen, eos: phenology parameters tensor [batch_size]
            
        Returns:
            evi: EVI values tensor [batch_size, time_points]
        """
        # Ensure dimension matching: [batch_size, 1] and [1, time_points] broadcasting
        t = t.unsqueeze(0)  # [1, time_points]
        M = M.unsqueeze(1)  # [batch_size, 1]
        m = m.unsqueeze(1)
        sos = sos.unsqueeze(1)
        mat = mat.unsqueeze(1) 
        sen = sen.unsqueeze(1)
        eos = eos.unsqueeze(1)
        
        # First logistic function (growth activation)
        exp_arg1 = 2 * (sos + mat - 2*t) / (mat - sos + 1e-6)  # avoid division by zero
        exp_arg1 = torch.clamp(exp_arg1, -500, 500)  # numerical stability
        S_sos_mat = 1 / (1 + torch.exp(exp_arg1))
        
        # Second logistic function (senescence activation)  
        exp_arg2 = 2 * (sen + eos - 2*t) / (eos - sen + 1e-6)
        exp_arg2 = torch.clamp(exp_arg2, -500, 500)
        S_sen_eos = 1 / (1 + torch.exp(exp_arg2))
        
        # Phenology model
        evi = (M - m) * (S_sos_mat - S_sen_eos) + m
        evi = torch.clamp(evi, 0.0, 1.0)  # EVI range constraint
        
        return evi
    
    def calculate_background_evi(self, t):
        """
        Calculate background EVI (simplified cosine function)
        
        Parameters:
            t: time points tensor [time_points]
            
        Returns:
            background_evi: background EVI [time_points]  
        """
        seasonal_variation = self.background_params['amplitude'] * torch.cos(
            2 * torch.pi * (t - self.background_params['peak_doy']) / 365
        )
        background_evi = self.background_params['base_evi'] + seasonal_variation
        return torch.clamp(background_evi, 0.1, 0.6)

    def run(self, **paras):
        """
        Run DPM forward model with time-segmented mixing
        
        Parameters:
            **paras: Dictionary containing all 21 DPM parameters
                rice_M, rice_m, rice_sos, rice_mat, rice_sen, rice_eos, rice_fraction
                maize_M, maize_m, maize_sos, maize_mat, maize_sen, maize_eos, maize_fraction  
                wheat_M, wheat_m, wheat_sos, wheat_mat, wheat_sen, wheat_eos, wheat_fraction
                
        Returns:
            mixed_evi: Mixed EVI time series [batch_size, 23]
        """
        # Get batch information
        batch_size = paras['rice_M'].shape[0]
        device = paras['rice_M'].device
        time_points = self.time_points.to(device)
        
        # === Step 1: Calculate EVI for each crop ===
        
        # Rice EVI
        rice_evi = self.phenology_model(
            time_points,
            paras['rice_M'], paras['rice_m'],
            paras['rice_sos'], paras['rice_mat'], 
            paras['rice_sen'], paras['rice_eos']
        )
        
        # Maize EVI  
        maize_evi = self.phenology_model(
            time_points,
            paras['maize_M'], paras['maize_m'],
            paras['maize_sos'], paras['maize_mat'],
            paras['maize_sen'], paras['maize_eos'] 
        )
        
        # Wheat EVI
        wheat_evi = self.phenology_model(
            time_points,
            paras['wheat_M'], paras['wheat_m'],
            paras['wheat_sos'], paras['wheat_mat'],
            paras['wheat_sen'], paras['wheat_eos']
        )
        
        # === Step 2: Calculate background EVI ===
        background_evi = self.calculate_background_evi(time_points)
        background_evi = background_evi.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 23]
        
        # === Step 3: Parse area parameters ===
        A1 = paras['wheat_fraction'].unsqueeze(1)  # [batch_size, 1]
        A2 = paras['rice_mix_maize_fraction'].unsqueeze(1)    # [batch_size, 1]
        A3 = paras['maize_in_mix_fraction'].unsqueeze(1)   # [batch_size, 1]
        
        # === Step 4: Time-segmented mixing ===
        mixed_evi = torch.zeros(batch_size, len(time_points), device=device)
        
        for i, doy in enumerate(time_points):
            if doy <= 170:  # Spring: wheat + background
            # Spring mixing: A1*wheat + (1-A1)*background
                mixed_evi[:, i] = (A1 * wheat_evi[:, i] + 
                                  (1 - A1) * background_evi[:, i])
            else:  # Summer: rice + maize + background
                # Summer mixing: A2*rice + A3*maize + (1-A2-A3)*background  
                mixed_evi[:, i] = (A2.squeeze(1)*(1-A3.squeeze(1)) * rice_evi[:, i] + 
                                  A2.squeeze(1)*A3.squeeze(1) * maize_evi[:, i] + 
                                  (1 - A2.squeeze(1)) * background_evi[:, i])
        
        return mixed_evi