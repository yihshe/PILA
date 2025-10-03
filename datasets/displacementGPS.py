import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
import math
import datetime
from datasets import PARENT_DIR


def time_feats(d, time_feat_dim=4):
    """
    Convert date to temporal features (annual and semi-annual cycles).
    
    Args:
        d: datetime.date or 'YYYY-MM-DD' string
        time_feat_dim: dimension of time features (2 for annual only, 4 for annual+semi-annual)
    
    Returns:
        np.array: temporal features [sin(annual), cos(annual), sin(semi-annual), cos(semi-annual)]
    """
    if isinstance(d, str):
        # Handle different date formats: 'YYYY-MM-DD' or 'YYYY.MM.DD'
        if '-' in d:
            y, m, dd = map(int, d.split('-'))
        elif '.' in d:
            y, m, dd = map(int, d.split('.'))
        else:
            raise ValueError(f"Unsupported date format: {d}")
        d = datetime.date(y, m, dd)
    
    doy = d.timetuple().tm_yday
    two_pi = 2 * math.pi
    
    # Use a consistent year length for all calculations
    year_length = 365.25
    
    # Annual cycle (normalized by consistent year length)
    a1 = two_pi * (doy / year_length)
    annual_feats = [math.sin(a1), math.cos(a1)]
    
    if time_feat_dim == 4:
        # Semi-annual cycle
        a2 = two_pi * (2 * doy / year_length)
        semi_annual_feats = [math.sin(a2), math.cos(a2)]
        return np.array(annual_feats + semi_annual_feats, dtype=np.float32)
    elif time_feat_dim == 2:
        return np.array(annual_feats, dtype=np.float32)
    else:
        raise ValueError(f"time_feat_dim must be 2 or 4, got {time_feat_dim}")


class DisplacementGPS(data.Dataset):
    def __init__(self, csv_path):
        super(DisplacementGPS, self).__init__()
        # the dataset is a tabular data of GPS displacement data
        # each row is displacements from 12 stations at the same time point
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample[:36].values.astype('float32')
        ).to(torch.float32)
        data_dict['date'] = sample[-3]
        
        # Always add time features (4-dim: annual + semi-annual)
        # The model will use only the first time_feat_dim elements
        date_str = sample[-3]  # Assuming date is in the last column
        time_features = time_feats(date_str, time_feat_dim=4)  # Always generate 4-dim features
        data_dict['time_feats'] = torch.tensor(time_features).to(torch.float32)

        return data_dict

# dataset by slicing data into sequences with temporal smoothness
class DisplacementGPSSeq(data.Dataset):
    def __init__(self, csv_path, seq_len=7, mode='train'):
        super(DisplacementGPSSeq, self).__init__()
        self.seq_len = seq_len
        self.mode = mode  # 'train' or 'inference'
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))
        self.data_df['date'] = pd.to_datetime(self.data_df['date'])
        self.data_df = self.data_df.sort_values(by='date')

    def __len__(self):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference
            return len(self.data_df) // self.seq_len
        else:
            # Overlapping sequences for training
            return max(1, len(self.data_df) - self.seq_len + 1)

    def __getitem__(self, idx):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference (no repeated dates)
            start_idx = idx * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx > len(self.data_df):
                # Handle the last incomplete sequence
                start_idx = max(0, len(self.data_df) - self.seq_len)
                end_idx = len(self.data_df)
            sample = self.data_df.iloc[start_idx:end_idx]
        else:
            # Random overlapping sequences for training
            max_start_idx = len(self.data_df) - self.seq_len
            start_idx = np.random.randint(0, max_start_idx + 1)
            sample = self.data_df.iloc[start_idx:start_idx+self.seq_len]
        
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample.iloc[:, :36].values.astype('float32')
        ).to(torch.float32)
        
        # Always add time features (4-dim: annual + semi-annual)
        # The model will use only the first time_feat_dim elements
        dates = sample['date'].values
        time_features_list = []
        for date in dates:
            # Convert pandas timestamp to string format
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            time_features = time_feats(date_str, time_feat_dim=4)  # Always generate 4-dim features
            time_features_list.append(time_features)
        
        data_dict['time_feats'] = torch.tensor(
            np.array(time_features_list, dtype=np.float32)
        ).to(torch.float32)

        return data_dict
