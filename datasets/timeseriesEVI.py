import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
from datasets import PARENT_DIR

class TimeSeriesEVI(data.Dataset):
    def __init__(self, csv_path):
        super(TimeSeriesEVI, self).__init__()
        # the dataset is a tabular data of EVI time series data
        # each row is a time series of EVI values for a specific location of the year
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        data_dict = {}

        evi_values = sample[:23].values.astype('float32')
        data_dict['evi'] = torch.tensor(evi_values).to(torch.float32)
        
        return data_dict