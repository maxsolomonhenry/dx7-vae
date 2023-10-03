import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, df):

        if 'VOICE NAME' in df.keys():
            df = df.drop(columns=['VOICE NAME'])
        
        self._parameter_names = df.keys()
        self._means = df.mean().values
        self._stds = df.std().values

        df = df - df.mean()
        df = df / df.std()

        self.df = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df[idx]
    
    def get_restorer(self):
        def restore(x):
            x *= self._stds
            x += self._means
            x = np.round(x.numpy()).astype(int)
            return pd.Series(x, self._parameter_names)
        return restore