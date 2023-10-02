import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, df):

        if 'VOICE NAME' in df.keys():
            df = df.drop(columns=['VOICE NAME'])
        
        self.parameter_names = df.keys()
        self.means = df.mean()
        self.stds = df.std()

        df = df - df.mean()
        df = df / df.std()

        self.df = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df[idx]