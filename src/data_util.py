import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def clean_data(df):
    # Remove patches with unrecognized operator algorithm number.
    df = df[df['ALGORITHM #'] <= 31]
    return df


def _get_categorical_features():
    categorical_columns = []
    for op in range(1, 7):
        categorical_columns += [f"OP{op} KBD LEV SCL LFT CURVE"]
        categorical_columns += [f"OP{op} KBD LEV SCL RHT CURVE"]
        categorical_columns += [f"OP{op} OSC MODE"]

    categorical_columns += ["ALGORITHM #", "OSCILLATOR SYNC", "LFO SYNC", "LFO WAVEFORM"]
    return categorical_columns

def onehot_decode(df):
    for feature in _get_categorical_features():
        onehot_group = [col for col in df.columns if col.startswith(feature)]

        # Extract the one-hot number from the column name (e.g., "TEST_3" -> 3).
        which_category = df[onehot_group].idxmax(axis=1)
        category_number = which_category.apply(lambda name: int(name.split("_")[-1]))

        # Replace original feature column with a numerical value.
        df[feature] = category_number

        # Remove (now unneeded) one-hot columns.
        df = df.drop(columns=onehot_group)
    return df


def onehot_encode(df):
    # Turn all categorical data into one-hot vectors.
    return pd.get_dummies(df, columns=_get_categorical_features())


class PatchDataset(Dataset):
    def __init__(self, df):

        if 'VOICE NAME' in df.keys():
            df = df.drop(columns=['VOICE NAME'])
        
        self._parameter_names = df.keys()
        self._means = df.mean().values
        self._stds = df.std().values

        # Force one-hot vector means and stds so they are not standardized.
        onehot_columns = []
        for feature in _get_categorical_features():
            onehot_columns += [col for col in df.columns if col.startswith(feature)]

        onehot_col_idx = [df.columns.get_loc(col) for col in onehot_columns]

        self._means[onehot_col_idx] = 0
        self._stds[onehot_col_idx] = 1

        df = df - self._means
        df = df / self._stds

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