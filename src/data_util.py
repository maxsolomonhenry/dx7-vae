import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get_categorical_features():
    categorical_columns = []
    for op in range(1, 7):
        categorical_columns += [f"OP{op} KBD LEV SCL LFT CURVE"]
        categorical_columns += [f"OP{op} KBD LEV SCL RHT CURVE"]
        categorical_columns += [f"OP{op} OSC MODE"]

    categorical_columns += ["ALGORITHM #", "OSCILLATOR SYNC", "LFO SYNC", "LFO WAVEFORM"]
    return categorical_columns


def clean_data(df):
    # Remove patches with unrecognized operator algorithm number.
    df = df[df['ALGORITHM #'] <= 31]
    return df


def onehot_decode(df):

    # Prase differential behaviour to work with either DataFrames or Series.
    # This makes the function a little messy. Sorry.

    is_dataframe = isinstance(df, pd.DataFrame)

    input_columns = df.columns if is_dataframe else df.index
    which_axis = 1 if is_dataframe else 0

    for feature in _get_categorical_features():
        onehot_group = [col for col in input_columns if col.startswith(feature)]

        # Extract the one-hot number from the column name (e.g., "TEST_3" -> 3).
        which_category = df[onehot_group].idxmax(axis=which_axis)

        if is_dataframe:
            category_number = which_category.apply(lambda name: int(name.split("_")[-1]))
        else:
            category_number = int(which_category.split("_")[-1])

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
        
        self.decoder_info = self.get_decoder_info(df.keys())

        self._parameter_names = df.keys()
        self._means = df.mean().values
        self._stds = df.std().values

        # Manually set one-hot vector means and stds so they are not standardized.
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
    
    def get_decoder_info(self, keys):
        # Determine n_neurons for each decoder head.

        base = ""
        ctr = 0
        all_counts = []

        for key in keys:
            new_base = key.split("_")[0]

            if new_base == base:
                ctr += 1
            else:
                all_counts.append(ctr + 1)
                ctr = 0
                base = new_base

        all_counts.append(ctr + 1)

        for i, element in enumerate(all_counts):
            if element > 1:
                break

        return [i - 1] + all_counts[i:]
    
    def get_restorer(self):
        def restore(x):
            x *= self._stds
            x += self._means
            x = np.round(x.numpy()).astype(int)
            return pd.Series(x, self._parameter_names)
        return restore