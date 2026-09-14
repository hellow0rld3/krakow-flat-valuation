import pandas as pd
import numpy as np
from pathlib import Path

FEATURES = ["log_metraz_m2", "pietro", "liczba_pokoi"]
TARGET = "log_cena_pln"
GROUP = "dzielnica"

DEFAULT_PATH = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "krakow.csv"
)

def load_data(path=None):
    path = Path(path) if path else DEFAULT_PATH
    return pd.read_csv(path)

def prepare_dataset(df):
    needed = FEATURES + [TARGET, GROUP]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[needed]
    return df

def encode(df, levels=None):
    if levels is None:
        levels = sorted(df[GROUP].unique())
    lookup = {name: i for i, name in enumerate(levels)}
    #Catch districts missing from levels - after a dataset update this usually means a data error, not a genuinely new district
    extra = set(df[GROUP]) - set(lookup)
    if extra:
        raise ValueError(f"Unknown district not present in levels: {extra}")
    idx = df[GROUP].map(lookup).to_numpy(dtype=np.int32)
    return idx, list(levels)

def split(df, fraction=0.2, seed=1234, min_offers=10):
    #In case someone passes a frame with duplicate index labels
    df = df.reset_index(drop=True)
    #Districts with fewer than 10 listings are kept out of the test set. No district is below the threshold today, but this guards against it as the dataset grows
    large = df.groupby(GROUP).filter(lambda x: len(x) >= min_offers)
    test = large.groupby(GROUP).sample(frac=fraction, random_state=seed)
    train = df.drop(test.index)
    return train, test

class Scaler():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Replace zeros with ones so transform never divides by zero
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    #JSON cannot store numpy arrays, so we convert them to lists on save and back on load
    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean = np.array(d["mean"]), std = np.array(d["std"]))


def build(df, levels=None, scaler=None):
    """Takes a frame already processed by prepare_dataset and returns X, y, group_idx, levels and scaler"""
    X_raw = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)
    group_idx, levels = encode(df, levels)

    if scaler is None:
        scaler = Scaler()
        scaler.fit(X_raw)
    X = scaler.transform(X_raw)

    return {"X" : X, "y" : y, "group_idx" : group_idx, "levels" : levels, "scaler" : scaler}
