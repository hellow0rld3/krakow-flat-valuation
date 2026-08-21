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
    #Sprawdzenie czy w przyszlosci po update datasetu nie dojdzie jakas nowa dzielnica w skutku bledu lub zlego zczytania
    extra = set(df[GROUP]) - set(lookup)
    if extra:
        raise ValueError(f"Pojawiła się nowa, nieznana dzielnica: {extra}")
    idx = df[GROUP].map(lookup).to_numpy(dtype=np.int32)
    return idx, list(levels)

def split(df, fraction=0.2, seed=1234, min_ofert=10):
    #Na wypadek gdyby ktos podal ramke z powtorzonymi etykietami
    df = df.reset_index(drop=True)
    #Daje prog 10, tak aby dzielnice gdzie nie ma nawet 10 ofert nie byly przenoszone do datasetu testowego. Obecnie nie wplywa to na zadna dzielnice ale jest to zabezpieczenie na przyszlosc
    duze = df.groupby(GROUP).filter(lambda x: len(x) >= min_ofert)
    test = duze.groupby(GROUP).sample(frac=fraction, random_state=seed)
    train = df.drop(test.index)
    return train, test

class Scaler():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Zera zamieniam na jedynki aby uniknac dzielenia przez zero w transform
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    #JSON nie zapisze tablic numpy, więc konwertujemy je na listy przy zapisie i z powrotem przy odczycie
    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean = np.array(d["mean"]), std = np.array(d["std"]))

    



