import os
import pandas as pd


def load_dataset(name, index_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name)
    if name.endswith("csv"):
        df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    elif name.endswith("tick") or name.endswith("parquet"):
        df = pd.read_parquet(path)
    return df
