import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def save_weights(model, path):
    if not hasattr(model, 'get_weights'):
        raise ValueError('Model must have get_weights method')
    filename = "weights.npy"
    p = Path(path + filename)
    p.parent.mkdir(parents=True, exist_ok=True)

    w = model.get_weights()
    if hasattr(w, 'get'):
        w = w.get()
    w = np.asarray(w)
    np.save(p, w)

def load_weights(path: str):
    return np.load(path)

def save_threshold(threshold, path):
    filename = "threshold.json"
    p = Path(path + filename)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump({"best_threshold": threshold}, f, indent=2, separators=(',', ':'))

def load_threshold(path: str):
    with open(path, 'r') as f:
        return float(json.load(f)['best_threshold'])

def save_preproc(preproc, path):
    filename = "preproc.pkl"
    p = Path(path + filename)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open('wb') as f:
        pickle.dump(preproc, f)

def load_preproc(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_data(cfg: dict):
    data_path = str(cfg['data']['path'])
    target = cfg['data']['target_col']
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=cfg["data"]['test_size'], random_state=cfg["data"]['random_state'], stratify=y)
    rel_val_size = cfg["data"]['val_size'] / (1 - cfg["data"]['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=rel_val_size, random_state=cfg["data"]['random_state'], stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_config(path_str: str = "configs/base.yaml") -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path_str}')

    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Error during create config")

    return cfg

def save_splits(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, path: str):
    filename = "splits.json"
    p = Path(path + filename)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open('w') as f:
        json.dump({"train": X_train.index.tolist(), "val": X_val.index.tolist(), "test": X_test.index.tolist()}, f, indent=2)


def load_splits(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
