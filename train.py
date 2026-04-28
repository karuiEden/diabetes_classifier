import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.Preprocessor import Preprocessor
from src.logistic_regression import MyLogisticRegression


def load_config(path_str: str = "configs/base.yaml") -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path_str}')

    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Error during create config")

    return cfg

def setup_logs(cfg: dict):
    log_level = cfg['logging'].get("level", "INFO").upper()
    log_path = cfg['logging']['file']
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding='utf-8')
        ],
        force=True
    )

def load_data(cfg: dict):
    data_path = cfg['data']['path']
    target = cfg['data']['target_col']
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=cfg["data"]['test_size'], random_state=cfg["data"]['random_state'], stratify=y)
    rel_val_size = cfg["data"]['val_size'] / (1 - cfg["data"]['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=rel_val_size, random_state=cfg["data"]['random_state'], stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_best_threshold(y_proba, y_true, grid: list, by: str = 'f1'):
    metric_map = {
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
        "balanced_accuracy": balanced_accuracy_score,
    }

    if by not in metric_map:
        raise ValueError(
            f"Unknown metric_name='{by}'. "
            f"Use one of: {list(metric_map.keys())}"
        )

    metric = metric_map[by]
    best_threshold = None
    best_score = -1.0

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_proba)

    for thr in grid:
        y_pred = (y_prob > thr).astype(int)
        score = metric(y_true, y_pred)
        if score > best_score:
            best_threshold = thr
            best_score = score

    return best_threshold, best_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/base.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logs(cfg)
    logger = logging.getLogger('train')
    logger.info("Run started")
    logger.info(f'Config path: {args.config}')
    logger.info(f'Mode: {cfg["runtime"]["mode"]}, seed: {cfg["experiment"]["seed"]}')
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(cfg)
    logger.info(f"Train size: {len(X_train)}; Validation size: {len(X_val)}, Test_size: {len(X_test)}")
    num_features = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
    cat_features = ['location', 'smoking_history', 'gender']
    preproc = Preprocessor(num_features, cat_features)
    X_train = preproc.fit_transform(X_train)
    X_val = preproc.transform(X_val)
    X_test = preproc.transform(X_test)
    model = MyLogisticRegression(mode=cfg["runtime"]["mode"], tol=float(cfg["model"]["tol"]), l2=float(cfg["model"]["l2"]))
    losses = model.fit(X_train.to_numpy(), y_train.to_numpy())
    logger.info(f"Train complete. Final loss: {losses[-1]:.6f}")
    proba = model.predict_prob(X_val)
    metric = cfg["model"]["metric_for_threshold"]
    grid = cfg["model"]["threshold_grid"]
    best_threshold, score = get_best_threshold(proba, y_val, grid, metric)
    logger.info(f"Best threshold (val): {best_threshold:.3f}, best_score: {score:.4f}")
    y_pred = model.predict(X_test, best_threshold)
    proba = model.predict_prob(X_test)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(
        f"Test metrics | pr_auc={pr_auc:.4f} f1={f1:.4f} recall={recall:.4f} precision={precision:.4f}"
    )


if __name__ == '__main__':
    main()
