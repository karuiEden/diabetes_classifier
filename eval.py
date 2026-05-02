import argparse
import logging

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from models.logistic_regression import MyLogisticRegression
from logs import setup_logs
from models.torch_models import TorchModel
from src.io import load_config, load_weights, load_preproc, load_threshold, load_data, save_metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/base.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logs(cfg, "eval")
    logger = logging.getLogger('eval')
    logger.info("Run started")
    logger.info(f'Config path: {args.config}')
    logger.info(f'Mode: {cfg["runtime"]["mode"]}, seed: {cfg["experiment"]["seed"]}')
    artifacts = cfg['experiment']['output_dir']

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(cfg)
    logger.info(f"Test_size: {len(X_test)}")
    preproc = load_preproc(artifacts + 'preproc.pkl')
    X_test = preproc.transform(X_test)
    model = TorchModel.from_file(artifacts + 'model.pt')
    threshold = load_threshold(artifacts + 'threshold.json')
    y_pred = model.predict(X_test, threshold)
    proba = model.predict_prob(X_test)
    metrics = dict()
    metrics['pr_auc'] = average_precision_score(y_test, proba)
    metrics['f1'] = f1_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    logger.info(
        f"Test metrics | pr_auc={metrics['pr_auc']:.4f} f1={metrics["f1"]:.4f} recall={metrics["recall"]:.4f} precision={metrics["precision"]:.4f}"
    )
    save_metrics(metrics, artifacts + 'metrics.json')
    logger.info("Run completed")

if __name__ == "__main__":
    main()
