import logging
from pathlib import Path


def setup_logs(cfg: dict, mode: str = "train"):
    log_level = cfg['logging'].get("level", "INFO").upper()
    log_path = cfg['logging']['train_file'] if mode == "train" else cfg['logging']['eval_file']
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