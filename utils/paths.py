from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
PRETRAINED = ROOT / "pretrained_transformers"

def get_dataset_path(dataset_name):
    return DATASETS / dataset_name