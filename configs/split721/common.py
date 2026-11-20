"""Opt-in new data; all forecasting hyperparameters inherited from actual runs."""
import json
import runpy
from pathlib import Path
from runners.speed_split721 import SpeedSplit721Runner

ROOT = Path(__file__).resolve().parents[2]


def make(dataset):
    if dataset not in ('METR-LA', 'PEMS-BAY'):
        raise ValueError(dataset)
    cfg = runpy.run_path(str(ROOT/'configs/actual'/(dataset+'.py')))['CFG']
    directory = ROOT/'datasets/split721_target_disjoint_v1'/dataset
    manifest = json.loads((directory/'split_manifest.json').read_text())
    assert manifest['dataset'] == dataset
    for section in (cfg.TRAIN, cfg.VAL, cfg.TEST):
        section.DATA.DIR = str(directory)
    cfg.RUNNER = SpeedSplit721Runner
    cfg.TRAIN.CKPT_SAVE_DIR = str(ROOT/'checkpoints/split721_target_disjoint_v1'/dataset)
    cfg.EXPERIMENT['data_protocol'] = 'split721_target_disjoint_v1'
    cfg.EXPERIMENT['legacy_results_comparable'] = False
    cfg.EXPERIMENT['note'] = ('Data/scaler version changed; inherited pretrained weights have '
                              'unverified provenance for this split. Do not claim strict '
                              'protocol reproduction from legacy checkpoint inference.')
    return cfg
