"""Adapt author configs without editing their files or changing model/LR/split."""
import ast
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]


def make(dataset, epochs, batch, local=False):
    path = ROOT / 'configs' / dataset / 'forecasting.py'
    graph_path = None
    for statement in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'GRAPH_PKL_PATH' for target in statement.targets):
            graph_path = ast.literal_eval(statement.value)
    # Original METR-LA/PEMS04 configs use ../datasets; BAY/08 use datasets.
    # Resolve that legacy working-directory difference only during config loading.
    previous = Path.cwd()
    try:
        os.chdir(ROOT/'configs' if graph_path and graph_path.startswith('../datasets/') else ROOT)
        cfg = runpy.run_path(str(path))['CFG']
    finally:
        os.chdir(previous)
    original = dict(gpu_num=cfg.GPU_NUM, loader_batch=cfg.TRAIN.DATA.BATCH_SIZE,
                    epochs=cfg.TRAIN.NUM_EPOCHS)
    cfg.GPU_NUM = 1
    cfg.USE_GPU = True
    cfg.GLOBAL_BATCH_SIZE = batch
    cfg.TRAIN.NUM_EPOCHS = epochs
    cfg.TRAIN.CKPT_SAVE_DIR = str(ROOT/'checkpoints/actual'/dataset)
    for section in (cfg.TRAIN, cfg.VAL, cfg.TEST):
        section.DATA.DIR = str(ROOT/'datasets'/dataset)
        section.DATA.BATCH_SIZE = batch
        if local:
            section.DATA.NUM_WORKERS = 2
            section.DATA.PREFETCH = False
    if local:
        from runners.local_lsttn import LocalLSTTNRunner
        cfg.RUNNER = LocalLSTTNRunner
        cfg.TRAIN.SETUP_GRAPH = False
        cfg.TRAIN.KEEP_LAST_EPOCHS = 2
    if dataset == 'METR-LA' and local:
        cfg.TEST.DATA.NUM_WORKERS = 0
        cfg.TEST.DATA.PIN_MEMORY = False
    cfg.MODEL.PARAM.LSTTN['pretrained_path'] = str(ROOT/'pretrained_transformers'/(dataset+'.pt'))
    cfg.EXPERIMENT = dict(dataset=dataset, role='exploratory' if local else 'main',
        author_config=str(path.relative_to(ROOT)), author_declared_settings=original,
        actual_epochs=epochs, actual_global_batch=batch, actual_gpu_num=1,
        pretrained='provided author checkpoint; no pretraining run',
        model='author width/hidden/LR/scheduler/curriculum/graph retained',
        note='Recorded settings; original run paths are in results/experiment_manifest.json')
    return cfg
