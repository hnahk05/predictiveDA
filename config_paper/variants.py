"""Load a paper config with one explicit experimental change."""
import runpy
from pathlib import Path

def make(dataset, stage='forecasting', width=64, hidden=4, ablation='full'):
    cfg = runpy.run_path(str(Path(__file__).parent / dataset / (stage+'.py')))['CFG']
    param = cfg.MODEL.PARAM if stage == 'pretraining' else cfg.MODEL.PARAM.TRANSFORMER
    param['out_channel'] = width
    tag = f'w{width}_h{hidden}_{ablation}'
    cfg.TRAIN.CKPT_SAVE_DIR = f'checkpoints/paper/{dataset}/{stage}/{tag}'
    if stage == 'forecasting':
        cfg.MODEL.PARAM.LSTTN.update(long_trend_hidden_dim=hidden, seasonality_hidden_dim=hidden, ablation=ablation,
            pretrained_path=f'pretrained_transformers/{dataset}_w{width}.pt')
    return cfg
