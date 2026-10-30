"""One isolated training attempt, called by run_experiments.py."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--attempt', required=True, type=Path)
    p.add_argument('--pretrained-dir', type=Path)
    p.add_argument('--resume-dir', type=Path)
    p.add_argument('--stop-at')
    a = p.parse_args()
    from easytorch import launch_training
    cfg = runpy.run_path(a.config)['CFG']
    cfg.TRAIN.CKPT_SAVE_DIR = str(a.attempt / 'checkpoints')
    if a.stop_at: cfg.TRAIN.REPRO_STOP_AT = a.stop_at
    if a.resume_dir:
        import torch
        candidates = list(a.resume_dir.rglob('*_[0-9]*.pt'))
        candidates.sort(key=lambda path:int(path.stem.rsplit('_',1)[1]), reverse=True)
        for path in candidates:
            try:
                state = torch.load(path, map_location='cpu')
                if 'reproduction' not in state: continue
                cfg.TRAIN.REPRO_RESUME_FROM = str(path)
                print('RESUME epoch', state['epoch'], 'from', path, flush=True)
                break
            except (OSError, EOFError, RuntimeError): continue
        if not cfg.TRAIN.get('REPRO_RESUME_FROM'):
            raise RuntimeError('No complete resumable epoch checkpoint')
    if a.pretrained_dir:

        candidates = list(a.pretrained_dir.rglob('Transformer_best_val_loss.pt'))
        if len(candidates) != 1: raise RuntimeError('Expected exactly one pretrained best checkpoint')
        cfg.MODEL.PARAM.LSTTN['pretrained_path'] = str(candidates[0])
    launch_training(cfg, devices='0')
    best = list(a.attempt.rglob('*_best_val_loss.pt'))
    if len(best) != 1: raise RuntimeError('Missing/ambiguous best checkpoint')
    summary = dict(config=a.config, best=str(best[0].relative_to(a.attempt)),
                   global_batch=cfg.GLOBAL_BATCH_SIZE)
    if cfg.MODEL.NAME == 'LSTTN':
        from utils.prediction_export import evaluate_export
        exports = list(a.attempt.rglob('test_export_*/metadata.json'))
        if len(exports) != 1: raise RuntimeError('Missing/ambiguous test export')
        summary['metrics'] = evaluate_export(exports[0].parent)
        if set(summary['metrics']) != {3, 6, 12}: raise RuntimeError('Incomplete horizon metrics')
    (a.attempt / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
