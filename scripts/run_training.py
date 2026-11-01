"""One isolated training attempt, called by run_experiments.py."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_pretrained(cfg):
    if cfg.MODEL.NAME != 'LSTTN' or cfg.MODEL.PARAM.LSTTN.get('ablation') == 'st_only':
        return None
    import torch
    from models.transformer import Transformer
    aliases = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY'}
    dataset = aliases.get(cfg.DATASET_NAME, cfg.DATASET_NAME)
    source = Path(cfg.MODEL.PARAM.LSTTN.get('pretrained_path') or ROOT / 'pretrained_transformers' / (dataset + '.pt'))
    state = torch.load(source, map_location='cpu')
    actual = state['model_state_dict']['patch.input_embedding.weight'].shape[0]
    expected = cfg.MODEL.PARAM.TRANSFORMER['out_channel']
    if actual != expected:
        raise ValueError('Checkpoint width %s != config width %s. Choose the matching config or pretrain matching weights; no partial loading.' % (actual, expected))
    cpu_rng = torch.get_rng_state()
    try:
        encoder = Transformer(mode='inference', **cfg.MODEL.PARAM.TRANSFORMER)
        encoder.load_state_dict(state['model_state_dict'], strict=True)
    finally:
        torch.set_rng_state(cpu_rng)
    return dict(path=str(source), width=actual, strict_load=True,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--attempt', required=True, type=Path)
    p.add_argument('--pretrained-dir', type=Path)
    p.add_argument('--resume-dir', type=Path)
    p.add_argument('--stop-at')
    p.add_argument('--check-config', action='store_true', help='CPU strict-load and report config; do not train')
    a = p.parse_args()
    from utils.cuda_softmax_compat import activate_from_env
    softmax_compat = activate_from_env()
    from easytorch import launch_training
    cfg = runpy.run_path(a.config)['CFG']
    configured_gpu_num = cfg.GPU_NUM
    cfg.GPU_NUM = 1
    cfg.USE_GPU = True
    cfg.GLOBAL_BATCH_SIZE = cfg.TRAIN.DATA.BATCH_SIZE
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
    pretrained_info = verify_pretrained(cfg)
    runtime_info = dict(config=a.config, configured_gpu_num=configured_gpu_num,
                        effective_gpu_num=1, global_batch=cfg.GLOBAL_BATCH_SIZE,
                        epochs=cfg.TRAIN.NUM_EPOCHS, softmax_compat=softmax_compat,
                        pretrained=pretrained_info)
    if cfg.MODEL.NAME == 'LSTTN':
        runtime_info['width'] = cfg.MODEL.PARAM.TRANSFORMER['out_channel']
        runtime_info['hidden'] = [cfg.MODEL.PARAM.LSTTN['long_trend_hidden_dim'],
                                  cfg.MODEL.PARAM.LSTTN['seasonality_hidden_dim']]
    if cfg.get('EXPERIMENT'):
        runtime_info['experiment'] = cfg.EXPERIMENT
    print('RUN_CONFIG', json.dumps(runtime_info), flush=True)
    if a.check_config:
        return
    launch_training(cfg, devices='0')
    best = list(a.attempt.rglob('*_best_val_loss.pt'))
    if len(best) != 1: raise RuntimeError('Missing/ambiguous best checkpoint')
    summary = dict(runtime_info, best=str(best[0].relative_to(a.attempt)))
    if cfg.MODEL.NAME == 'LSTTN':
        from utils.prediction_export import evaluate_export
        exports = list(a.attempt.rglob('test_export_*/metadata.json'))
        if len(exports) != 1: raise RuntimeError('Missing/ambiguous test export')
        summary['metrics'] = evaluate_export(exports[0].parent)
        if set(summary['metrics']) != {3, 6, 12}: raise RuntimeError('Incomplete horizon metrics')
    (a.attempt / 'completed.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
