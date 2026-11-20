"""CPU-only regression checks using the actual Dataset and EasyTorch DataLoader.
Does not construct a model, invoke a runner constructor, or train/infer.
"""
import copy
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from easytorch.core.data_loader import build_data_loader
from runners.lsttn import LSTTNRunner
from runners.speed_split721 import scaler_args
from preprocessing.prepare_speed_split721 import plan, validate, read, sha


def verify(dataset):
    cfg = runpy.run_path(str(ROOT/'configs/split721'/(dataset+'.py')))['CFG']
    old_cfg = runpy.run_path(str(ROOT/'configs/actual'/(dataset+'.py')))['CFG']
    # Architecture, optimizer, schedule, batch and epoch budget must not change.
    assert str(cfg.MODEL) == str(old_cfg.MODEL)
    for key in ('OPTIM','LR_SCHEDULER','NUM_EPOCHS','WARMUP_EPOCHS','CL_EPOCHS'):
        assert cfg.TRAIN[key] == old_cfg.TRAIN[key]
    args = scaler_args(cfg)
    directory = Path(cfg.TRAIN.DATA.DIR)
    manifest = json.loads((directory/'split_manifest.json').read_text())
    for name,h in manifest['source_sha256'].items():
        assert sha(ROOT/'datasets'/dataset/name) == h
    for name,h in manifest['output_sha256'].items():
        assert sha(directory/name) == h
    indices = {s:np.asarray(read(directory/(s+'_index.pkl'))) for s in ('train','valid','test')}
    validate(indices, manifest['shape'][0])
    info = {}
    for split, section in [('train','TRAIN'),('valid','VAL'),('test','TEST')]:
        if split == 'train':
            ds = LSTTNRunner.build_train_dataset(SimpleNamespace(), cfg)
        elif split == 'valid':
            ds = LSTTNRunner.build_val_dataset(cfg)
        else:
            ds = LSTTNRunner.build_test_dataset(cfg)
        assert np.array_equal(ds.index, indices[split])
        loader_cfg = copy.deepcopy(cfg[section].DATA)
        # Only inspection worker settings change; no training configuration changed.
        loader_cfg.NUM_WORKERS = 0
        loader_cfg.PREFETCH = False
        loader = build_data_loader(ds, loader_cfg)
        assert len(loader.dataset) == len(indices[split])
        y, short, long = next(iter(loader))
        nodes = manifest['shape'][1]
        assert tuple(y.shape[1:]) == (12,nodes,3)
        assert tuple(short.shape[1:]) == (12,nodes,3)
        assert tuple(long.shape[1:]) == (4032,nodes,3)
        for i in [0,len(ds)-1]:
            target,xs,xl = ds[i]
            a,b,c = ds.index[i]
            assert a < b < c
            assert np.array_equal(target.numpy(), ds.data[b:c].numpy())
            assert np.array_equal(xs.numpy(), ds.data[a:b].numpy())
            if b >= 4032:
                assert np.array_equal(xl.numpy(), ds.data[b-4032:b].numpy())
            else:
                assert not xl.any()
        info[split] = {'samples':len(ds), 'batches':len(loader),
                       'sampler':type(loader.sampler).__name__}
        del loader, ds, y, short, long
    values = read(directory/'data.pkl')
    end = manifest['scaler']['fit_timestamp_range_half_open'][1]
    assert abs(values[:end,:,0].mean()) < 1e-6
    assert abs(values[:end,:,0].std()-1) < 1e-6
    assert args == {'mean':manifest['scaler']['mean'], 'std':manifest['scaler']['std']}
    # A deliberate cross-boundary target overlap must be rejected.
    bad = {s:a.copy() for s,a in indices.items()}
    bad['valid'] -= 11
    try:
        validate(bad,len(values))
    except AssertionError:
        pass
    else:
        raise AssertionError('Leakage guard failed')
    print(json.dumps({'dataset':dataset,'passed':True,'loaders':info,'scaler':args}))


if __name__ == '__main__':
    for length in (1000,34272,52116):
        validate(plan(length),length)
    for dataset in ('METR-LA','PEMS-BAY'):
        verify(dataset)
