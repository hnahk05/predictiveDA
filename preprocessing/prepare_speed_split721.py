"""Version METR-LA/PEMS-BAY only; no training and no legacy-file writes.

Retain chronological forecasting windows, purge H-1 starts at each boundary,
then allocate retained windows 70/20/10 (integer rounding). Target timestamps
are disjoint; past inputs may overlap. Paper specifies ratios, not this purge.
Raw speed is recovered by the stored affine inverse; no raw-HDF dependency.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'split721_target_disjoint_v1'
DATASETS = ('METR-LA', 'PEMS-BAY')


def read(path):
    with path.open('rb') as stream:
        return pickle.load(stream, encoding='latin1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plan(length, history=12, horizon=12):
    total = length - history - horizon + 1 - 2 * (horizon - 1)
    if total < 100:
        raise ValueError('Insufficient observations for this protocol')
    counts = [round(total * .7), round(total * .2)]
    counts.append(total - sum(counts))
    result = {}
    start = history
    for split, count in zip(('train', 'valid', 'test'), counts):
        t = np.arange(start, start + count, dtype=np.int64)
        result[split] = np.stack((t - history, t, t + horizon), axis=1)
        start += count + horizon - 1
    validate(result, length)
    return result


def validate(indices, length):
    total = sum(map(len, indices.values()))
    previous = None
    for split, ratio in zip(('train', 'valid', 'test'), (.7, .2, .1)):
        a = np.asarray(indices[split])
        assert a.ndim == 2 and a.shape[1] == 3
        assert np.all(a[:, 1] - a[:, 0] == 12)
        assert np.all(a[:, 2] - a[:, 1] == 12)
        assert np.all(np.diff(a[:, 1]) == 1)
        assert a[0, 0] >= 0 and a[-1, 2] <= length
        assert abs(len(a) - total * ratio) <= 1
        if previous is not None:
            assert previous[-1, 2] <= a[0, 1], 'Target leakage'
        previous = a
    assert indices['test'][-1, 2] == length


def summary(indices):
    return {s: {'windows': len(a), 'first_index': a[0].tolist(),
                'last_index': a[-1].tolist()} for s, a in indices.items()}


def prepare(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    source = ROOT / 'datasets' / dataset
    dest = ROOT / 'datasets' / VERSION / dataset
    if dest.exists():
        raise FileExistsError('Refusing to overwrite existing version: ' + str(dest))
    protected = {p.name: sha(p) for p in source.glob('*.pkl')}
    data = read(source / 'data.pkl')
    assert data.ndim == 3 and data.shape[2] == 3 and np.isfinite(data).all()
    old = {s: np.asarray(read(source / (s + '_index.pkl')))
           for s in ('train', 'valid', 'test')}
    joined = np.concatenate(list(old.values()))
    assert np.all(np.diff(joined[:, 1]) == 1)
    assert len(joined) == len(data) - 23
    old_mean, old_std = read(source / 'mean.pkl'), read(source / 'std.pkl')
    speed = data[..., 0].astype(np.float64) * old_std + old_mean
    indices = plan(len(data))
    fit_end = int(indices['train'][-1, 2])
    mean, std = float(speed[:fit_end].mean()), float(speed[:fit_end].std())
    assert std > 0 and np.isfinite([mean, std]).all()
    result = data.copy()
    result[..., 0] = (speed - mean) / std
    assert np.allclose(result[..., 0] * std + mean, speed, atol=1e-5)
    assert np.array_equal(result[..., 1:], data[..., 1:])
    assert abs(result[:fit_end, :, 0].mean()) < 1e-6
    assert abs(result[:fit_end, :, 0].std() - 1) < 1e-6
    old_train_input_end = int(old['train'][-1, 1])
    report = dict(dataset=dataset, version=VERSION, shape=list(data.shape),
        before=summary(old), after=summary(indices),
        retained_windows=sum(map(len, indices.values())), purged_windows=22,
        split_method='chronological sliding windows; H-1 purged starts at each boundary; round retained counts 70/20/10',
        paper_difference='paper specifies 7:2:1; purge is an explicit leakage-prevention addition',
        scaler=dict(type='StandardScaler/global speed', mean=mean, std=std,
                    fit_timestamp_range_half_open=[0, fit_end],
                    source='inverse affine of legacy data.pkl using legacy mean/std; original HDF not read'),
        legacy_scaler=dict(mean=float(old_mean), std=float(old_std),
            matches_train_input_prefix=bool(np.allclose(
                [speed[:old_train_input_end].mean(), speed[:old_train_input_end].std()],
                [old_mean, old_std], atol=1e-10, rtol=0)),
            full_series_mean=float(speed.mean()), full_series_std=float(speed.std()),
            provenance='empirical match only; original preprocessing code not available'),
        target_overlap_before={a+'_'+b: max(0,int(old[a][-1,2]-old[b][0,1]))
                              for a,b in [('train','valid'),('valid','test')]},
        target_overlap_after={'train_valid':0,'valid_test':0,'train_test':0},
        source_sha256=protected,
        checkpoint_policy='Old checkpoints are legacy-protocol artifacts, not validated new-protocol models. No training performed.')
    dest.mkdir(parents=True, exist_ok=False)
    values = {'data.pkl': result, 'mean.pkl': mean, 'std.pkl': std}
    values.update({s+'_index.pkl': a.tolist() for s,a in indices.items()})
    for name,value in values.items():
        with (dest/name).open('xb') as stream:
            pickle.dump(value, stream, protocol=4)
    reloaded = {s: np.asarray(read(dest/(s+'_index.pkl'))) for s in indices}
    validate(reloaded, len(read(dest/'data.pkl')))
    assert protected == {p.name:sha(p) for p in source.glob('*.pkl')}
    report['legacy_files_unchanged'] = True
    report['output_sha256'] = {p.name:sha(p) for p in dest.glob('*.pkl')}
    (dest/'split_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    prepare(parser.parse_args().dataset)
