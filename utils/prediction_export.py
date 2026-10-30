"""CPU batch exports. Axes: sample, horizon, node. Values in original units."""
import json
from pathlib import Path
import numpy as np
import torch
from utils.metrics import all_metrics

def export_batches(runner, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    offset = 0
    for batch, data in enumerate(runner.test_data_loader):
        pred, target = runner.test_iters(batch, data)
        pred, target = pred.detach().cpu(), target.detach().cpu()[..., 0]
        pred = runner.scaler(pred, **runner.scaler_args)
        target = runner.scaler(target, **runner.scaler_args)
        count = len(pred)
        index = np.asarray(runner.test_data_loader.dataset.index[offset:offset+count])
        mask = torch.isfinite(target) & (target != 0)
        np.savez_compressed(directory / f"batch_{batch:06d}.npz", prediction=pred.numpy(), target=target.numpy(),
                            mask=mask.numpy(), index=index)
        offset += count
    (directory / 'metadata.json').write_text(json.dumps(dict(dataset=runner.dataset_name, axes=['sample','horizon','node'],
        units='original', mape='ratio; multiply by 100 for percent', null_value=0,
        mape_rule='author: labels below 0.01 become zero', mae_rule='unmasked for PEMS04/PEMS08; masked zero otherwise', samples=offset)), encoding='utf-8')

def evaluate_export(directory):
    # Accumulate exact global masked sums; no GPU or full-dataset concatenation.
    metadata = json.loads((Path(directory)/'metadata.json').read_text()) if (Path(directory)/'metadata.json').exists() else {}
    unmasked_mae = metadata.get('dataset') in ('PEMS04','PEMS08')
    stats = {h:np.zeros(7, dtype=np.float64) for h in (3,6,12)}
    for path in sorted(Path(directory).glob('batch_*.npz')):
        with np.load(path) as data:
            for h in stats:
                p,y = data['prediction'][:,h-1].astype('float64'),data['target'][:,h-1].astype('float64')
                valid = data['mask'][:,h-1]
                if not np.isfinite(p[valid]).all(): raise ValueError('Nonfinite predictions')
                e = p[valid]-y[valid]
                mv = valid & (y >= .01)
                mae_valid = np.isfinite(y) if unmasked_mae else valid
                if not np.isfinite(p[mae_valid]).all(): raise ValueError('Nonfinite predictions')
                stats[h] += [np.abs(p[mae_valid]-y[mae_valid]).sum(),np.square(e).sum(),valid.sum(),
                             (np.abs(p[mv]-y[mv])/y[mv]).sum(),mv.sum(),mae_valid.sum(),0]
    return {h:dict(MAE=float(v[0]/v[5]),RMSE=float(np.sqrt(v[1]/v[2])),MAPE_ratio=float(v[3]/v[4]))
            for h,v in stats.items() if v[2] and v[4]}

if __name__ == '__main__':
    import sys
    print(json.dumps(evaluate_export(sys.argv[1]), indent=2))
