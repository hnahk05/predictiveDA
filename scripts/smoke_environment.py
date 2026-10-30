"""Small CPU smoke, no real dataset/checkpoint and no GPU allocation."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_copy = ROOT / '.local_checks/environment-source'
    for name in ('models', 'runners', 'utils', 'configs'):
        shutil.copytree(ROOT/name, source_copy/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0, str(source_copy))
    import torch
    from models.transformer import Transformer
    from models.lsttn import DynamicGraphConv
    from utils.metrics import masked_mae, masked_rmse, masked_mape
    from easytorch import launch_training
    import runners.lsttn
    import runners.transformer
    assert sys.version_info[:2] == (3, 9), sys.version
    assert torch.__version__ == '1.8.1+cu111', torch.__version__
    assert metadata.version('easy-torch') == '1.3.2'
    torch.set_num_threads(2)
    model = Transformer(12, 1, 64, .1, 336, .75, 4)
    pred, target = model(torch.randn(1, 2, 1, 4032))
    loss = (pred-target).abs().mean()
    assert torch.isfinite(loss)
    loss.backward()
    conv = DynamicGraphConv(2, 1, 1, 0)
    x = torch.tensor([[[1.], [10.]]]); adjacency = torch.tensor([[0.,1.],[1.,0.]])
    assert torch.equal(conv.conv(x, adjacency), x)
    actual = [f(torch.tensor([11.,99.]), torch.tensor([10.,0.]), 0).item()
              for f in (masked_mae, masked_rmse, masked_mape)]
    assert abs(actual[0]-1) < 1e-6 and abs(actual[1]-1) < 1e-6 and abs(actual[2]-.1) < 1e-6
    report = dict(status='passed_cpu_smoke', platform=platform.platform(), python=sys.version,
                  source_copy=str(source_copy.relative_to(ROOT)),
                  packages={p:metadata.version(p) for p in ('torch','torchvision','easy-torch','numpy')},
                  cuda_available=torch.cuda.is_available(), loss=float(loss), gpu_ready=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
