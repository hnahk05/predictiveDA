"""Opt-in scaler routing for versioned speed data; legacy runners unchanged."""
from pathlib import Path
from runners.local_lsttn import LocalLSTTNRunner
from utils.log import load_pkl
from utils.load_data import standard_scale


def scaler_args(cfg):
    if cfg['DATASET_NAME'] not in ('METR-LA', 'PEMS-BAY'):
        raise ValueError('Versioned speed runner is restricted to METR-LA/PEMS-BAY')
    directory = Path(cfg['TRAIN']['DATA']['DIR'])
    for split in ('VAL', 'TEST'):
        if Path(cfg[split]['DATA']['DIR']).resolve() != directory.resolve():
            raise ValueError('All splits must share the same versioned scaler')
    return {'mean': load_pkl(str(directory/'mean.pkl')),
            'std': load_pkl(str(directory/'std.pkl'))}


class SpeedSplit721Runner(LocalLSTTNRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Parent initializes legacy scaler; replace before any data/model iteration.
        self.scaler = standard_scale
        self.scaler_args = scaler_args(cfg)
