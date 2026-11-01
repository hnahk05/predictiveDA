"""Local-only guards and checkpoint retention; author forecasting computation inherited."""
from pathlib import Path
import re
import torch
from runners.lsttn import LSTTNRunner


class LocalLSTTNRunner(LSTTNRunner):
    def train_iters(self, epoch, iter_index, data):
        loss = super().train_iters(epoch, iter_index, data)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite local loss; stopped before optimizer update')
        return loss

    def save_model(self, epoch):
        super().save_model(epoch)
        folder = Path(self.ckpt_save_dir).resolve()
        root = Path(__file__).resolve().parents[1]
        if root not in folder.parents:
            raise ValueError('Local checkpoint retention must stay inside project')
        keep = int(self.cfg.TRAIN.get('KEEP_LAST_EPOCHS', 2))
        if keep < 1:
            raise ValueError('Keep at least one resumable epoch')
        numbered = []
        for path in folder.glob('*.pt'):
            match = re.search(r'_(\d+)\.pt$', path.name)
            if match:
                numbered.append((int(match[1]), path))
        for _, path in sorted(numbered)[:-keep]:
            if path.parent.resolve() != folder:
                raise ValueError('Checkpoint path escapes retention folder')
            path.unlink()
