"""Single-GPU epoch-boundary resume; trusted local checkpoints only."""
import random
import shutil
import datetime as dt
from pathlib import Path
import numpy as np
import torch

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def model_of(runner):
    return getattr(runner.model, 'module', runner.model)

class ReproductionMixin:
    def save_model(self, epoch):
        super().save_model(epoch)
        path = self.get_ckpt_path(epoch)
        state = torch.load(path, map_location='cpu')
        state['reproduction'] = dict(rng=rng_state(), scheduler=self.scheduler.state_dict() if self.scheduler else None,
            curriculum={k:getattr(self,k) for k in ('cl_len','warmup_steps','cl_steps','itera_per_epoch','init_lr') if hasattr(self,k)})
        temporary = str(path) + '.tmp'
        torch.save(state, temporary)
        Path(temporary).replace(path)

    def load_model_resume(self, strict=True):
        source = self.cfg.get('TRAIN.REPRO_RESUME_FROM')
        if source:
            state = torch.load(source, map_location='cpu')
            model_of(self).load_state_dict(state['model_state_dict'], strict=strict)
            self.optim.load_state_dict(state['optim_state_dict'])
            self.start_epoch = state['epoch']
            self.best_metrics = state.get('best_metrics', {})
            best = Path(source).parent / (self.model_name + '_best_val_loss.pt')
            if not best.exists(): raise RuntimeError('Resume requires previous best-val checkpoint')
            shutil.copy2(best, Path(self.ckpt_save_dir)/best.name)
        else:
            super().load_model_resume(strict)
            if not self.start_epoch: return
            state = torch.load(self.get_ckpt_path(self.start_epoch), map_location='cpu')
        extra = state.get('reproduction')
        if extra is None:
            raise RuntimeError('Legacy checkpoint lacks RNG/scheduler/curriculum; use a new run directory.')
        if self.scheduler and extra['scheduler'] is not None: self.scheduler.load_state_dict(extra['scheduler'])
        for k,v in extra['curriculum'].items(): setattr(self,k,v)
        restore_rng(extra['rng'])

    def on_epoch_end(self, epoch):
        super().on_epoch_end(epoch)
        stop_at = self.cfg.get('TRAIN.REPRO_STOP_AT')
        if stop_at and dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(stop_at):
            raise RuntimeError('LEASE_DEADLINE: saved epoch; stopping before next epoch')

    def on_validating_end(self, train_epoch):
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val_loss', greater_best=False)
        super().on_validating_end(train_epoch)
