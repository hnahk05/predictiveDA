"""Small tests independent of train/test metrics; no backbone training."""
import unittest
import copy
import pickle
from types import SimpleNamespace
import numpy as np
import torch
from pipeline import Head, metric, Context, valid_mask, ROOT

class Tests(unittest.TestCase):
    def test_alpha_zero_and_zero_initialization(self):
        h=Head(16);x=torch.randn(2,170,28);base=torch.randn(2,12,170)
        self.assertTrue(torch.equal(h(x,base),base))
        with torch.no_grad():h.net[-1].weight.fill_(.2)
        self.assertTrue(torch.equal(h(x,base,0),base))

    def test_head_gradients_and_finite(self):
        h=Head(16);x=torch.randn(2,170,28);base=torch.randn(2,12,170)
        h(x,base).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))

    def test_metric_masks(self):
        r=metric(np.array([2.,4.,6.]),np.array([0.,2.,3.]))
        self.assertAlmostEqual(r['MAE'],7/3)
        self.assertAlmostEqual(r['RMSE'],np.sqrt(6.5))
        self.assertAlmostEqual(r['MAPE_percent'],100.)

    def test_features_do_not_accept_target(self):
        # Method consumes only observed short history, forecast, and index.
        short=torch.zeros(2,12,170,3);pred=np.ones((2,12,170),dtype='float32')*538.5
        idx=np.array([[0,12,24],[288,300,312]])
        x=Context.features(None,short,pred,idx)
        self.assertEqual(x.shape,(2,170,28))
        self.assertTrue(np.array_equal(x[:,:,:12],np.zeros((2,170,12),dtype='float32')))
        self.assertTrue(np.array_equal(x[:,:,12:24],short[...,0].numpy().transpose(0,2,1)))

    def test_adam_resume_matches_uninterrupted_step(self):
        torch.manual_seed(7);a=Head(16);oa=torch.optim.Adam(a.parameters(),lr=.001)
        x=torch.randn(2,170,28);base=torch.randn(2,12,170);y=torch.randn_like(base)
        def step(h,o):
            o.zero_grad();(h(x,base)-y).abs().mean().backward();o.step()
        step(a,oa);b=Head(16);b.load_state_dict(copy.deepcopy(a.state_dict()))
        ob=torch.optim.Adam(b.parameters(),lr=.001);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
        step(a,oa);step(b,ob)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_real_validation_selection_excludes_shared_labels(self):
        index={}
        for name in ('train','valid','test'):
            with (ROOT/'datasets/PEMS08'/(name+'_index.pkl')).open('rb') as f:index[name]=np.asarray(pickle.load(f))
        ctx=SimpleNamespace(ds={k:SimpleNamespace(index=v) for k,v in index.items()})
        mask=valid_mask(ctx,{'index':index['valid']})[:,:,0]
        times=index['valid'][:,1,None]+np.arange(12)
        self.assertTrue(np.all(times[mask]>index['train'][-1,2]-1))
        self.assertTrue(np.all(times[mask]<index['test'][0,1]))
        self.assertEqual(int((~mask).sum()),132)

if __name__=='__main__':unittest.main()
