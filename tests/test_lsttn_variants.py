"""Small CPU unit tests; optional real-checkpoint CUDA smoke, no training runs."""
import copy
import json
import runpy
import subprocess
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from models.lsttn import DynamicGraphConv, LSTTN
from utils.parameter_audit import audit_parameters
torch.set_num_threads(1)


def fixture(ablation='full', mode='mask_only', policy='baseline'):
    cfg = runpy.run_path(str(ROOT/'configs/PEMS08/forecasting.py'))['CFG']
    p = copy.deepcopy(cfg.MODEL.PARAM)
    p.LSTTN.update(num_nodes=3, supports=[torch.eye(3),torch.eye(3)],
                   ablation=ablation, ablation_mode=mode, transformer_policy=policy)
    p.STGNN.GWNET.update(num_nodes=3,supports=p.LSTTN['supports'])
    p.TRANSFORMER.update(out_channel=8,num_encoder_layers=1)
    def frozen(model):
        for value in model.transformer.parameters():
            value.requires_grad=False
    with patch.object(LSTTN,'load_pretrained_transformer', frozen), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=LSTTN(cfg,**p)
    return model, (torch.randn(1,12,3,3),torch.randn(1,4032,3,3))


class VariantTests(unittest.TestCase):
    def test_graph_direction_shape_and_gradient(self):
        layer=DynamicGraphConv(3,2,2,0,graph_mode='corrected').double()
        x=torch.tensor([[[1.,2.],[10.,20.],[100.,200.]]],requires_grad=True,dtype=torch.double)
        a=torch.tensor([[0.,2.,0.],[0.,0.,3.],[4.,0.,0.]],requires_grad=True,dtype=torch.double)
        expected=torch.tensor([[[400.,800.],[2.,4.],[30.,60.]]],dtype=torch.double)
        self.assertTrue(torch.equal(layer.conv(x,a),expected))
        self.assertTrue(torch.autograd.gradcheck(layer.conv,(x,a)))
        from models.graph_wavenet import nconv
        control=nconv()(x.transpose(1,2).unsqueeze(-1),a).squeeze(-1).transpose(1,2)
        self.assertTrue(torch.equal(control,expected))
        layer.graph_mode='baseline'
        self.assertTrue(torch.equal(layer.conv(x,a),x*a.sum(1)[None,:,None]))
        layer.graph_mode='author'
        self.assertTrue(torch.equal(layer.conv(x,a),x*a.sum(1)[None,:,None]))
        with self.assertRaises(ValueError): DynamicGraphConv(3,2,2,0,graph_mode='wrong')

    def test_ablation_order_calls_gradients(self):
        for ablation in ('full','wo_lt','wo_p','wo_st','st_only'):
            masked,inputs=fixture(ablation,'mask_only'); skipped,_=fixture(ablation,'skip_compute')
            skipped.load_state_dict(masked.state_dict(),strict=True)
            masked.eval();skipped.eval()
            calls={}; fusion=[];handles=[]
            masked_calls=set()
            masked_handles=[m.register_forward_hook(lambda m,i,o,n=n:masked_calls.add(n))
                            for n,m in masked.named_children()]
            for name,module in skipped.named_children():
                handles.append(module.register_forward_hook(lambda m,i,o,n=name:calls.__setitem__(n,calls.get(n,0)+1)))
            handles.append(skipped.mlp.register_forward_pre_hook(lambda m,i:fusion.append(i[0].detach().clone())))
            actual=skipped(*inputs);reference=masked(*inputs)
            self.assertEqual(tuple(actual.shape),(1,3,12));self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_allclose(actual,reference,atol=1e-6,rtol=1e-5)
            if ablation in ('wo_lt','st_only'):self.assertNotIn('long_term_trend_extractor',calls)
            if ablation in ('wo_p','st_only'):
                self.assertNotIn('weekly_seasonality_extractor',calls);self.assertNotIn('daily_seasonality_extractor',calls)
            if ablation=='wo_st':
                self.assertNotIn('short_term_trend_extractor',calls)
                self.assertEqual(torch.count_nonzero(fusion[0][...,:128]).item(),0)
            if ablation=='st_only':
                self.assertNotIn('transformer',calls);self.assertNotIn('trend_seasonality_mlp',calls)
                self.assertEqual(torch.count_nonzero(fusion[0][...,128:]).item(),0)
            actual.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in skipped.mlp.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in skipped.parameters() if p.grad is not None))
            for name in ('transformer','long_term_trend_extractor','weekly_seasonality_extractor',
                         'daily_seasonality_extractor','short_term_trend_extractor'):
                self.assertIn(name,masked_calls)
                if name not in calls:
                    self.assertTrue(all(p.grad is None for p in getattr(skipped,name).parameters()))
            for handle in handles:handle.remove()
            for handle in masked_handles:handle.remove()

    def test_full_matches_legacy(self):
        model,inputs=fixture();model.eval()
        modern=model(*inputs);model.ablation_mode='legacy';legacy=model(*inputs)
        self.assertTrue(torch.equal(modern,legacy))

    def test_transformer_policy_cache_and_invalidation(self):
        model,inputs=fixture(policy='baseline');model.train()
        self.assertTrue(model.transformer.training)
        x=inputs[1][...,[0]].permute(0,2,3,1)
        first=model.encode_long(x);second=model.encode_long(x)
        self.assertFalse(torch.equal(first,second))
        with self.assertRaises(ValueError):model.encode_long(x,{})
        model.transformer_policy='fixed_eval';model.train()
        self.assertFalse(model.transformer.training)
        self.assertTrue(model.short_term_trend_extractor.training)
        model.eval(); cache={}
        with torch.no_grad():
            direct=model(*inputs);cached=model(*inputs,transformer_cache=cache)
            repeated=model(*inputs,transformer_cache=cache)
        torch.testing.assert_allclose(direct,cached,atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(cached,repeated))
        key=cache['key'];model.encode_long(x+1,cache);self.assertNotEqual(key,cache['key'])
        key=cache['key']
        with torch.no_grad():next(model.transformer.parameters()).add_(.01)
        model.encode_long(x+1,cache);self.assertNotEqual(key,cache['key'])
        with self.assertRaises(ValueError):model.encode_long(x.requires_grad_(),cache)

    def test_parameter_audit_restores_and_distinguishes(self):
        model,inputs=fixture();model.eval()
        before={n:p.clone() for n,p in model.state_dict().items()}
        result=audit_parameters(model,inputs,{'TYPE':'Adam','PARAM':{'lr':.001}})
        self.assertTrue(result['state_restored_exactly'])
        self.assertTrue(result['actually_changed_in_isolated_step']['numel']>0)
        for n in result['backward_with_gradient']['names']:
            self.assertFalse(n.startswith('transformer'))
        for n in result['forward_observed_owner_parameters']['names']:
            self.assertFalse(n.startswith('transformer1.'))
            self.assertFalse(n.startswith('transformer.decoder.'))
        self.assertTrue(all(torch.equal(model.state_dict()[n],v) for n,v in before.items()))


def real_smoke():
    # Compare unchanged default behavior against repository HEAD before this edit.
    source=subprocess.check_output(['git','show','HEAD:models/lsttn.py'],cwd=ROOT,text=True,encoding='utf-8')
    old=types.ModuleType('baseline_reference');exec(compile(source,'baseline_reference','exec'),old.__dict__)
    cfg=runpy.run_path(str(ROOT/'configs/PEMS08/forecasting.py'))['CFG']
    meta=json.loads((ROOT/'results/extension/inference_cost.json').read_text())
    checkpoint=ROOT/meta['checkpoint']['path']
    state=torch.load(str(checkpoint),map_location='cpu')['model_state_dict']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    new=LSTTN(cfg,**cfg.MODEL.PARAM).eval().to(device)
    reference=old.LSTTN(cfg,**cfg.MODEL.PARAM).eval().to(device)
    new.load_state_dict(state,strict=True);reference.load_state_dict(state,strict=True)
    from utils.datasets import MTSDataset
    ds=MTSDataset(str(ROOT/'datasets/PEMS08/data.pkl'),str(ROOT/'datasets/PEMS08/test_index.pkl'),4032)
    _,short,long=ds[0];inputs=(short.unsqueeze(0).to(device),long.unsqueeze(0).to(device))
    with torch.no_grad():
        a=reference(*inputs);b=new(*inputs)
        new.transformer_policy='fixed_eval';new.eval();cache={}
        c=new(*inputs,transformer_cache=cache);d=new(*inputs,transformer_cache=cache)
    torch.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    torch.testing.assert_allclose(b,c,atol=1e-6,rtol=1e-5)
    assert torch.equal(c,d) and torch.isfinite(b).all()
    print(json.dumps({'real_checkpoint_smoke':'PASS','device':device,
        'baseline_max_abs':float((a-b).abs().max()),'cache_max_abs':float((b-c).abs().max()),
        'cache_max_relative':float(((b-c).abs()/b.abs().clamp_min(1e-8)).max()),
        'atol':1e-6,'rtol':1e-5,'strict_load':True}))


if __name__=='__main__':
    if '--real-smoke' in sys.argv:real_smoke()
    else:unittest.main()
