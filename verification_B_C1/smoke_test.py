"""Read-only baseline inference; writes only a new verification JSON (never overwrites)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def read(path):
    with open(path, 'rb') as f: return pickle.load(f, encoding='latin1')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('smoke_results.json'))
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists; choose a new --output path.')
    result = {'status':'BLOCKED', 'tolerance':{'atol':1e-5,'rtol':1e-4,
        'reason':'FP32 CPU inverse scaling and GPU kernel/batch-size rounding; fixed before inference.'}}
    protected = {}
    try:
        os.chdir(ROOT)
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset
        from utils.datasets import MTSDataset
        from utils.load_data import max_min_scale
        from models.lsttn import LSTTN, DynamicGraphConv
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        result['environment'] = dict(python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            device=device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            platform=platform.platform())
        meta=json.loads((ROOT/'results/extension/inference_cost.json').read_text())
        ck=ROOT/meta['checkpoint']['path']; pre=ROOT/'pretrained_transformers/PEMS08.pt'
        data=ROOT/'datasets/PEMS08'; exports=ROOT/'artifacts/report-inputs/PEMS08'
        paths=[ck,pre,data/'data.pkl',data/'min.pkl',data/'max.pkl',ROOT/'datasets/sensor_graph/adj_mx_08.pkl']
        paths += [data/(s+'_index.pkl') for s in ('train','valid','test')]
        protected={str(p):sha(p) for p in paths}
        result['checkpoint']={'path':str(ck),'sha256':sha(ck),'matches_recorded_hash':sha(ck)==meta['checkpoint']['sha256']}
        result['pretrained']={'path':str(pre),'sha256':sha(pre)}
        indices={s:np.asarray(read(data/(s+'_index.pkl')),dtype=np.int64) for s in ('train','valid','test')}
        stats={'min':read(data/'min.pkl'),'max':read(data/'max.pkl')}
        ds=MTSDataset(str(data/'data.pkl'),str(data/'test_index.pkl'),4032)
        audit={s:dict(count=len(v),first=v[0].tolist(),last=v[-1].tolist(),
                     chronological=bool(np.all(np.diff(v[:,1])==1))) for s,v in indices.items()}
        sets={s:set((v[:,1,None]+np.arange(12)).ravel().tolist()) for s,v in indices.items()}
        audit['label_overlap']={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','valid'),('valid','test'),('train','test')]}
        audit['data_shape']=list(ds.data.shape)
        audit['scaler']={s:float(stats[s][0,0,0]) for s in stats}
        rawpath=data/'raw/pems08.npz'
        if rawpath.exists():
            with np.load(rawpath) as z: raw=z['data']
            audit['raw_shape']=list(raw.shape)
            audit['raw_flow_extrema']={'all':[float(raw[...,0].min()),float(raw[...,0].max())],
                'train_through_last_label':[float(raw[:indices['train'][-1,2],:,0].min()),float(raw[:indices['train'][-1,2],:,0].max())]}
            reconstructed=max_min_scale(ds.data[...,0],**stats).numpy()
            audit['inverse_flow_raw_max_abs']=float(np.max(np.abs(reconstructed-raw[...,0])))
        t=np.arange(len(ds.data))
        audit['time_channels_match']=[bool(np.allclose(ds.data[:,:,1].numpy(),(t%288/288)[:,None])),
                                     bool(np.allclose(ds.data[:,:,2].numpy(),(t//288%7)[:,None]))]
        result['data_audit']=audit
        chosen=[0,31,32,len(ds)-1]
        wanted={tuple(indices['test'][i]):i for i in chosen}; refs={}
        for path in sorted(exports.glob('batch_*.npz')):
            with np.load(path) as z:
                for row, idx in enumerate(z['index']):
                    key=tuple(idx)
                    if key in wanted:
                        sid=wanted[key]
                        if sid in refs: raise ValueError('Duplicate export sample')
                        refs[sid]=(z['prediction'][row].copy(),z['target'][row].copy(),z['mask'][row].copy(),str(path),row)
                        protected[str(path)]=sha(path)
        if len(refs)!=4: raise FileNotFoundError('Cannot match all selected sample indices to exports')
        cfg=runpy.run_path(str(ROOT/'configs/PEMS08/forecasting.py'))['CFG']
        cfg.MODEL.PARAM.LSTTN['pretrained_path']=str(pre)
        checkpoint=torch.load(str(ck),map_location='cpu')
        result['checkpoint']['epoch']=int(checkpoint['epoch'])
        if result['checkpoint']['epoch']!=68: raise ValueError('Not epoch 68')
        model=LSTTN(cfg,**cfg.MODEL.PARAM)
        keys=model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        result['checkpoint'].update(strict_load=True,missing_keys=list(keys.missing_keys),unexpected_keys=list(keys.unexpected_keys))
        pretrained=torch.load(str(pre),map_location='cpu')['model_state_dict']
        current=model.transformer.state_dict()
        result['pretrained']['matches_forecasting_transformer']=all(torch.equal(current[k],v) for k,v in pretrained.items())
        model=model.to(device).eval()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        result['eval_all_modules']=all(not m.training for m in model.modules())
        result['model_modes']={'ablation':model.ablation,'ablation_mode':model.ablation_mode,'transformer_policy':model.transformer_policy,
                               'graph_modes':sorted({m.graph_mode for m in model.modules() if isinstance(m,DynamicGraphConv)}),'cache':False,'amp':False}
        result['samples']=[]
        # Batch size 1 avoids exceeding 6 GB; four independent real windows.
        loader=DataLoader(Subset(ds,chosen),batch_size=1,shuffle=False,num_workers=0)
        for sid,(y,short,long) in zip(chosen,loader):
            with torch.no_grad(): output=model(short.to(device),long.to(device)).detach().cpu()
            pred=max_min_scale(output.transpose(1,2),**stats).numpy()[0]
            target=max_min_scale(y[...,0],**stats).numpy()[0]
            reference,old_target,old_mask,path,row=refs[sid]
            delta=np.abs(pred.astype(np.float64)-reference.astype(np.float64))
            over=delta>(1e-5+1e-4*np.abs(reference.astype(np.float64)))
            finite=all(bool(torch.isfinite(v).all()) for v in (short,long,y,output)) and bool(np.isfinite(pred).all())
            result['samples'].append(dict(test_position=sid,index=indices['test'][sid].tolist(),export=path,export_row=row,
                shapes={'short_x':list(short.shape),'long_x':list(long.shape),'y':list(y.shape),'model_output':list(output.shape),'prediction':list(pred.shape)},
                finite=finite,target_exact=bool(np.array_equal(target,old_target)),mask_exact=bool(np.array_equal(np.isfinite(target)&(target!=0),old_mask)),
                max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean()),elements=int(delta.size),
                above_tolerance=int(over.sum()),export_dtype=str(reference.dtype),
                violations=[dict(horizon_step=int(h+1),sensor_position=int(n),prediction=float(pred[h,n]),
                    reference=float(reference[h,n]),absolute_difference=float(delta[h,n]),
                    allowed_difference=float(1e-5+1e-4*abs(float(reference[h,n])))) for h,n in np.argwhere(over)[:100]]))
        result['comparison_summary']={'elements':sum(s['elements'] for s in result['samples']),
            'above_tolerance':sum(s['above_tolerance'] for s in result['samples']),
            'max_abs_difference':max(s['max_abs_difference'] for s in result['samples']),
            'mean_abs_difference':sum(s['mean_abs_difference']*s['elements'] for s in result['samples'])/sum(s['elements'] for s in result['samples'])}
        result['no_parameter_gradients']=all(p.grad is None for p in model.parameters())
        x=torch.tensor([[[1.],[10.]]]); adj=torch.tensor([[0.,1.],[1.,0.]])
        graph=DynamicGraphConv(2,1,1,0)
        result['graph_toy']={'author':graph.conv(x,adj).flatten().tolist(),'neighbor_aggregation':torch.einsum('bnh,nm->bmh',x,adj).flatten().tolist()}
        result['status']='PASS' if all(s['finite'] and s['target_exact'] and s['mask_exact'] and s['above_tolerance']==0 and s['shapes']['model_output']==[1,170,12] for s in result['samples']) and result['eval_all_modules'] and result['checkpoint']['matches_recorded_hash'] and result['pretrained']['matches_forecasting_transformer'] else 'FAIL'
    except FileNotFoundError as exc:
        result.update(status='BLOCKED',error=str(exc))
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['protected_files_sha256']=protected
        result['protected_files_unchanged']=all(Path(p).exists() and sha(p)==h for p,h in protected.items())
        if not result['protected_files_unchanged']: result['status']='FAIL'
        with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': sys.exit(main())
