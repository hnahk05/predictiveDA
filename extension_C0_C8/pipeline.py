"""Frozen LSTTN residual experiment. All new artifacts live under run_v2/.

Commands: verify, pilot, cache, train, evaluate, xai, demo.
Existing completed shards/runs are validated and reused, never silently overwritten.
"""
import argparse, csv, hashlib, inspect, json, os, pickle, platform, runpy, subprocess, sys, time
from pathlib import Path
import threading
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
os.chdir(ROOT)
import numpy as np
import torch
import psutil
from torch import nn
from utils.datasets import MTSDataset
from utils.load_data import max_min_scale
from models.lsttn import LSTTN

HERE=ROOT/'extension_C0_C8'; RUN=HERE/'run_v2'
P=json.loads((HERE/'protocol_v2.json').read_text())
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load(p):
    with open(p,'rb') as f:return pickle.load(f,encoding='latin1')

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    temp.replace(path)

def table(path,rows):
    if Path(path).exists():raise FileExistsError(str(path))
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def info():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda=torch.version.cuda,device=DEVICE,gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None)

class Monitor:
    def __enter__(self):
        self.stop=False;self.peak=0;self.start=time.perf_counter()
        if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
        def loop():
            while not self.stop:
                self.peak=max(self.peak,psutil.Process().memory_info().rss);time.sleep(.01)
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop=True;self.thread.join()
        self.result=dict(seconds=time.perf_counter()-self.start,process_rss_sampled_peak_bytes=self.peak,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if DEVICE=='cuda' else 0)

def metric(pred,y,mask=None):
    pred=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if mask is None:mask=np.ones(y.shape,dtype=bool)
    mask=np.broadcast_to(mask,y.shape)
    assert np.isfinite(pred).all() and np.isfinite(y).all() and mask.any()
    e=pred-y;rm=mask&(y!=0);mp=mask&(y>=.01)
    return dict(MAE=float(np.abs(e[mask]).mean()),RMSE=float(np.sqrt((e[rm]**2).mean())),
                MAPE_percent=float(100*np.mean(np.abs(e[mp])/y[mp])))

def horizons(pred,y,mask=None):
    return {str(h):metric(pred[:,h-1],y[:,h-1],None if mask is None else mask[:,h-1]) for h in (3,6,12)}

class Context:
    def __init__(self):
        self.data=ROOT/'datasets/PEMS08'
        self.ds={s:MTSDataset(str(self.data/'data.pkl'),str(self.data/(s+'_index.pkl')),4032) for s in ('train','valid','test')}
        self.stats={s:load(self.data/(s+'.pkl')) for s in ('min','max')}
        meta=json.loads((ROOT/'results/extension/inference_cost.json').read_text())
        self.ck=ROOT/meta['checkpoint']['path'];self.pre=ROOT/'pretrained_transformers/PEMS08.pt'
        assert sha(self.ck)==meta['checkpoint']['sha256']
        self.hashes={str(p.relative_to(ROOT)):sha(p) for p in [self.ck,self.pre,self.data/'data.pkl',self.data/'min.pkl',self.data/'max.pkl',
           *[self.data/(s+'_index.pkl') for s in ('train','valid','test')],ROOT/'datasets/sensor_graph/adj_mx_08.pkl',
           ROOT/'models/lsttn.py',ROOT/'models/transformer.py',ROOT/'models/graph_wavenet.py',ROOT/'utils/datasets.py',
           ROOT/'configs/PEMS08/forecasting.py',HERE/'protocol_v2.json']}
        self.fingerprint=hashlib.sha256(json.dumps(self.hashes,sort_keys=True).encode()).hexdigest()
        self.model=None
    def backbone(self):
        if self.model is None:
            cfg=runpy.run_path(str(ROOT/'configs/PEMS08/forecasting.py'))['CFG']
            cfg.MODEL.PARAM.LSTTN['pretrained_path']=str(self.pre)
            self.model=LSTTN(cfg,**cfg.MODEL.PARAM)
            checkpoint=torch.load(str(self.ck),map_location='cpu');assert checkpoint['epoch']==68
            self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            pre=torch.load(str(self.pre),map_location='cpu')['model_state_dict']
            assert all(torch.equal(self.model.transformer.state_dict()[k],v) for k,v in pre.items())
            self.model.requires_grad_(False);self.model.to(DEVICE).eval()
        return self.model
    def batch(self,split,ids):
        items=[self.ds[split][int(i)] for i in ids]
        return [torch.stack([a[j] for a in items]) for j in range(3)]
    def infer(self,short,long):
        with torch.no_grad():z=self.backbone()(short.to(DEVICE),long.to(DEVICE)).transpose(1,2).cpu()
        assert torch.isfinite(z).all()
        return z.numpy(),max_min_scale(z,**self.stats).numpy()
    def features(self,short,pred,index):
        # Calendar is known at forecast origin. No target read here.
        normalized=pred.transpose(0,2,1)/np.float32(538.5)-np.float32(1)
        history=short[:,:,:,0].numpy().transpose(0,2,1)
        b=np.asarray(index)[:,1];angles=np.stack((2*np.pi*(b%288)/288,2*np.pi*((b//288)%7)/7),axis=-1)
        calendar=np.stack((np.sin(angles[:,0]),np.cos(angles[:,0]),np.sin(angles[:,1]),np.cos(angles[:,1])),axis=-1)
        calendar=np.broadcast_to(calendar[:,None,:],(len(b),170,4))
        x=np.concatenate((normalized,history,calendar),axis=-1).astype('float32')
        assert x.shape==(len(b),170,28) and np.isfinite(x).all()
        return x
    def record(self,split,ids):
        y,short,long=self.batch(split,ids);idx=np.asarray(self.ds[split].index,dtype=np.int64)[ids]
        _,pred=self.infer(short,long)
        return dict(features=self.features(short,pred,idx),prediction=pred,
                    target=max_min_scale(y[...,0],**self.stats).numpy(),index=idx,position=np.asarray(ids,dtype=np.int64))
    def unchanged(self):return all(sha(ROOT/p)==h for p,h in self.hashes.items())

def gate(ctx):
    d=json.loads((RUN/'c1.json').read_text())
    assert d['status']=='C1_CONDITIONAL_PASS' and d['fingerprint']==ctx.fingerprint,'C1 gate not satisfied'

def verify(ctx):
    if (RUN/'c1.json').exists():gate(ctx);print('C1 evidence reused');return
    ids=P['c1']['sample_positions'];refs={}
    wanted={tuple(ctx.ds['test'].index[i]):i for i in ids}
    for path in sorted((ROOT/'artifacts/report-inputs/PEMS08').glob('batch_*.npz')):
        with np.load(path) as z:
            for j,idx in enumerate(z['index']):
                if tuple(idx) in wanted:refs[wanted[tuple(idx)]]=(z['prediction'][j].copy(),z['target'][j].copy(),str(path),j)
    assert len(refs)==len(ids)
    rows=[];checks=[];preds=[];targets=[];old=[]
    with Monitor() as mon:
        for sid in ids:
            y,short,long=ctx.batch('test',[sid]);zn,p=ctx.infer(short,long)
            _,repeat=ctx.infer(short,long)
            y=max_min_scale(y[...,0],**ctx.stats).numpy()
            ref,truth,path,row=refs[sid];assert np.array_equal(y[0],truth)
            diff=p[0].astype('float64')-ref.astype('float64');over=np.abs(diff)>1e-5+1e-4*np.abs(ref)
            inferred_norm=ref.astype('float64')/538.5-1
            equivalent_norm=np.abs(diff)/538.5
            fp64_inverse=(zn.astype('float64')+1)*538.5
            for h,n in np.argwhere(over):
                rows.append(dict(sample_position=sid,index=str(list(ctx.ds['test'].index[sid])),sensor=int(n),horizon=int(h+1),
                    prediction=float(p[0,h,n]),export=float(ref[h,n]),absolute_difference=float(abs(diff[h,n])),
                    relative_difference=float(abs(diff[h,n])/max(abs(float(ref[h,n])),1e-30)),
                    normalized_new=float(zn[0,h,n]),normalized_export_inferred=float(inferred_norm[h,n]),
                    normalized_difference_inferred=float(abs(zn[0,h,n]-inferred_norm[h,n])),
                    fp64_inverse=float(fp64_inverse[0,h,n]),
                    note='Export does not store original normalized tensor; inverse reconstruction is not exact pre-scaler evidence'))
            mnew=horizons(p,y);mold=horizons(ref[None],y)
            md={h:{k:abs(mnew[h][k]-mold[h][k]) for k in mnew[h]} for h in mnew}
            ok=all(v<=P['c1']['max_metric_absolute_difference'][k] for h in md.values() for k,v in h.items())
            checks.append(dict(position=sid,index=list(map(int,ctx.ds['test'].index[sid])),export=path,export_row=row,
                target_exact=True,finite=bool(np.isfinite(p).all()),max_abs=float(np.abs(diff).max()),mean_abs=float(np.abs(diff).mean()),
                above_original_tolerance=int(over.sum()),bitwise_equal=bool(np.array_equal(p[0],ref)),
                repeated_forward_bitwise=bool(np.array_equal(p,repeat)),max_normalized_equivalent_error=float(equivalent_norm.max()),
                max_fp32_vs_fp64_inverse=float(np.abs(fp64_inverse-p).max()),metrics_new=mnew,metrics_export=mold,
                metric_absolute_differences=md,metric_budget_passed=ok))
            preds.append(p);targets.append(y);old.append(ref[None])
    accepted=all(c['metric_budget_passed'] and c['repeated_forward_bitwise'] and c['max_normalized_equivalent_error']<=P['c1']['max_normalized_equivalent_error'] for c in checks)
    assert ctx.unchanged()
    table(RUN/'c1_numeric_violations.csv',rows)
    save(RUN/'c1.json',dict(status='C1_CONDITIONAL_PASS' if accepted else 'FAIL',numeric_status='NUMERIC_MISMATCH',
        scientific_status='CONDITIONAL_ON_CHECKED_SAMPLES_NOT_GLOBAL_EQUIVALENCE',fingerprint=ctx.fingerprint,
        protocol=P['c1'],environment=info(),checks=checks,resources=mon.result,protected_unchanged=True,
        all_modules_eval=all(not m.training for m in ctx.model.modules()),backbone_frozen=all(not p.requires_grad for p in ctx.model.parameters()),
        combined_metrics_new=horizons(np.concatenate(preds),np.concatenate(targets)),
        combined_metrics_export=horizons(np.concatenate(old),np.concatenate(targets))))
    save(RUN/'provenance.json',dict(hashes=ctx.hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol=P,environment=info(),preprocessing_source='UNKNOWN; supplied pickle indices and train-consistent min/max verified',
        pretrained_fit_scope='UNKNOWN; author checkpoint reused and weights matched exactly',
        historical_training_commit='c4c7abb5cbccde7fe87754776ba01fb8ce5fa383; dirty tree per backup'))
    print('C1',accepted,flush=True)
    if not accepted:raise RuntimeError('C1 failed')

class Head(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(28,width),nn.ReLU(),nn.Linear(width,12))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,baseline,alpha=1.):return baseline+alpha*538.5*self.net(x).transpose(1,2)

def pilot(ctx):
    gate(ctx)
    if (RUN/'pilot.json').exists():print('Pilot reused');return
    times=[];records=[]
    with Monitor() as mon:
        # Varied training origins include both zero long-history and full-history cases.
        for ids in ([0,1],[4020,4021],[6000,6001],[10000,10001]):
            t=time.perf_counter();r=ctx.record('train',ids)
            path=RUN/('pilot_'+str(ids[0])+'.npz')
            with path.open('xb') as f:np.savez_compressed(f,**r)
            with np.load(path) as z:assert all(np.array_equal(z[k],v) for k,v in r.items())
            times.append(time.perf_counter()-t);records.append(r)
        fresh=ctx.record('train',[6000,6001]);assert np.array_equal(fresh['prediction'],records[2]['prediction'])
        # Resource-only batch of 64 repeats eight measured real windows; never used for final training.
        pil={k:np.tile(np.concatenate([r[k] for r in records]),(8,)+(1,)*(records[0][k].ndim-1)) for k in ('features','prediction','target')}
        h=Head(16).to(DEVICE);x=torch.from_numpy(pil['features']).to(DEVICE)
        b=torch.from_numpy(pil['prediction']).to(DEVICE);y=torch.from_numpy(pil['target']).to(DEVICE)
        assert torch.equal(h(x,b,alpha=0),b) and torch.equal(h(x,b),b)
        optimizer=torch.optim.Adam(h.parameters(),lr=.001)
        t=time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad();loss=(h(x,b)-y).abs().mean();loss.backward()
            assert any(p.grad is not None for p in h.parameters())
            assert all(p.grad is None for p in ctx.model.parameters())
            optimizer.step()
        if DEVICE=='cuda':torch.cuda.synchronize()
        head_seconds=(time.perf_counter()-t)/20
        assert all(np.array_equal(fresh[k],records[2][k]) for k in fresh)
    disk=sum(p.stat().st_size for p in RUN.glob('pilot_*.npz'))
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True).strip()
    except Exception:gpu='UNKNOWN'
    estimate=float(np.median(times[1:]))/2*sum(len(ctx.ds[s]) for s in ('train','valid'))
    save(RUN/'pilot.json',dict(status='PASS',resources=mon.result,export_batch_seconds=times,
        train_head_step_seconds_batch64=head_seconds,head_pilot_batch_note='8 real windows repeated 8 times for resource measurement only; not training evidence',compressed_bytes_8_windows=disk,gpu_snapshot_after_pilot=gpu,
        train_val_export_estimate_seconds=estimate,estimate_range_observed_seconds=[min(times[1:])/2*14267,max(times[1:])/2*14267],
        uncertainty='Observed range from three post-startup batches, not confidence interval; sustained thermals/I/O unmeasured',
        tests=dict(alpha_zero_exact=True,zero_initialized_head_exact=True,only_head_gradients=True,cache_roundtrip=True,direct_repeat_exact=True),
        feasible=mon.result['cuda_peak_reserved_bytes']<5.5*1024**3 and estimate<4*3600))
    print('Pilot complete',estimate,flush=True)

def validate_shard(path,ctx,split,start,end):
    side=json.loads(path.with_suffix('.json').read_text())
    assert side['fingerprint']==ctx.fingerprint and sha(path)==side['sha256']
    assert side['extractor_sha256']==extractor_hash(),'Cache extraction code changed'
    with np.load(path) as z:
        assert np.array_equal(z['position'],np.arange(start,end))
        assert np.array_equal(z['index'],np.asarray(ctx.ds[split].index)[start:end])
        assert z['features'].shape==(end-start,170,28)
        assert all(np.isfinite(z[k]).all() for k in z.files)

def extractor_hash():
    return hashlib.sha256(''.join(inspect.getsource(getattr(Context,n)) for n in ('batch','infer','features','record')).encode()).hexdigest()

def cache(ctx,split):
    gate(ctx);assert json.loads((RUN/'pilot.json').read_text())['feasible']
    if split=='test':assert (RUN/'locked.json').exists(),'No test cache before configuration lock'
    folder=RUN/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
    total=len(ctx.ds[split]);shards=[];created=0
    with Monitor() as mon:
        for start in range(0,total,64):
            end=min(start+64,total);path=folder/('%06d.npz'%start)
            if path.exists():
                if not path.with_suffix('.json').exists():
                    # Crash between atomic data rename and sidecar commit: verify against fresh inference.
                    with np.load(path) as z:
                        assert np.array_equal(z['position'],np.arange(start,end))
                        for pos in range(start,end,2):
                            fresh=ctx.record(split,list(range(pos,min(pos+2,end))))
                            for key in fresh:assert np.array_equal(z[key][pos-start:pos-start+len(fresh['position'])],fresh[key])
                    save(path.with_suffix('.json'),dict(fingerprint=ctx.fingerprint,extractor_sha256=extractor_hash(),sha256=sha(path),start=start,end=end,recovered_by_direct_inference=True))
                validate_shard(path,ctx,split,start,end)
            else:
                records=[]
                for pos in range(start,end,2):records.append(ctx.record(split,list(range(pos,min(pos+2,end)))))
                record={k:np.concatenate([r[k] for r in records]) for k in records[0]}
                temp=path.with_suffix('.partial')
                with temp.open('wb') as f:np.savez_compressed(f,**record)
                temp.replace(path)
                save(path.with_suffix('.json'),dict(fingerprint=ctx.fingerprint,extractor_sha256=extractor_hash(),sha256=sha(path),start=start,end=end))
                validate_shard(path,ctx,split,start,end);created+=1
            shards.append(dict(file=path.name,sha256=sha(path),start=start,end=end,bytes=path.stat().st_size))
            if start%640==0:print('CACHE',split,end,'/',total,flush=True)
    manifest=folder/'manifest.json'
    if not manifest.exists():save(manifest,dict(status='PASS',split=split,windows=total,shards=shards,fingerprint=ctx.fingerprint,
        resources=mon.result,new_shards_this_session=created,features=P['features'],units='original Flow for prediction and target'))
    print('CACHE DONE',split,mon.result,flush=True)

def arrays(split):
    folder=RUN/'cache'/split;m=json.loads((folder/'manifest.json').read_text());out=[]
    for s in m['shards']:
        p=folder/s['file'];assert sha(p)==s['sha256']
        with np.load(p) as z:out.append({k:z[k].copy() for k in z.files})
    return {k:np.concatenate([r[k] for r in out]) for k in out[0]}

def predict(h,a):
    out=[];h.eval()
    with torch.no_grad():
        for pos in range(0,len(a['features']),64):
            x=torch.from_numpy(a['features'][pos:pos+64]).to(DEVICE);b=torch.from_numpy(a['prediction'][pos:pos+64]).to(DEVICE)
            out.append(h(x,b).cpu().numpy())
    result=np.concatenate(out);assert np.isfinite(result).all();return result

def valid_mask(ctx,a):
    times=a['index'][:,1,None]+np.arange(12)[None,:]
    return ((times>ctx.ds['train'].index[-1][2]-1)&(times<ctx.ds['test'].index[0][1]))[:,:,None]

def train_one(cfg,seed,tr,va,mask):
    dest=RUN/'heads'/(cfg['id']+'_seed'+str(seed));dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():
        result=json.loads((dest/'result.json').read_text())
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        assert result['config']==cfg and result['seed']==seed
        return result
    torch.manual_seed(seed);np.random.seed(seed)
    h=Head(cfg['width']).to(DEVICE);opt=torch.optim.Adam(h.parameters(),lr=P['head']['lr'],weight_decay=P['head']['weight_decay'])
    resume=dest/'resume.pt';history=[];best=float('inf');stale=0;start_epoch=0;best_state=None
    rng=np.random.RandomState(seed)
    if resume.exists():
        r=torch.load(str(resume),map_location='cpu');h.load_state_dict(r['model']);opt.load_state_dict(r['optimizer'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(DEVICE)
        history=r['history'];best=r['best'];stale=r['stale'];start_epoch=r['epoch']+1;best_state=r['best_state'];rng.set_state(r['rng'])
    with Monitor() as mon:
        for epoch in range(start_epoch,P['head']['epochs']):
            if stale>=P['head']['patience']:break
            t=time.perf_counter();h.train();order=rng.permutation(len(tr['features']));loss_sum=0
            for pos in range(0,len(order),64):
                ids=order[pos:pos+64];x=torch.from_numpy(tr['features'][ids]).to(DEVICE)
                b=torch.from_numpy(tr['prediction'][ids]).to(DEVICE);y=torch.from_numpy(tr['target'][ids]).to(DEVICE)
                opt.zero_grad();loss=(h(x,b)-y).abs().mean();loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
                opt.step();loss_sum+=float(loss)*len(ids)
            pred=predict(h,va);selected=metric(pred,va['target'],mask);full=horizons(pred,va['target'])
            history.append(dict(epoch=epoch,train_mae=loss_sum/len(order),validation_interior=selected,validation_full=full,seconds=time.perf_counter()-t))
            if selected['MAE']<best:
                best=selected['MAE'];best_state={k:v.detach().cpu().clone() for k,v in h.state_dict().items()};stale=0
            else:stale+=1
            tmp=resume.with_suffix('.partial');torch.save(dict(model=h.state_dict(),optimizer=opt.state_dict(),history=history,best=best,
                best_state=best_state,stale=stale,epoch=epoch,rng=rng.get_state()),str(tmp));tmp.replace(resume)
            print('HEAD',cfg['id'],seed,epoch,'val',best,flush=True)
        h.load_state_dict(best_state);pred=predict(h,va)
    checkpoint=dest/'head.pt'
    if not checkpoint.exists():torch.save(dict(model_state_dict=best_state,config=cfg,seed=seed,protocol_sha256=sha(HERE/'protocol_v2.json')),str(checkpoint))
    result=dict(config=cfg,seed=seed,history=history,validation_interior=metric(pred,va['target'],mask),validation_full=horizons(pred,va['target']),
                resources=mon.result,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in h.parameters()))
    save(dest/'result.json',result);return result

def train(ctx):
    gate(ctx);tr=arrays('train');va=arrays('valid');mask=valid_mask(ctx,va)
    assert not ctx.model,'Training process must not instantiate backbone'
    # No test arrays are read. Selection mask uses index boundaries only.
    results=[train_one(cfg,0,tr,va,mask) for cfg in P['head']['configs']]
    winner=min(results,key=lambda r:r['validation_interior']['MAE'])
    lock=RUN/'locked.json'
    if not lock.exists():save(lock,dict(config=winner['config'],selection='minimum interior validation MAE seed0',
        selected_before_test=True,seed=0,configs=results,baseline_interior=metric(va['prediction'],va['target'],mask),
        excluded_label_occurrences=int((~mask).sum()),config_sha256=sha(HERE/'protocol_v2.json')))
    else:assert json.loads(lock.read_text())['config']==winner['config']
    seeds=[winner]+[train_one(winner['config'],s,tr,va,mask) for s in (1,2)]
    if not (RUN/'validation_summary.json').exists():
        ms={k:dict(mean=float(np.mean([r['validation_interior'][k] for r in seeds])),sample_std=float(np.std([r['validation_interior'][k] for r in seeds],ddof=1))) for k in ('MAE','RMSE','MAPE_percent')}
        save(RUN/'validation_summary.json',dict(status='PASS',head_seeds=[0,1,2],backbone_seeds=1,seed_results=seeds,interior_metric_mean_std=ms,
            baseline_full=horizons(va['prediction'],va['target']),alpha_zero_control='exact identity checked in pilot',untrained_head_control='zero initialized output exactly baseline'))

def selected_head(seed=0):
    cfg=json.loads((RUN/'locked.json').read_text())['config'];path=RUN/'heads'/(cfg['id']+'_seed'+str(seed))/'head.pt'
    result=json.loads(path.with_name('result.json').read_text());assert sha(path)==result['checkpoint_sha256']
    checkpoint=torch.load(str(path),map_location='cpu');assert checkpoint['protocol_sha256']==sha(HERE/'protocol_v2.json')
    h=Head(cfg['width']).to(DEVICE);h.load_state_dict(checkpoint['model_state_dict'],strict=True);h.eval();return h

def evaluate(ctx):
    gate(ctx);assert (RUN/'validation_summary.json').exists()
    if (RUN/'test_summary.json').exists():print('Test already evaluated');return
    a=arrays('test');rows=[];pred0=None;resources=[]
    for seed in (0,1,2):
        with Monitor() as mon:pred=predict(selected_head(seed),a)
        resources.append(mon.result)
        if seed==0:pred0=pred
        for h,v in horizons(pred,a['target']).items():rows.append(dict(method='residual',seed=seed,horizon_minutes=int(h)*5,**v))
    for h,v in horizons(a['prediction'],a['target']).items():rows.append(dict(method='baseline_local',seed=0,horizon_minutes=int(h)*5,**v))
    old=[];oldtarget=[];oldindex=[]
    for path in sorted((ROOT/'artifacts/report-inputs/PEMS08').glob('batch_*.npz')):
        with np.load(path) as z:old.append(z['prediction']);oldtarget.append(z['target']);oldindex.append(z['index'])
    old=np.concatenate(old);assert np.array_equal(np.concatenate(oldtarget),a['target']) and np.array_equal(np.concatenate(oldindex),a['index'])
    for h,v in horizons(old,a['target']).items():rows.append(dict(method='baseline_original_export',seed=0,horizon_minutes=int(h)*5,**v))
    table(RUN/'test_metrics.csv',rows)
    with (RUN/'test_prediction_seed0.npz').open('xb') as f:np.savez_compressed(f,prediction=pred0,baseline=a['prediction'],target=a['target'],index=a['index'])
    group=[]
    for h in (3,6,12):
        e0=np.abs(a['prediction'][:,h-1]-a['target'][:,h-1]);e1=np.abs(pred0[:,h-1]-a['target'][:,h-1])
        for n in range(170):group.append(dict(group_type='sensor',group=n,horizon_minutes=h*5,baseline_MAE=float(e0[:,n].mean()),head_MAE=float(e1[:,n].mean()),improvement=float((e0[:,n]-e1[:,n]).mean())))
        hour=((a['index'][:,1]+h-1)%288)//12
        for hr in range(24):
            mask=hour==hr
            if mask.any():group.append(dict(group_type='relative_clock_hour',group=hr,horizon_minutes=h*5,baseline_MAE=float(e0[mask].mean()),head_MAE=float(e1[mask].mean()),improvement=float((e0[mask]-e1[mask]).mean())))
        target=a['target'][:,h-1];edges=np.quantile(target,[.25,.5,.75]);bins=np.digitize(target,edges)
        for q in range(4):
            mask=bins==q;group.append(dict(group_type='test_flow_quartile_descriptive',group=q,horizon_minutes=h*5,baseline_MAE=float(e0[mask].mean()),head_MAE=float(e1[mask].mean()),improvement=float((e0[mask]-e1[mask]).mean())))
    table(RUN/'test_groups.csv',group)
    save(RUN/'test_summary.json',dict(status='PASS',metrics=rows,resources=resources,selected_seed_for_demo=0,
        original_export_prediction_max_abs=float(np.abs(old-a['prediction']).max()),
        scientific_claim='Paired held-out comparison under inherited baseline protocol; no statistical significance claim',
        limitations=['11 validation/test shared label timestamps inherited; excluded from new head tuning but baseline checkpoint selection used original validation',
                     'head trained on in-sample predictions from fitted backbone; not out-of-fold stacking', 'absolute calendar unverified; use relative clock only']))
    plots(rows,group)

def plots(rows,group):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('MAE','RMSE','MAPE_percent'):
        fig,ax=plt.subplots(figsize=(7,4))
        for method,label in [('baseline_local','LSTTN cùng pipeline'),('residual','LSTTN + head (seed 0)')]:
            rs=[r for r in rows if r['method']==method and r['seed']==0];ax.plot([r['horizon_minutes'] for r in rs],[r[key] for r in rs],marker='o',label=label)
        ax.set(xlabel='Thời hạn dự báo (phút)',ylabel='MAPE (%)' if key=='MAPE_percent' else key);ax.legend();fig.tight_layout();fig.savefig(RUN/('test_'+key+'.png'),dpi=160);plt.close(fig)
    rs=[r for r in group if r['group_type']=='sensor' and r['horizon_minutes']==60]
    fig,ax=plt.subplots(figsize=(10,3));ax.bar([r['group'] for r in rs],[r['improvement'] for r in rs]);ax.axhline(0,color='black',lw=.5)
    ax.set(xlabel='Vị trí cảm biến',ylabel='MAE baseline − head',title='Cải thiện (+) / suy giảm (−), 60 phút');fig.tight_layout();fig.savefig(RUN/'node_gain_60.png',dpi=160);plt.close(fig)

def xai(ctx):
    gate(ctx);assert (RUN/'test_summary.json').exists()
    if (RUN/'xai.json').exists():print('XAI complete');return
    tr=arrays('train');means=tr['features'].mean(axis=0,dtype=np.float64).astype('float32');del tr
    a=arrays('test');ids=np.linspace(0,len(a['features'])-1,16,dtype=int);s={k:v[ids].copy() for k,v in a.items()}
    model=selected_head();before={k:v.clone() for k,v in model.state_dict().items()};base=predict(model,s)
    groups={'baseline_prediction':slice(0,12),'short_old_4':slice(12,16),'short_middle_4':slice(16,20),
            'short_recent_4':slice(20,24),'time_of_day':slice(24,26),'relative_day_of_week':slice(26,28)}
    attr=[];rows=[]
    with Monitor() as mon:
        for replacement in P['xai']['replacements']:
            for name,sl in groups.items():
                changed={**s,'features':s['features'].copy()}
                changed['features'][:,:,sl]=means[None,:,sl] if replacement.startswith('train') else 0
                pred=predict(model,changed);delta=base-pred;attr.append(delta)
                for h in (3,6,12):rows.append(dict(replacement=replacement,group=name,horizon_minutes=h*5,
                    mean_abs_prediction_change=float(np.abs(delta[:,h-1]).mean()),signed_mean_change=float(delta[:,h-1].mean()),
                    original_MAE=metric(base[:,h-1],s['target'][:,h-1])['MAE'],occluded_MAE=metric(pred[:,h-1],s['target'][:,h-1])['MAE']))
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    with (RUN/'xai_attributions.npz').open('xb') as f:np.savez_compressed(f,attribution=np.stack(attr),index=s['index'],test_position=ids,original_prediction=base,target=s['target'])
    table(RUN/'xai_summary.csv',rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Dự báo baseline','Flow 40–60 phút trước','Flow 20–40 phút trước','Flow 0–20 phút trước','Giờ trong ngày','Ngày tương đối']
    replacement_labels={'train_feature_mean_per_sensor':'Che bằng trung bình train','zero_normalized':'Che bằng 0 chuẩn hóa'}
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for j,replacement in enumerate(P['xai']['replacements']):
        mat=np.array([[r['mean_abs_prediction_change'] for r in rows if r['replacement']==replacement and r['group']==g] for g in groups])
        im=axes[j].imshow(mat,aspect='auto');axes[j].set(xticks=[0,1,2],xticklabels=['15','30','60'],yticks=range(6),yticklabels=labels,title=replacement_labels[replacement],xlabel='Thời hạn dự báo (phút)');fig.colorbar(im,ax=axes[j])
    fig.suptitle('Occlusion đầu vào HEAD — không phải toàn LSTTN');fig.tight_layout();fig.savefig(RUN/'xai_heatmap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));xx=np.arange(6)
    for j,replacement in enumerate(P['xai']['replacements']):
        vals=[r['mean_abs_prediction_change'] for r in rows if r['replacement']==replacement and r['horizon_minutes']==60]
        ax.bar(xx+(j-.5)*.35,vals,.35,label=replacement_labels[replacement])
    ax.set(xticks=xx,xticklabels=labels,ylabel='Trung bình |Δ prediction|',title='Độ nhạy giá trị che — head, horizon 60 phút');ax.tick_params(axis='x',rotation=20);ax.legend();fig.tight_layout();fig.savefig(RUN/'xai_replacement_sensitivity.png',dpi=160);plt.close(fig)
    save(RUN/'xai.json',dict(status='PASS',protocol=P['xai'],test_positions=ids.tolist(),groups=list(groups),
        attribution_axes=['replacement_then_group','sample','horizon','sensor'],resources=mon.result,weights_unchanged=True,
        limits=['Head feature sensitivity only; additive baseline fixed even when baseline prediction feature is masked.',
                'Not raw history attribution of whole LSTTN; no causal claim; replacement may be off distribution.'],
        integrated_gradients='NOT RUN: optional; required occlusion completed with two reference choices'))

def demo(ctx):
    gate(ctx);h=selected_head();a=arrays('test');s={k:v[:2] for k,v in a.items()};pred=predict(h,s)
    assert np.isfinite(pred).all();print(json.dumps(dict(shape=list(pred.shape),index=s['index'].tolist(),metrics=horizons(pred,s['target']),scope='2 cached windows; not whole-test metric')))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['verify','pilot','cache','train','evaluate','xai','demo'])
    parser.add_argument('--split',choices=['train','valid','test'],default='train');args=parser.parse_args()
    RUN.mkdir(exist_ok=True);ctx=Context()
    if args.command=='cache':cache(ctx,args.split)
    else:globals()[args.command](ctx)
