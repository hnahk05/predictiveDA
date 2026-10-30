"""Full-node/full-batch GPU preflight. Temporary weights, not paper results."""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='METR-LA',choices=['METR-LA','PEMS-BAY','PEMS04','PEMS08'])
    args=parser.parse_args()
    import torch
    from models.transformer import Transformer
    from models.lsttn import LSTTN
    from utils.datasets import MTSDataset
    from utils.load_data import standard_scale,max_min_scale
    from utils.log import load_pkl
    from utils.metrics import masked_mae
    from torch.utils.data import DataLoader,Subset
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; do not start main experiments')
    if torch.__version__ != '1.8.1+cu111': raise RuntimeError('Wrong Torch build')
    ds=args.dataset; result={}; temporary=ROOT/'artifacts/gpu-benchmark'/ds
    temporary.mkdir(parents=True,exist_ok=True)
    scale=max_min_scale if ds in ('PEMS04','PEMS08') else standard_scale
    names=('min','max') if ds in ('PEMS04','PEMS08') else ('mean','std')
    scaler={key:load_pkl(str(ROOT/'datasets'/ds/(key+'.pkl'))) for key in names}
    for stage in ('pretraining','forecasting'):
        cfg=runpy.run_path(str(ROOT/'config_paper'/ds/(stage+'.py')))['CFG']
        data=MTSDataset(str(ROOT/'datasets'/ds/'data.pkl'),str(ROOT/'datasets'/ds/'train_index.pkl'),4032,throw=(stage=='pretraining'),pretrain=(stage=='pretraining'))
        size=cfg.TRAIN.DATA.BATCH_SIZE
        eligible=[i for i,row in enumerate(data.index) if row[1]>=4032][:size*2]
        loader=DataLoader(Subset(data,eligible),batch_size=size,shuffle=False,num_workers=0)
        if stage=='pretraining':
            model=Transformer(**cfg.MODEL.PARAM).cuda()
        else:
            cfg.MODEL.PARAM.LSTTN['pretrained_path']=str(temporary/'temporary-transformer.pt')
            model=LSTTN(cfg,**cfg.MODEL.PARAM).cuda()
        model.train()
        optim=getattr(torch.optim,cfg.TRAIN.OPTIM.TYPE)(model.parameters(),**cfg.TRAIN.OPTIM.PARAM)
        torch.cuda.reset_peak_memory_stats(); times=[]
        for batch in loader:
            torch.cuda.synchronize(); start=time.perf_counter(); optim.zero_grad()
            if stage=='pretraining':
                prediction,target=model(batch[0].cuda())
                null=0 if ds in ('METR-LA','PEMS-BAY') else float('nan')
            else:
                target,short,long=batch
                prediction=model(short.cuda(),long.cuda()).transpose(1,2)
                target=target.cuda()[...,0];null=0
            loss=masked_mae(scale(prediction,**scaler),scale(target,**scaler),null)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward();optim.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        # A conservative planning estimate, not a guaranteed completion time.
        per_step=max(times); estimate=math.ceil(2*per_step*math.ceil(len(data)/size)*cfg.TRAIN.NUM_EPOCHS+600)
        result[stage]=dict(batch=size,nodes=data.data.shape[1],seconds_per_step=times,
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,estimated_seconds=estimate,
            note='2x slowest measured step + 10min; validation/I/O and contention may exceed this')
        if stage=='pretraining':torch.save({'model_state_dict':model.cpu().state_dict()},temporary/'temporary-transformer.pt')
        del model,optim,data,loader,prediction,target,loss,batch
        torch.cuda.empty_cache()
    (temporary/'benchmark.json').write_text(json.dumps(result,indent=2))
    plan=json.loads((ROOT/'environment/experiments-paper.json').read_text())
    plan['runs']=[job for job in plan['runs'] if job['id'].startswith(ds+'-')]
    for job in plan['runs']:
        stage='pretraining' if job['id'].endswith('pretrain') else 'forecasting'
        job['estimated_seconds']=result[stage]['estimated_seconds']
    (temporary/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(result,indent=2))
    print('Measured plan:',temporary/'plan.json')


if __name__=='__main__':main()
