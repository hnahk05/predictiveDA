"""Validate produced artifacts and build evidence tables; never manufactures metrics."""
import csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from pipeline import ROOT,HERE,RUN,P,Context,sha,save,table,arrays,gate,metric

def main():
    ctx=Context();gate(ctx)
    assert (RUN/'xai.json').exists(),'Required occlusion has not completed'
    old_violations=[]
    for device,name in [('cpu','smoke_results_cpu_diagnostic.json'),('cuda','smoke_results_diagnostic.json')]:
        old=json.loads((ROOT/'verification_B_C1'/name).read_text())
        for sample in old['samples']:
            for v in sample.get('violations',[]):
                old_violations.append(dict(device=device,sample_position=sample['test_position'],index=str(sample['index']),
                    sensor=v['sensor_position'],horizon=v['horizon_step'],prediction=v['prediction'],export=v['reference'],
                    absolute_difference=v['absolute_difference'],relative_difference=v['absolute_difference']/max(abs(v['reference']),1e-30),
                    normalized_difference_inferred=v['absolute_difference']/538.5,
                    normalized_evidence='Inferred from exported original units; original normalized tensor unavailable'))
    table(RUN/'prior_cpu_gpu_violations.csv',old_violations)
    evidence=[];allshards=0;disk=0
    for split in ('train','valid','test'):
        a=arrays(split);ds=ctx.ds[split]
        assert np.array_equal(a['index'],np.asarray(ds.index)) and np.array_equal(a['position'],np.arange(len(ds)))
        assert np.isfinite(a['features']).all() and np.isfinite(a['target']).all()
        assert np.all(a['index'][:,0]<a['index'][:,1]) and np.all(a['index'][:,1]<a['index'][:,2])
        ids=[0,len(ds)//2,len(ds)-1]
        for i in ids:
            direct=ctx.record(split,[i]);d=np.abs(direct['prediction'][0].astype('float64')-a['prediction'][i])
            assert np.array_equal(direct['target'][0],a['target'][i])
            assert d.max()/538.5<=P['c1']['max_normalized_equivalent_error']
            assert np.array_equal(direct['features'][0,:,12:],a['features'][i,:,12:])
            evidence.append(dict(split=split,position=i,index=a['index'][i].tolist(),max_abs=float(d.max()),target_exact=True,observed_features_exact=True))
        m=json.loads((RUN/'cache'/split/'manifest.json').read_text());allshards+=len(m['shards']);disk+=sum(s['bytes'] for s in m['shards'])
    assert ctx.unchanged()
    # Original export manifest independently verifies that no historical file was altered.
    original=json.loads((ROOT/'artifacts/report-inputs/PEMS08/result.json').read_text())['artifact_sha256']
    export_checks=[]
    for relative,expected in original.items():
        name=Path(relative).name
        if name.startswith('batch_') or name=='metadata.json':
            local=ROOT/'artifacts/report-inputs/PEMS08'/name
            if local.exists():assert sha(local)==expected;export_checks.append(name)
    save(RUN/'final_integrity.json',dict(status='PASS',cache_direct_comparisons=evidence,cache_shards_validated=allshards,
        compressed_cache_bytes=disk,original_export_files_checked=len(export_checks),protected_baseline_unchanged=True))
    train=[]
    for path in sorted((RUN/'heads').glob('*/result.json')):
        r=json.loads(path.read_text());train.append(dict(config=r['config']['id'],seed=r['seed'],parameters=r['parameters'],
            epochs=len(r['history']),validation_interior_MAE=r['validation_interior']['MAE'],
            validation_interior_RMSE=r['validation_interior']['RMSE'],validation_interior_MAPE_percent=r['validation_interior']['MAPE_percent'],
            training_seconds=r['resources']['seconds'],peak_allocated_bytes=r['resources']['cuda_peak_allocated_bytes'],
            peak_rss_bytes=r['resources']['process_rss_sampled_peak_bytes']))
    table(RUN/'config_seed_results.csv',train)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for path in sorted((RUN/'heads').glob('*/result.json')):
        r=json.loads(path.read_text());ax.plot([x['epoch']+1 for x in r['history']],[x['validation_interior']['MAE'] for x in r['history']],label=path.parent.name)
    ax.set(xlabel='Epoch head',ylabel='MAE validation nội vùng',title='Chọn checkpoint bằng validation, không dùng test');ax.legend();fig.tight_layout();fig.savefig(RUN/'head_learning_curves.png',dpi=160);plt.close(fig)
    test=json.loads((RUN/'test_summary.json').read_text());val=json.loads((RUN/'validation_summary.json').read_text());lock=json.loads((RUN/'locked.json').read_text())
    seed_summary=[]
    for horizon in (15,30,60):
        rs=[r for r in test['metrics'] if r['method']=='residual' and r['horizon_minutes']==horizon]
        assert sorted(r['seed'] for r in rs)==[0,1,2]
        for key in ('MAE','RMSE','MAPE_percent'):
            values=[r[key] for r in rs];seed_summary.append(dict(horizon_minutes=horizon,metric=key,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),seeds='0,1,2',scope='head seeds only; same frozen backbone'))
    table(RUN/'test_seed_summary.csv',seed_summary)
    status=dict(status={'C0':'PARTIAL','C1':'PARTIAL','C2':'PASS','C3':'PASS','PILOT':'PASS','C4':'PASS','C5':'PASS','C6':'PASS','C7':'PASS','C8':'PASS'},
        c1_decision='C1_CONDITIONAL_PASS; NUMERIC_MISMATCH retained',baseline_provenance_unknown=['original preprocessing program','pretraining fitting data scope'],
        head_configs=P['head']['configs'],head_seeds=[0,1,2],selected_config=lock['config'],selected_seed_for_demo=0,
        metrics=test['metrics'],validation_seed_statistics=val['interior_metric_mean_std'],
        cache_bytes=disk,head_runs=train,protected_baseline_unchanged=True,environment=json.loads((RUN/'c1.json').read_text())['environment'],
        evidence_folder=str(RUN),integrated_gradients=json.loads((RUN/'ig_results.json').read_text()) if (RUN/'ig_results.json').exists() else {'status':'NOT RUN'},
        claims='Execution PASS does not mean the head improves accuracy or resolves baseline leakage')
    save(RUN/'status_final.json',status)
    # User requested updating documentation/status; preserve historical copies first.
    for name in ('status.json','README.md','writing_guide.md'):
        src=HERE/name;backup=HERE/(src.stem+'_before_v2'+src.suffix)
        if not backup.exists():shutil.copy2(src,backup)
    (HERE/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=['pipeline.py','run_remaining.py','test_pipeline.py','finalize.py','package_docs.py','monitor_gpu.py','ig_head.py','protocol_v2.json']
    save(RUN/'code_manifest.json',dict(files={n:sha(HERE/n) for n in sources},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        note='Experiment code is new/uncommitted; file hashes identify exact working code. Not a clean commit claim.'))
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
