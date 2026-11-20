"""Collect existing evidence and block downstream runs unless C1 passes. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('status.json'))
    a=p.parse_args()
    if a.output.exists(): p.error('Choose a new output; existing evidence is never overwritten.')
    def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
    gpu=read('verification_B_C1/smoke_results_diagnostic.json')
    cpu=read('verification_B_C1/smoke_results_cpu_diagnostic.json')
    benchmark=read('results/extension/inference_cost_v2.json')
    files=['models/lsttn.py','models/transformer.py','models/graph_wavenet.py','utils/datasets.py',
           'utils/load_data.py','runners/lsttn.py','configs/PEMS08/forecasting.py',
           'verification_B_C1/smoke_results_diagnostic.json','verification_B_C1/smoke_results_cpu_diagnostic.json']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    passed=gpu['status']=='PASS' and cpu['status']=='PASS'
    output={'current_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'baseline_training_commit_from_protocol':'c4c7abb5cbccde7fe87754776ba01fb8ce5fa383',
        'baseline_training_tree_dirty':True,'source_sha256':hashes,
        'status':{'C0':'PARTIAL','C1':'PASS' if passed else 'FAIL',
                  **{f'C{i}':'NOT STARTED' if passed else 'BLOCKED' for i in range(2,8)},'PILOT':'PARTIAL','C8':'PARTIAL'},
        'downstream_allowed':passed,'gpu_comparison':gpu['comparison_summary'],
        'cpu_comparison':cpu['comparison_summary'], 'checkpoint':gpu['checkpoint'],
        'historical_inference_measurements':benchmark['results'],
        'pilot_cache_export_seconds':None,'pilot_head_training_seconds':None,'cache_size_bytes':None,
        'head_configurations_run':0,'head_seeds_run':[],
        'rounding_diagnostic':{'float32_spacing_near_normalized_minus_one':2**-23,
            'inverse_scaler_slope':1077/2,'one_spacing_in_original_units':1077/2*2**-23,
            'interpretation':'Observed 6.41942e-5 differences are consistent with FP32 spacing amplified by inverse scaling; not proof of unique cause or PASS.'},
        'limitations':['Historical training code had uncommitted changes.',
            'Split retained at 60/20/20 with 11 shared label timestamps at each adjacent boundary.',
            'Pretrained fitting data scope has not been fully established.',
            'No cache/head/full-system XAI pilot may proceed while C1 fails.']}
    with a.output.open('x',encoding='utf-8') as f: json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':output['status'],'downstream_allowed':passed}))
    return 0 if passed else 2

if __name__=='__main__': sys.exit(main())
