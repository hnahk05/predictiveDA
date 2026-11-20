"""Optional head-only IG after required occlusion; fixed additive baseline skip."""
import json
import numpy as np
import torch
from pipeline import RUN,P,DEVICE,arrays,selected_head,Monitor,save,table

def main():
    assert json.loads((RUN/'xai.json').read_text())['status']=='PASS'
    if (RUN/'ig_results.json').exists():print('IG evidence reused');return
    protocol=dict(scope='residual head features only, additive baseline held fixed',steps=[64,256],integration='trapezoidal',
        reference='train feature mean per sensor',horizons=[3,6,12],sample_rule='same 16 evenly spaced windows as required occlusion',
        completeness_atol_Flow=.05,completeness_rtol=.01,stability_mean_relative_L1_limit=.05,
        criteria_basis='Optional diagnostic: absolute budget 0.05 Flow plus 1% output difference; stability average relative attribution L1 5%. Fixed before IG results; failures reported without increasing steps or tolerances.')
    if not (RUN/'ig_protocol.json').exists():save(RUN/'ig_protocol.json',protocol)
    else:assert json.loads((RUN/'ig_protocol.json').read_text())==protocol
    tr=arrays('train');reference=tr['features'].mean(0,dtype=np.float64).astype('float32');del tr
    a=arrays('test');ids=np.linspace(0,len(a['features'])-1,16,dtype=int)
    x=torch.from_numpy(a['features'][ids]).to(DEVICE);ref=torch.from_numpy(np.broadcast_to(reference,x.shape).copy()).to(DEVICE)
    b=torch.from_numpy(a['prediction'][ids]).to(DEVICE);model=selected_head();weights={k:v.clone() for k,v in model.state_dict().items()}
    outcomes=[];attrs64=[];attrs256=[];deltas=[]
    with Monitor() as mon:
        for horizon in protocol['horizons']:
            with torch.no_grad():difference=(model(x,b)[:,horizon-1]-model(ref,b)[:,horizon-1]).cpu().numpy()
            attrs=[]
            for steps in protocol['steps']:
                grad_sum=torch.zeros_like(x)
                for step in range(steps+1):
                    point=(ref+(x-ref)*(step/steps)).detach().requires_grad_(True)
                    output=model(point,b)[:,horizon-1]
                    # Head is independent per node: sum gives each node's own target gradient.
                    grad=torch.autograd.grad(output.sum(),point)[0]
                    assert torch.isfinite(grad).all()
                    grad_sum+=grad*(.5 if step in (0,steps) else 1.)
                attrs.append(((x-ref)*grad_sum/steps).detach().cpu().numpy())
            coarse,fine=attrs;delta=fine.sum(-1)-difference
            relative_l1=np.abs(fine-coarse).sum(-1)/np.maximum(np.abs(fine).sum(-1),1e-6)
            passed=np.abs(delta)<=.05+.01*np.abs(difference)
            outcomes.append(dict(horizon_minutes=horizon*5,completeness_max_abs=float(np.abs(delta).max()),
                completeness_mean_abs=float(np.abs(delta).mean()),completeness_violations=int((~passed).sum()),
                outputs=int(delta.size),stability_mean_relative_L1=float(relative_l1.mean()),stability_max_relative_L1=float(relative_l1.max()),
                passed=bool(passed.all() and relative_l1.mean()<=.05)))
            attrs64.append(coarse);attrs256.append(fine);deltas.append(delta)
        assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    with (RUN/'ig_attributions.npz').open('xb') as f:np.savez_compressed(f,attribution64=np.stack(attrs64),attribution256=np.stack(attrs256),
        completeness_delta=np.stack(deltas),index=a['index'][ids],test_position=ids)
    table(RUN/'ig_summary.csv',outcomes)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[slice(0,12),slice(12,16),slice(16,20),slice(20,24),slice(24,26),slice(26,28)]
    mat=np.array([[np.abs(at[...,sl]).sum(-1).mean() for at in attrs256] for sl in groups])
    fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(mat,aspect='auto');ax.set(xticks=range(3),xticklabels=['15','30','60'],
        yticks=range(6),yticklabels=['Dự báo baseline','Flow cũ 4 bước','Flow giữa 4 bước','Flow gần 4 bước','Giờ trong ngày','Ngày tương đối'],
        xlabel='Thời hạn dự báo (phút)',title='IG của head — trung bình tổng |attribution| theo nhóm');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(RUN/'ig_head_heatmap.png',dpi=160);plt.close(fig)
    save(RUN/'ig_results.json',dict(status='PASS' if all(r['passed'] for r in outcomes) else 'PARTIAL',protocol=protocol,results=outcomes,
        resources=mon.result,weights_unchanged=True,parameter_gradients_accumulated=False,gradient_path='selected horizon output sum; independent per-node MLP',
        limitations=['Head features only; no raw-history/backbone IG','Reference path can be off distribution','No causal interpretation']))
    print(json.dumps(outcomes,indent=2))

if __name__=='__main__':main()
