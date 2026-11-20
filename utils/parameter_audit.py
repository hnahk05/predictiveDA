"""Observed forward usage and isolated optimizer-step diagnostics, not training history."""
import torch


def audit_parameters(model, inputs, optimizer_config, target=None):
    from easytorch.core.optimizer_builder import build_optim
    params = dict(model.named_parameters())
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    training = {m:m.training for m in model.modules()}
    requires = {n:p.requires_grad for n,p in params.items()}
    calls, observed = set(), set()
    handles = []
    def hook(name):
        def observe(module, args, output):
            calls.add(name)
            for key,_ in module.named_parameters(recurse=False):
                observed.add((name+'.' if name else '')+key)
            # MultiheadAttention invokes out_proj via functional linear, not module.forward.
            if isinstance(module, torch.nn.MultiheadAttention):
                for key,_ in module.out_proj.named_parameters():
                    observed.add(name+'.out_proj.'+key)
        return observe
    def describe(names):
        return {'numel':sum(params[n].numel() for n in names), 'names':sorted(names)}
    try:
        model.eval()
        for name,module in model.named_modules():
            handles.append(module.register_forward_hook(hook(name)))
        model.zero_grad()
        output = model(*inputs)
        loss = output.square().mean() if target is None else (output-target).abs().mean()
        loss.backward()
        backward = {n for n,p in params.items() if p.grad is not None}
        nonzero = {n for n in backward if torch.count_nonzero(params[n].grad).item()}
        optimizer = build_optim(optimizer_config, model)
        ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        optimizer_names = {n for n,p in params.items() if id(p) in ids}
        optimizer.step()  # One diagnostic step; immediately restored, never saved.
        changed = {n for n,p in params.items() if not torch.equal(p.detach(),initial[n])}
        result = dict(total_registered=describe(set(params)),
            trainable_requires_grad=describe({n for n in params if requires[n]}),
            frozen=describe({n for n in params if not requires[n]}),
            optimizer_membership=describe(optimizer_names),
            forward_observed_owner_parameters=describe(observed),
            backward_with_gradient=describe(backward), backward_nonzero_gradient=describe(nonzero),
            actually_changed_in_isolated_step=describe(changed),
            called_modules=sorted(calls),
            scope='One eval-mode diagnostic backward/optimizer step, state restored; not historical training updates.',
            forward_scope='Module-call observation plus MultiheadAttention functional out_proj; not proof every observed scalar influences output.')
    finally:
        for h in handles:
            h.remove()
        model.load_state_dict(initial, strict=True)
        model.zero_grad()
        for m,mode in training.items():
            m.training = mode
    assert all(torch.equal(model.state_dict()[k],v) for k,v in initial.items())
    result['state_restored_exactly'] = True
    return result
