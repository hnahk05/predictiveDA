"""Explicit bounded in-memory cache; never enabled by default or used for XAI."""
import hashlib
import torch


def fingerprint(tensor):
    x = tensor.detach().cpu().contiguous()
    return (str(x.dtype), tuple(x.shape), hashlib.sha256(x.numpy().tobytes()).hexdigest())


def encode_cached(encoder, x, cache, policy):
    if policy != 'fixed_eval' or any(m.training for m in encoder.modules()):
        raise ValueError('Cache requires fixed_eval policy and every encoder module in eval')
    if any(p.requires_grad for p in encoder.parameters()) or x.requires_grad:
        raise ValueError('Cache requires frozen weights and no input gradients; disable for XAI')
    if getattr(encoder, 'mode', None) != 'inference' or torch.is_autocast_enabled():
        raise ValueError('This cache is validated only for inference backend without autocast')
    # Hash full state including buffers: stale checkpoint/input cannot reuse entries.
    key = (id(encoder), fingerprint(x), str(x.device), tuple(
        (name, fingerprint(value)) for name,value in encoder.state_dict().items()))
    if cache.get('key') != key:
        with torch.no_grad():
            value = encoder(x).detach()
        cache.clear()  # One entry only; no unbounded GPU allocation.
        cache.update(key=key, value=value.clone())
    return cache['value'].clone()
