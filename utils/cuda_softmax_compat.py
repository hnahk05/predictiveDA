"""Opt-in Torch 1.8.1 large-softmax compatibility; model/config unchanged."""
import torch
import torch.nn.functional as F
_ORIGINAL=F.softmax

def split_softmax(input,dim=-1,max_elements=2**27,_stacklevel=3,dtype=None):
 dim=dim if dim>=0 else input.dim()+dim
 if input.dim()<2 or dim==0:raise ValueError('Chunk axis must be independent of the normalization axis')
 per_item=input.numel()//input.shape[0]
 step=max(1,max_elements//per_item)
 if input.requires_grad:
  return torch.cat([_ORIGINAL(part,dim=dim,_stacklevel=_stacklevel,dtype=dtype) for part in input.split(step,dim=0)],dim=0)
 output=torch.empty_like(input,dtype=dtype if dtype is not None else input.dtype)
 for start in range(0,input.shape[0],step):
  output[start:start+step]=_ORIGINAL(input[start:start+step],dim=dim,_stacklevel=_stacklevel,dtype=dtype)
 return output

def install():
 if getattr(F.softmax, '_lsttn_chunked', False): return
 announced = [False]
 def safe(input,dim=None,_stacklevel=3,dtype=None):
  if input.is_cuda and input.dim()==3 and dim in (-1,2) and input.numel()>2**30:
   if not announced[0]:
    print('SPLIT_SOFTMAX',tuple(input.shape),'elements',input.numel(),flush=True); announced[0]=True
   return split_softmax(input,dim=dim,_stacklevel=_stacklevel,dtype=dtype)
  return _ORIGINAL(input,dim=dim,_stacklevel=_stacklevel,dtype=dtype)
 safe._lsttn_chunked=True
 F.softmax=safe


def activate_from_env():
 import os
 enabled=os.environ.get("LSTTN_SPLIT_SOFTMAX")=="1"
 if enabled: install()
 return enabled
