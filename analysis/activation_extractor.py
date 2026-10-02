"""Selective batched extraction, immediately transferred to CPU."""
import numpy as np
import torch

@torch.no_grad()
def extract(model,inputs,keys,device,batch_size=256,last_position=True):
    model.eval(); parts={key:[] for key in keys}
    for start in range(0,len(inputs),batch_size):
        cache=model.get_activations(inputs[start:start+batch_size].to(device),keys)
        for key,value in cache.items():
            if last_position:
                if key.endswith(('.heads','.q','.k','.v')): value=value[:,:,-1,:].flatten(1)
                elif key.endswith('.pattern'): value=value[:,:,-1,:]
                elif value.ndim==3: value=value[:,-1,:]
            parts[key].append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}
