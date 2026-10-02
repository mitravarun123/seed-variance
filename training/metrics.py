"""Batched task and intervention metrics."""
import numpy as np
import torch
from torch.nn import functional as F

@torch.no_grad()
def evaluate(model,x,y,device,batch_size=512,interventions=None,per_example=False):
    model.eval(); losses=[]; correct=[]; logits_correct=[]; margins=[]; predictions=[]
    for start in range(0,len(x),batch_size):
        xb=x[start:start+batch_size].to(device); yb=y[start:start+batch_size].to(device)
        logits=model(xb,interventions=interventions); true=logits.gather(1,yb[:,None]).squeeze(1)
        other=logits.clone(); other.scatter_(1,yb[:,None],float('-inf'))
        losses.append(F.cross_entropy(logits,yb,reduction='none').cpu().numpy()); predictions.append(logits.argmax(-1).cpu().numpy())
        correct.append((logits.argmax(-1)==yb).cpu().numpy()); logits_correct.append(true.cpu().numpy()); margins.append((true-other.max(-1).values).cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in [('loss',losses),('accuracy',correct),('correct_logit',logits_correct),('logit_margin',margins)]}
    result={k:float(v.mean()) for k,v in arrays.items()}
    if per_example: result['arrays']=arrays; result['predictions']=np.concatenate(predictions)
    return result
