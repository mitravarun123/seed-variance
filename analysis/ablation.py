"""Interventions on head outputs before output projection and on MLP outputs."""
import numpy as np
import torch
from training.metrics import evaluate

def zero_head(head):
    def intervention(z):
        out=z.clone(); out[:,head]=0; return out
    return intervention

def zero(z): return torch.zeros_like(z)

def component_intervention(component):
    kind,layer,index=component
    return {f'L{layer}.heads':zero_head(index)} if kind=='head' else {f'L{layer}.mlp_out':zero}

def components(model):
    return [('head',l,h) for l in range(len(model.blocks)) for h in range(model.config['n_heads'])]+[('mlp',l,None) for l in range(len(model.blocks))]

def run_ablations(model,x,y,device,batch_size=256):
    baseline=evaluate(model,x,y,device,batch_size,per_example=True); rows=[]
    for component in components(model):
        result=evaluate(model,x,y,device,batch_size,component_intervention(component),per_example=True)
        row={'kind':component[0],'layer':component[1],'head':component[2],'n_examples':len(x)}
        for metric in ['accuracy','loss','correct_logit','logit_margin']:
            delta=baseline['arrays'][metric].astype(float)-result['arrays'][metric].astype(float)
            row[f'delta_{metric}']=float(delta.mean()); row[f'std_delta_{metric}']=float(delta.std()); row[f'median_delta_{metric}']=float(np.median(delta))
        rows.append(row)
    return baseline,rows

@torch.no_grad()
def patching(model,x,y,device,keys):
    clean=x.to(device); corrupted=clean.clone(); corrupted[:,0]=(corrupted[:,0]+1)%model.prime
    clean_logits,cache=model(clean,cache_keys=keys); corrupt_logits=model(corrupted); target=y.to(device)
    def margin(logits):
        correct=logits.gather(1,target[:,None]).squeeze(1); other=logits.clone(); other.scatter_(1,target[:,None],float('-inf'))
        return correct-other.max(-1).values
    clean_m=margin(clean_logits); corrupt_m=margin(corrupt_logits); denom=clean_m-corrupt_m
    rows=[]
    for key in keys:
        def patch(z,key=key):
            out=z.clone(); out[:,-1]=cache[key][:,-1]; return out
        logits=model(corrupted,interventions={key:patch}); recovered=margin(logits)-corrupt_m
        usable=denom.abs()>1e-6
        rows.append({'component':key,'n_examples':len(x),'clean_margin':float(clean_m.mean()),'corrupted_margin':float(corrupt_m.mean()),
         'margin_recovery':float(recovered.mean()),'normalized_recovery':float((recovered[usable]/denom[usable]).mean()) if usable.any() else None,
         'n_valid_normalized':int(usable.sum())})
    return rows
