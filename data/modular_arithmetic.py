"""Complete modular task, with split randomness independent of model randomness."""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

@dataclass
class ModularDataset:
    inputs: torch.Tensor
    targets: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    split_hash: str
    train_hash: str
    test_hash: str
    control: str

    def subset(self,split):
        idx=self.train_indices if split=='train' else self.test_indices
        return self.inputs[idx],self.targets[idx]

def array_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()

def make_dataset(config):
    p=config['prime']; a=np.repeat(np.arange(p),p); b=np.tile(np.arange(p),p)
    x=np.stack([a,np.full(p*p,p),b,np.full(p*p,p+1)],axis=1).astype(np.int64)
    op=config['operation']; y={'addition':a+b,'subtraction':a-b,'multiplication':a*b}[op]%p
    rng=np.random.default_rng(config['data_seed']); perm=rng.permutation(p*p); n=int(p*p*config['train_fraction'])
    tr,te=perm[:n],perm[n:]
    control=config.get('control','none')
    if control=='random_labels':
        # Fixed random target function over all pairs, not modular targets at test time.
        y=np.random.default_rng(config['data_seed']+99173).permutation(y)
    elif control!='none': raise ValueError('Unknown control')
    assert len(x)==p*p and len(set(tr)&set(te))==0 and len(tr)+len(te)==p*p
    if control=='none': assert np.array_equal(y,{'addition':a+b,'subtraction':a-b,'multiplication':a*b}[op]%p)
    th=array_hash(tr,x[tr],y[tr]); eh=array_hash(te,x[te],y[te])
    return ModularDataset(torch.from_numpy(x),torch.from_numpy(y.astype(np.int64)),torch.from_numpy(tr),torch.from_numpy(te),array_hash(tr,te,x,y),th,eh,control)
