"""Device, reproducibility and atomic artifact utilities."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
import torch

def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def json_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)); os.replace(tmp,path)

def torch_save(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); os.replace(tmp,path)

def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items(): h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def environment():
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    return {'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'git_commit':commit,'device':str(device())}

def cleanup():
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
