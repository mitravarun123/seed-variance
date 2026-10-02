"""Validated configuration and stable scientific identities."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml

DEFAULT = {
 'experiment': {'name': 'modular_addition_baseline', 'seed': 42, 'output_root': 'results', 'checkpoint_root': 'checkpoints'},
 'data': {'prime': 113, 'operation': 'addition', 'train_fraction': .3, 'data_seed': 42, 'control': 'none'},
 'model': {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_mlp': 512, 'dropout': 0.0},
 'training': {'max_steps': 100000, 'batch_size': 'full', 'microbatch_size': 256, 'eval_batch_size': 512,
 'learning_rate': .001, 'weight_decay': 1.0, 'beta1': .9, 'beta2': .98, 'log_every': 100,
 'eval_every': 100, 'checkpoint_every': 1000, 'milestones': [100,500,1000,2000,5000,10000,20000,50000], 'cpu_threads': 4},
 'memorization': {'train_accuracy_threshold': .99},
 'grokking': {'train_accuracy_threshold': .99, 'test_accuracy_threshold': .95, 'required_consecutive_evals': 3, 'post_grok_steps': 2000},
 'multi_seed': {'model_seeds': list(range(10))},
 'analysis': {'seeds': list(range(10)), 'checkpoints': ['initial','memorization','grokking','final'],
 'batch_size': 256, 'representation_examples': 512, 'patching_examples': 128, 'top_k': [3,5,10], 'tier': 3},
 'phase4': {'model_seeds': list(range(5)), 'experiments': {
 'train_fraction': [.2,.3,.4,.5], 'prime': [59,97,113,127], 'width': [64,128,256],
 'depth': [1,2,3], 'heads': [1,2,4], 'weight_decay': [0,.1,.5,1.,2.],
 'learning_rate': [.0003,.001,.003], 'operation': ['addition','subtraction','multiplication'],
 'data_seed': [42,123,456,789,1001], 'control': ['random_labels']},
 'tier2_factors': ['train_fraction','operation','weight_decay'], 'bootstrap_seed': 2026, 'bootstrap_samples': 2000}}

def merge(base, extra):
    result = deepcopy(base)
    for k,v in extra.items():
        result[k] = merge(result[k],v) if isinstance(v,dict) and isinstance(result.get(k),dict) else deepcopy(v)
    return result

def load_config(path):
    with open(path) as f: raw = yaml.safe_load(f)
    if not isinstance(raw,dict): raise ValueError('Configuration must be a YAML mapping')
    cfg = merge(DEFAULT,raw)
    if 'data_seed' not in raw.get('data',{}): cfg['data']['data_seed'] = cfg['experiment']['seed']
    validate(cfg)
    return cfg

def validate(c):
    d,m,t,g = (c[k] for k in ['data','model','training','grokking'])
    p = d['prime']
    if isinstance(p,bool) or not isinstance(p,int) or p < 3 or any(p%i == 0 for i in range(2,int(p**.5)+1)):
        raise ValueError('data.prime must be an odd prime >= 3')
    if d['operation'] not in ['addition','subtraction','multiplication']: raise ValueError('Unknown arithmetic operation')
    if d['control'] not in ['none','random_labels']: raise ValueError('Unknown control')
    if not 0 < d['train_fraction'] < 1 or not 1 <= int(p*p*d['train_fraction']) < p*p: raise ValueError('Invalid train fraction')
    for k in ['d_model','n_heads','n_layers','d_mlp']:
        if not isinstance(m[k],int) or m[k] < 1: raise ValueError(f'model.{k} must be a positive integer')
    if m['d_model'] % m['n_heads']: raise ValueError('d_model must be divisible by n_heads')
    if not 0 <= m['dropout'] < 1: raise ValueError('Invalid dropout')
    for k in ['max_steps','microbatch_size','eval_batch_size','log_every','eval_every','checkpoint_every','cpu_threads']:
        if not isinstance(t[k],int) or t[k] < 1: raise ValueError(f'training.{k} must be a positive integer')
    if t['batch_size'] != 'full' and (not isinstance(t['batch_size'],int) or t['batch_size'] < 1): raise ValueError('batch_size must be full or positive integer')
    if t['learning_rate'] <= 0 or t['weight_decay'] < 0: raise ValueError('Invalid optimizer settings')
    if not 0 <= t['beta1'] < 1 or not 0 <= t['beta2'] < 1: raise ValueError('Invalid Adam betas')
    for section,key in [('memorization','train_accuracy_threshold'),('grokking','train_accuracy_threshold'),('grokking','test_accuracy_threshold')]:
        if not 0 < c[section][key] <= 1: raise ValueError('Accuracy thresholds must lie in (0,1]')
    if g['required_consecutive_evals'] < 1 or g['post_grok_steps'] < 0: raise ValueError('Invalid grokking confirmation settings')
    for section in ['multi_seed','phase4']:
        seeds=c[section]['model_seeds']
        if not seeds or len(seeds)!=len(set(seeds)) or any(not isinstance(s,int) or s<0 for s in seeds): raise ValueError('Seeds must be unique nonnegative integers')

def identity(c):
    """Only experiment-defining settings; runtime output locations are excluded."""
    return {k: c[k] for k in ['data','model','training','memorization','grokking']}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def config_hash(c): return digest(identity(c))

def write_config(c,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: yaml.safe_dump(c,f,sort_keys=False)
