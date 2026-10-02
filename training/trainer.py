"""One trainer shared by baseline, seed study and controlled robustness runs."""
import csv
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.tiny_transformer import TinyTransformer
from data.modular_arithmetic import make_dataset
from training.config import config_hash,write_config
from training.utils import device,set_seed,parameter_hash,environment,json_save,torch_save
from training.metrics import evaluate
from analysis.grokking import GrokkingState
from analysis.plots import training_curves

class Trainer:
    def __init__(self,config,model_seed,result_dir,checkpoint_dir):
        self.config=config; self.seed=model_seed; self.result_dir=Path(result_dir); self.checkpoint_dir=Path(checkpoint_dir)
        self.result_dir.mkdir(parents=True,exist_ok=True); self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        torch.set_num_threads(config['training']['cpu_threads']); self.device=device(); set_seed(model_seed)
        self.dataset=make_dataset(config['data']); self.model=TinyTransformer(config['data']['prime'],config['model']).to(self.device)
        t=config['training']; self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'],betas=(t['beta1'],t['beta2']))
        self.generator=torch.Generator().manual_seed(model_seed+1777)
        self.initial_hash=parameter_hash(self.model); self.state=GrokkingState(); self.step=0; self.elapsed=0.; self.rows=[]; self.best=-1.; self.stop_requested=False
    def sanity(self):
        x,y=self.dataset.subset('train'); self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        logits=self.model(x[:min(8,len(x))].to(self.device))
        if logits.shape!=(min(8,len(x)),self.config['data']['prime']): raise RuntimeError('Incorrect model logits shape')
        loss=F.cross_entropy(logits,y[:len(logits)].to(self.device)); loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()): raise RuntimeError('Missing or nonfinite gradients')
        self.optimizer.zero_grad(set_to_none=True)
    def payload(self):
        return {'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),
         'step':self.step,'config':self.config,'config_hash':config_hash(self.config),'seed':self.seed,'model_seed':self.seed,
         'dataset_split_hash':self.dataset.split_hash,'initial_model_hash':self.initial_hash,'metrics':self.rows,
         'grokking_state':self.state.to_dict(),'best_test_accuracy':self.best,'elapsed_seconds':self.elapsed,
         'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
         'batch':self.generator.get_state(),'mps':torch.mps.get_rng_state() if self.device.type=='mps' else None}}
    def checkpoint(self,name): torch_save(self.payload(),self.checkpoint_dir/f'{name}.pt')
    def resume(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['config_hash']!=config_hash(self.config) or ck['model_seed']!=self.seed or ck['dataset_split_hash']!=self.dataset.split_hash:
            raise ValueError('Resume checkpoint configuration, model seed or dataset differs')
        self.model.load_state_dict(ck['model_state_dict']); self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.step=ck['step']; self.rows=ck['metrics']; self.elapsed=ck['elapsed_seconds']; self.best=ck['best_test_accuracy']
        self.initial_hash=ck['initial_model_hash']; self.state=GrokkingState(**ck['grokking_state'])
        r=ck['rng']; random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch']); self.generator.set_state(r['batch'])
        if self.device.type=='mps' and r['mps'] is not None: torch.mps.set_rng_state(r['mps'])
    def write_metrics(self):
        if not self.rows: return
        path=self.result_dir/'metrics.csv'; tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0])); w.writeheader(); w.writerows(self.rows)
        tmp.replace(path)
    def evaluation(self):
        t=self.config['training']; train=evaluate(self.model,*self.dataset.subset('train'),self.device,t['eval_batch_size'])
        test=evaluate(self.model,*self.dataset.subset('test'),self.device,t['eval_batch_size'])
        row={'step':self.step,'train_loss':train['loss'],'test_loss':test['loss'],'train_accuracy':train['accuracy'],
         'test_accuracy':test['accuracy'],'elapsed_seconds':self.elapsed,'learning_rate':self.optimizer.param_groups[0]['lr']}
        self.rows.append(row); self.write_metrics()
        events=self.state.update(self.step,self.elapsed,train['accuracy'],test['accuracy'],self.config)
        for event in events:
            print(f'{event.upper()} ACHIEVED | step={self.step} | elapsed={self.elapsed:.1f}s',flush=True); self.checkpoint(event)
        if test['accuracy']>self.best: self.best=test['accuracy']; self.checkpoint('best')
        if self.step%t['log_every']==0 or events or self.step==t['max_steps']:
            print(f"Step {self.step} | Train Loss {train['loss']:.4f} | Test Loss {test['loss']:.4f} | Train Acc {train['accuracy']:.4f} | Test Acc {test['accuracy']:.4f}",flush=True)
    def summary(self,status):
        c=self.config; row=self.rows[-1]; s=self.state
        value={'status':status,'seed':self.seed,'model_seed':self.seed,'data_seed':c['data']['data_seed'],'dataset_split_hash':self.dataset.split_hash,
         'train_split_hash':self.dataset.train_hash,'test_split_hash':self.dataset.test_hash,'initial_model_hash':self.initial_hash,
         'config_hash':config_hash(c),'device':str(self.device),'prime':c['data']['prime'],'operation':c['data']['operation'],'control':c['data']['control'],
         'train_fraction':c['data']['train_fraction'],'number_train_samples':len(self.dataset.train_indices),'number_test_samples':len(self.dataset.test_indices),
         'parameter_count':sum(p.numel() for p in self.model.parameters()),'memorization_detected':s.memorization_step is not None,
         'grokking_detected':s.grokking_step is not None,**s.to_dict(),
         'confirmed_grokking_step':s.grokking_step,'grokking_delay_steps':s.grokking_step-s.memorization_step if s.grokking_step is not None else None,
         'grokking_delay_seconds':s.grokking_time_seconds-s.memorization_time_seconds if s.grokking_step is not None else None,
         'final_train_accuracy':row['train_accuracy'],'final_test_accuracy':row['test_accuracy'],
         'final_train_loss':row['train_loss'],'final_test_loss':row['test_loss'],'best_test_accuracy':self.best,
         'total_training_steps':self.step,'total_training_time_seconds':self.elapsed,
         'right_censored':status=='completed' and s.grokking_step is None,'max_steps':c['training']['max_steps'],
         'checkpoint_dir':str(self.checkpoint_dir.resolve()),'result_dir':str(self.result_dir.resolve()),'environment':environment()}
        json_save(value,self.result_dir/'summary.json'); return value
    def run(self,resume=None):
        self.sanity()
        if resume: self.resume(resume)
        elif (self.checkpoint_dir/'initial.pt').exists(): raise FileExistsError('Run already exists; use resume or force')
        write_config(self.config,self.result_dir/'config.yaml')
        json_save({'config':self.config,'environment':environment(),'model_seed':self.seed,'dataset_split_hash':self.dataset.split_hash},self.result_dir/'manifest.json')
        print(f"GROKKING EXPERIMENT | {self.config['data']} | seed={self.seed} | device={self.device} | parameters={sum(p.numel() for p in self.model.parameters())}",flush=True)
        if not resume: self.checkpoint('initial'); self.evaluation(); self.checkpoint('latest')
        x,y=self.dataset.subset('train'); t=self.config['training']; started=time.monotonic(); previous=self.elapsed
        old_handlers={}
        def request_stop(signum,frame): self.stop_requested=True
        for sig in [signal.SIGINT,signal.SIGTERM]: old_handlers[sig]=signal.signal(sig,request_stop)
        try:
            while self.step<t['max_steps']:
                if self.state.grokking_step is not None and self.step>=self.state.grokking_step+self.config['grokking']['post_grok_steps']: break
                self.model.train(); self.optimizer.zero_grad(set_to_none=True)
                idx=torch.arange(len(x)) if t['batch_size']=='full' else torch.randperm(len(x),generator=self.generator)[:min(t['batch_size'],len(x))]
                for sub in idx.split(t['microbatch_size']):
                    logits=self.model(x[sub].to(self.device)); loss=F.cross_entropy(logits,y[sub].to(self.device),reduction='sum')/len(idx)
                    if not torch.isfinite(loss): raise FloatingPointError('Training loss is nonfinite')
                    loss.backward()
                self.optimizer.step(); self.step+=1; self.elapsed=previous+time.monotonic()-started
                end=self.step==t['max_steps'] or self.stop_requested or (self.state.grokking_step is not None and self.step>=self.state.grokking_step+self.config['grokking']['post_grok_steps'])
                if self.step%t['eval_every']==0 or end: self.evaluation()
                if self.step%t['checkpoint_every']==0: self.checkpoint('latest')
                if self.step in t['milestones']: self.checkpoint(f'step_{self.step}')
                if self.stop_requested: break
        finally:
            self.elapsed=previous+time.monotonic()-started
            for sig,handler in old_handlers.items(): signal.signal(sig,handler)
            self.checkpoint('latest'); self.write_metrics()
        status='interrupted' if self.stop_requested else 'completed'
        if status=='completed': self.checkpoint('final')
        training_curves(self.rows,self.result_dir/'training_curves.png',self.state.grokking_step)
        return self.summary(status)
