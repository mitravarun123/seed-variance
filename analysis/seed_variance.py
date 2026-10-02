"""Seed aggregates retain incomplete runs and horizon censoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.plots import plt,save
from training.utils import json_save

def descriptive(values):
    x=np.array([float(v) for v in values if v is not None and pd.notna(v)])
    if len(x)==0: return {'n':0,'mean':None,'std':None,'median':None,'min':None,'max':None,'iqr':None,'cv':None}
    std=float(x.std(ddof=1)) if len(x)>1 else None
    return {'n':len(x),'mean':float(x.mean()),'std':std,'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),
     'iqr':float(np.percentile(x,75)-np.percentile(x,25)),'cv':std/float(x.mean()) if std is not None and x.mean()!=0 else None}

def aggregate(root):
    root=Path(root); rows=[json.loads(p.read_text()) for p in sorted(root.glob('seed_*/summary.json'))]
    if not rows: return None
    hashes={r['dataset_split_hash'] for r in rows}
    if len(hashes)!=1: raise ValueError('Fixed-data study has mismatched split hashes')
    df=pd.DataFrame(rows); df.drop(columns=['environment'],errors='ignore').to_csv(root/'aggregate_metrics.csv',index=False)
    df[['model_seed','memorization_step','grokking_step','grokking_delay_steps','best_test_accuracy','status','right_censored']].to_csv(root/'seed_summary.csv',index=False)
    complete=[r for r in rows if r['status']=='completed']; summary={'number_of_seeds':len(rows),'number_completed':len(complete),
     'number_grokked':sum(r['grokking_detected'] for r in complete),'number_censored':sum(r['right_censored'] for r in complete),
     'fraction_grokked':sum(r['grokking_detected'] for r in complete)/len(complete) if complete else None,
     'incomplete_seeds':[r['model_seed'] for r in rows if r['status']!='completed'],'dataset_split_hash':next(iter(hashes))}
    for k in ['memorization_step','grokking_step','grokking_delay_steps','best_test_accuracy','final_test_accuracy']:
        summary[k]=descriptive([r.get(k) for r in complete])
    json_save(summary,root/'aggregate_summary.json')
    # A compact, legible row-per-seed figure complements the machine-readable CSV.
    table_columns=['model_seed','memorization_step','grokking_step','grokking_delay_steps','best_test_accuracy','status']
    table=df[table_columns].copy()
    table=table.where(pd.notna(table), '—')
    fig,ax=plt.subplots(figsize=(10,max(2.2,.42*(len(table)+1))))
    ax.axis('off')
    rendered=ax.table(cellText=table.astype(str).values,colLabels=table_columns,loc='center',cellLoc='center')
    rendered.auto_set_font_size(False); rendered.set_fontsize(8); rendered.scale(1,1.35)
    save(fig,root/'seed_summary.png')
    for metric,filename in [('test_accuracy','accuracy_curves'),('train_accuracy','training_accuracy_curves'),('test_loss','loss_curves')]:
        fig,ax=plt.subplots(figsize=(7,4)); trajectories=[]
        for r in rows:
            m=pd.read_csv(Path(r['result_dir'])/'metrics.csv'); ax.plot(m.step,m[metric],alpha=.55,label=str(r['model_seed'])); trajectories.append(m.set_index('step')[metric])
        aligned=pd.concat(trajectories,axis=1); mean=aligned.mean(1); ax.plot(mean.index,mean,linewidth=2,label='Available-seed mean')
        ax.set(xlabel='Optimization step',ylabel=metric); ax.legend(); save(fig,root/f'{filename}.png')
    for key,file in [('grokking_step','grokking_steps'),('grokking_delay_steps','grokking_delay')]:
        fig,ax=plt.subplots(figsize=(6,4))
        for r in complete:
            if r[key] is not None: ax.scatter(r['model_seed'],r[key],marker='o')
            elif key=='grokking_step': ax.scatter(r['model_seed'],r['total_training_steps'],marker='x',label=f"Seed {r['model_seed']}: censored")
        ax.set(xlabel='Model seed',ylabel=key); save(fig,root/f'{file}.png')
    observed=[r['grokking_step'] for r in complete if r['grokking_detected']]
    fig,ax=plt.subplots(); ax.hist(observed,bins=max(1,min(10,len(observed)))); ax.set(xlabel='Observed grokking step',ylabel='Count'); save(fig,root/'grokking_distribution.png')
    # Kaplan-Meier product-limit representation, censoring at actual horizon.
    times=sorted({r['grokking_step'] if r['grokking_detected'] else r['total_training_steps'] for r in complete})
    survival=1.; sx=[0]; sy=[1.]
    for time in times:
        at_risk=sum((r['grokking_step'] if r['grokking_detected'] else r['total_training_steps'])>=time for r in complete)
        events=sum(r['grokking_detected'] and r['grokking_step']==time for r in complete)
        if at_risk: survival*=1-events/at_risk
        sx.append(time); sy.append(survival)
    fig,ax=plt.subplots(); ax.step(sx,sy,where='post'); ax.set(xlabel='Optimization step',ylabel='Fraction not yet grokked',ylim=(0,1.05)); save(fig,root/'grokking_survival.png')
    return summary
