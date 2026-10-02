"""Headless publication figures with default matplotlib colors."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def save(fig,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)

def training_curves(rows,path,grok_step=None):
    df=pd.DataFrame(rows); fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,metric in zip(axes,['accuracy','loss']):
        for split in ['train','test']: ax.plot(df.step,df[f'{split}_{metric}'],label=split)
        if grok_step is not None: ax.axvline(grok_step,linestyle='--',label='Confirmed grokking')
        ax.set(xlabel='Optimization step',ylabel=metric.capitalize()); ax.legend(); ax.grid(alpha=.2)
    save(fig,path)

def heatmap(matrix,path,title='',labels=None):
    fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(matrix,aspect='auto'); fig.colorbar(im,ax=ax)
    ax.set_title(title)
    if labels is not None:
        ax.set_xticks(range(len(labels)),labels,rotation=45); ax.set_yticks(range(len(labels)),labels)
    save(fig,path)
