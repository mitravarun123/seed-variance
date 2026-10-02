"""Explicit causal pre-LN Transformer with selective caches and interventions."""
import math
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout):
        super().__init__(); self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model); self.output=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,layer,record,intervene):
        n,s,d=x.shape
        q,k,v=self.qkv(x).reshape(n,s,3,self.n_heads,self.d_head).permute(2,0,3,1,4).unbind(0)
        for name,z in [('q',q),('k',k),('v',v)]: record(f'L{layer}.{name}',z)
        scores=(q@k.transpose(-1,-2))/math.sqrt(self.d_head)
        mask=torch.ones(s,s,device=x.device,dtype=torch.bool).triu(1)
        pattern=scores.masked_fill(mask,float('-inf')).softmax(-1)
        record(f'L{layer}.pattern',pattern)
        heads=self.dropout(pattern)@v
        heads=intervene(f'L{layer}.heads',heads); record(f'L{layer}.heads',heads)
        return self.output(heads.transpose(1,2).reshape(n,s,d))

class TransformerBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['d_model']; self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.attention=MultiHeadSelfAttention(d,c['n_heads'],c['dropout'])
        self.mlp_in=nn.Linear(d,c['d_mlp']); self.mlp_out=nn.Linear(c['d_mlp'],d)
        self.activation=nn.GELU(); self.dropout=nn.Dropout(c['dropout'])
    def forward(self,x,layer,record,intervene):
        x=intervene(f'L{layer}.resid_pre',x); record(f'L{layer}.resid_pre',x)
        att=intervene(f'L{layer}.attn_out',self.attention(self.ln1(x),layer,record,intervene)); record(f'L{layer}.attn_out',att)
        x=x+self.dropout(att); record(f'L{layer}.resid_mid',x)
        hidden=intervene(f'L{layer}.mlp_hidden',self.activation(self.mlp_in(self.ln2(x)))); record(f'L{layer}.mlp_hidden',hidden)
        out=intervene(f'L{layer}.mlp_out',self.mlp_out(hidden)); record(f'L{layer}.mlp_out',out)
        x=intervene(f'L{layer}.resid_post',x+self.dropout(out)); record(f'L{layer}.resid_post',x)
        return x

class TinyTransformer(nn.Module):
    def __init__(self,prime,config):
        super().__init__(); self.prime=prime; self.config=config
        self.embedding=nn.Embedding(prime+2,config['d_model']); self.position=nn.Parameter(torch.randn(4,config['d_model'])*.02)
        self.blocks=nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layers'])])
        self.final_norm=nn.LayerNorm(config['d_model']); self.unembedding=nn.Linear(config['d_model'],prime,bias=False)
    def forward(self,tokens,cache_keys=None,interventions=None):
        cache={}; keys=set(cache_keys or []); interventions=interventions or {}
        def record(name,z):
            if name in keys or '*' in keys: cache[name]=z.detach()
        def intervene(name,z): return interventions[name](z) if name in interventions else z
        x=intervene('embedding',self.embedding(tokens)+self.position); record('embedding',x)
        for i,block in enumerate(self.blocks): x=block(x,i,record,intervene)
        x=intervene('final_residual',self.final_norm(x)); record('final_residual',x)
        logits=self.unembedding(x[:,-1]); record('logits',logits)
        return (logits,cache) if cache_keys is not None else logits
    def get_activations(self,tokens,keys): return self(tokens,cache_keys=keys)[1]
    def get_attention_patterns(self,tokens): return self.get_activations(tokens,[f'L{i}.pattern' for i in range(len(self.blocks))])
