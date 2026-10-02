"""Rotation/permutation invariant linear CKA for aligned examples."""
import numpy as np

def linear_cka(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if x.shape[0]!=y.shape[0]: raise ValueError('CKA requires aligned common examples')
    x=x-x.mean(0); y=y-y.mean(0)
    # Sample-space form avoids a d_mlp by d_mlp matrix.
    k=x@x.T; l=y@y.T; denom=np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.sum(k*l)/denom) if denom>0 else None

def cosine(x,y):
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/denom) if denom else None
