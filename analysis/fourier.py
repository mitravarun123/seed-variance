"""Explicit orthonormal real Fourier analysis for odd cyclic groups."""
import numpy as np

def build_fourier_basis(p):
    """Return a p by p orthonormal real Fourier basis for an odd p-cycle."""
    if not isinstance(p,int) or p < 3 or p % 2 == 0:
        raise ValueError('p must be an odd integer >= 3')
    x=np.arange(p); columns=[np.ones(p)/np.sqrt(p)]
    for k in range(1,(p+1)//2):
        columns.extend([np.sqrt(2/p)*np.cos(2*np.pi*k*x/p),np.sqrt(2/p)*np.sin(2*np.pi*k*x/p)])
    basis=np.column_stack(columns)
    if not np.allclose(basis.T@basis,np.eye(p),atol=1e-10): raise ValueError('Fourier basis failed orthogonality')
    return basis

def fourier_transform_vector(x):
    x=np.asarray(x)
    if x.ndim!=1: raise ValueError('Expected a one-dimensional signal')
    return build_fourier_basis(len(x)).T@x

def fourier_transform_matrix(x,axis=0):
    x=np.asarray(x)
    if x.ndim==0: raise ValueError('Expected an array with at least one dimension')
    axis=np.core.numeric.normalize_axis_index(axis,x.ndim)
    z=np.moveaxis(x,axis,0); transformed=np.tensordot(build_fourier_basis(len(z)).T,z,axes=(1,0))
    return np.moveaxis(transformed,0,axis)

def fourier_power(coefficients): return np.abs(coefficients)**2

def frequency_energy(x,exclude_dc=True):
    co=fourier_transform_matrix(x); power=fourier_power(co).reshape(len(x),-1).sum(1)
    energy=np.r_[power[0],power[1:].reshape(-1,2).sum(1)]
    if exclude_dc: energy[0]=0
    total=energy.sum(); return energy/total if total>0 else energy

def dominant_frequencies(energy,k=5): return (np.argsort(energy[1:])[::-1][:min(k,len(energy)-1)]+1).tolist()

def concentration(q,k=5):
    q=np.asarray(q)[1:]; total=q.sum()
    if total==0: return {'top_k_fraction':None,'normalized_entropy':None}
    q=q/total; nz=q[q>0]
    return {'top_k_fraction':float(np.sort(q)[-min(k,len(q)):].sum()),'normalized_entropy':float(-(nz*np.log(nz)).sum()/np.log(len(q))) if len(q)>1 else 0.}

def grid_frequency_energy(representations,p):
    """2D Fourier of all operand pairs; include interaction energy, not just marginals."""
    x=np.asarray(representations)
    if x.shape[0]!=p*p: raise ValueError(f'Expected exactly p^2={p*p} ordered pair representations')
    x=x.reshape(p,p,-1)
    # The unitary FFT is the same orthonormal cyclic Fourier transform and is
    # substantially cheaper than two dense p-by-p basis multiplications.
    coeff=np.fft.fft2(x,axes=(0,1),norm='ortho')
    power=(np.abs(coeff)**2).sum(-1); grouped=np.zeros(((p+1)//2,(p+1)//2))
    indices=[[0]]+[[k,p-k] for k in range(1,(p+1)//2)]
    for a,ai in enumerate(indices):
        for b,bi in enumerate(indices): grouped[a,b]=power[np.ix_(ai,bi)].sum()
    grouped[0,0]=0; total=grouped.sum()
    return grouped/total if total else grouped
