"""Public definitions only. Raw data and complete experiment logs are not distributed here."""
import numpy as np
CLASSES=['up_only','down_only','up_then_down','down_then_up','neither']
def path_class(returns):
    a=np.asarray(returns,dtype=float)
    assert a.ndim==2 and np.isfinite(a).all()
    u=a>=.05; d=a<=-.05; up=u.any(1); down=d.any(1)
    y=np.full(len(a),4)
    y[up&~down]=0; y[down&~up]=1
    y[up&down&(u.argmax(1)<d.argmax(1))]=2
    y[up&down&(u.argmax(1)>d.argmax(1))]=3
    return y
def coverage(probabilities,y,n):
    order=np.argsort(-np.asarray(probabilities),axis=1,kind='stable')
    return (order[:,:n]==np.asarray(y)[:,None]).any(1)
def date_block_interval(correct_counts,row_counts,block=5,repeats=5000,seed=361500):
    correct_counts=np.asarray(correct_counts);row_counts=np.asarray(row_counts)
    n=len(row_counts);block=min(block,n);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-block+1,(repeats,int(np.ceil(n/block))))
    ix=(starts[:,:,None]+np.arange(block)).reshape(repeats,-1)[:,:n]
    values=correct_counts[ix].sum(1)/row_counts[ix].sum(1)
    return np.quantile(values,[.025,.975])
