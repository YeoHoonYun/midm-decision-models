"""Experimental frozen-MiDM heads. Not deployed and no financial weights included."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def engineering(d):
 n=d['numeric'];p=n[:,17:24];p=p/np.maximum(p.sum(1,keepdims=True),1e-8);sort=np.sort(p,axis=1);ent=-(p*np.log(np.maximum(p,1e-8))).sum(1)/np.log(7)
 return np.column_stack([n,ent,sort[:,-1]-sort[:,-2],n[:,2]/np.maximum(abs(n[:,4]),.01),n[:,0]/np.maximum(abs(n[:,4]),.01),np.sign(n[:,6])+np.sign(n[:,7])+np.sign(n[:,8]),np.clip(d['margin'],-6,6)])
class Head:
 def __init__(self,kind,penalty):self.kind=kind;self.penalty=penalty
 def transform(self,d,fit=False):
  x=engineering(d)
  if fit:self.mu=x.mean(0);self.sd=np.maximum(x.std(0),1e-4)
  z=np.clip((x-self.mu)/self.sd,-6,6)
  if self.kind=='latent_residual':
   l=d['latent']
   if fit:
    self.lmu=l.mean(0);_,_,v=np.linalg.svd(l-self.lmu,full_matrices=False);self.proj=v[:8].T;self.lsd=np.maximum(((l-self.lmu)@self.proj).std(0),1e-4)
   z=np.column_stack([z,np.clip((l-self.lmu)@self.proj/self.lsd,-6,6)])
  if self.kind=='horizon_residual':
   h=d['numeric'][:,16]>.5;held=d['numeric'][:,15]>.5;z=np.column_stack([z,z*h[:,None],z*held[:,None]])
  if self.kind=='random_features':
   if fit:self.proj=np.random.default_rng(731).normal(size=(z.shape[1],32))/np.sqrt(z.shape[1]);self.phase=np.random.default_rng(732).uniform(0,2*np.pi,32)
   z=np.column_stack([z,np.cos(z@self.proj+self.phase)])
  return np.column_stack([np.ones(len(z)),z])
 def fit(self,d):
  z=self.transform(d,True);y=d['y'];offset=np.clip(d['margin'],-6,6) if self.kind!='random_features' else np.zeros(len(y))
  def fun(w):
   a=offset+z@w;return np.mean(np.logaddexp(0,a)-y*a)+self.penalty*np.dot(w,w)/2,z.T@(expit(a)-y)/len(y)+self.penalty*w
  opt=minimize(fun,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7});assert opt.success,opt.message
  self.w=opt.x;return self
 def predict(self,d):
  offset=np.clip(d['margin'],-6,6) if self.kind!='random_features' else 0
  return expit(offset+self.transform(d)@self.w)
