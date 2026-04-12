import numpy as np, pickle, os
from scipy import stats
from scipy.stats import wasserstein_distance, anderson
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

D=pickle.load(open('/home/claude/sim_data.pkl','rb'))
lt,ht,pmt=D['lt'],D['ht'],D['pmt']
mg,ms,nb=D['mg'],D['ms'],D['nb']
gx=lambda x:x

def compute_var(t,g,mv):
    gv=g(t); c=gv-np.mean(gv); v=np.zeros(len(mv))
    for i,m in enumerate(mv):
        if m>t.shape[1]: v[i]=np.nan; continue
        v[i]=np.var(np.sum(c[:,:m],axis=1),ddof=1)
    return v

def compute_Zm(t,g,m):
    gv=g(t); c=gv-np.mean(gv); Sm=np.sum(c[:,:m],axis=1); s=np.std(Sm,ddof=1)
    return Sm/s if s>1e-15 else None

def kl_hist(Z,nb=80):
    if Z is None: return 50.
    h,e=np.histogram(Z,bins=nb,range=(-5,5)); h=h+1; p=h/h.sum()
    bc=.5*(e[:-1]+e[1:]); bw=e[1]-e[0]; q=stats.norm.pdf(bc)*bw; q=q/q.sum()
    return sum(p[i]*np.log(p[i]/q[i]) for i in range(len(p)) if p[i]>0 and q[i]>0)

def fit_a(m,v):
    ok=(v>0)&np.isfinite(v)
    if ok.sum()<3: return 0.,0.
    s,_,r,_,_=stats.linregress(np.log(m[ok]),np.log(v[ok]))
    return s,r**2

def lyap(r,N=30000):
    x=0.4
    for _ in range(500): x=r*x*(1-x)
    l=0.
    for _ in range(N):
        x=r*x*(1-x); d=abs(r*(1-2*x))
        if d>0: l+=np.log(d)
    return l/N

# ---- m★ dependence ----
print("[4] m★ dependence...")
msv=[100,200,400,800,1600]
mstar_res={}
for r in [3.50,3.60,4.00]:
    res=[]
    for m in msv:
        if m>lt[r].shape[1]: continue
        msub=mg[mg<=m]; 
        if len(msub)<3: msub=np.array([50,100,m])
        v=compute_var(lt[r],gx,msub); a,R2=fit_a(msub,v)
        Z=compute_Zm(lt[r],gx,m); kl=kl_hist(Z)
        res.append(dict(m=m,a=a,kl=kl,nfi=kl+abs(a-1),R2=R2))
    mstar_res[r]=res
    for d in res: print(f"  r={r} m★={d['m']}: α={d['a']:.4f} KL={d['kl']:.4f} NFI={d['nfi']:.4f}")

# ---- Noise robustness ----
print("\n[5] Noise robustness...")
eps_vals=[0.,0.001,0.005,0.01,0.05,0.1]
noise_res={}
for r in [3.60,4.00]:
    res=[]
    for eps in eps_vals:
        tn=lt[r]+eps*np.random.randn(*lt[r].shape)
        tn=np.clip(tn,0.001,0.999)
        v=compute_var(tn,gx,mg); a,_=fit_a(mg,v)
        Z=compute_Zm(tn,gx,ms); kl=kl_hist(Z)
        res.append(dict(eps=eps,a=a,kl=kl,nfi=kl+abs(a-1)))
    noise_res[r]=res
    for d in res: print(f"  r={r} ε={d['eps']}: α={d['a']:.4f} NFI={d['nfi']:.4f}")

# ---- NFI vs r scan ----
print("\n[6] NFI vs r scan...")
from sim1_core import logistic_map
rscan=np.linspace(3.50,4.00,80)
Nts=400
nfi_scan=[]; lam_scan=[]
for r in rscan:
    t=np.zeros((Nts,1200))
    for i in range(Nts): t[i]=logistic_map(r,np.random.uniform(0.01,0.99),1200)
    msub=np.array([50,100,200,400,800])
    v=compute_var(t,gx,msub); a,_=fit_a(msub,v)
    Z=compute_Zm(t,gx,800); kl=kl_hist(Z)
    nfi_scan.append(kl+abs(a-1)); lam_scan.append(lyap(r))
nfi_scan=np.array(nfi_scan); lam_scan=np.array(lam_scan)
print(f"  {len(rscan)} points done")

# ---- Binning sensitivity ----
print("\n[7] Binning sensitivity...")
binr=np.arange(30,140,10)
bsens={}
for r in [3.50,3.60,3.80,4.00]:
    Z=compute_Zm(lt[r],gx,ms)
    bsens[r]=[kl_hist(Z,n) for n in binr]

# ---- Close-neighborhood weighting ----
print("[8] Close-neighborhood weighting...")
rcl=np.linspace(3.58,3.66,9); cv=[0.5,1.,2.]
cnfi={c:[] for c in cv}
for r in rcl:
    t=np.zeros((800,1200))
    for i in range(800): t[i]=logistic_map(r,np.random.uniform(0.01,0.99),1200)
    v=compute_var(t,gx,np.array([50,100,200,400,800])); a,_=fit_a(np.array([50,100,200,400,800]),v)
    Z=compute_Zm(t,gx,800); kl=kl_hist(Z)
    for c in cv: cnfi[c].append(kl+c*abs(a-1))

# Save
pickle.dump(dict(mstar_res=mstar_res,noise_res=noise_res,rscan=rscan,
    nfi_scan=nfi_scan,lam_scan=lam_scan,bsens=bsens,binr=binr,
    rcl=rcl,cnfi=cnfi,cv=cv,eps_vals=eps_vals,msv=msv),
    open('/home/claude/sim_extra.pkl','wb'))
print("\n[OK] Extra simulations done.")
