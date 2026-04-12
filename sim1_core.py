import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, anderson
import warnings; warnings.filterwarnings('ignore')
import os, pickle

np.random.seed(42)
figdir = '/home/claude/figs'
os.makedirs(figdir, exist_ok=True)

# ---- Maps ----
def logistic_map(r, x0, N, burn=800):
    x = np.empty(burn+N); x[0]=x0
    for i in range(1, burn+N): x[i]=r*x[i-1]*(1-x[i-1])
    return x[burn:]

def henon_map(a, b, x0, y0, N, burn=1500):
    x=np.empty(burn+N); y=np.empty(burn+N); x[0],y[0]=x0,y0
    for i in range(1,burn+N): x[i]=1-a*x[i-1]**2+y[i-1]; y[i]=b*x[i-1]
    return x[burn:], y[burn:]

def pomeau_manneville(z, x0, N, burn=2000):
    x=np.empty(burn+N); x[0]=x0
    for i in range(1,burn+N): x[i]=(x[i-1]+x[i-1]**z)%1.0
    return x[burn:]

# ---- Core functions ----
def compute_var(trajs, g, m_vals):
    g_vals=g(trajs); g_mean=np.mean(g_vals); c=g_vals-g_mean
    v=np.zeros(len(m_vals))
    for i,m in enumerate(m_vals):
        if m>trajs.shape[1]: v[i]=np.nan; continue
        Sm=np.sum(c[:,:m],axis=1); v[i]=np.var(Sm,ddof=1)
    return v

def compute_Zm(trajs, g, ms):
    gv=g(trajs); c=gv-np.mean(gv); Sm=np.sum(c[:,:ms],axis=1)
    s=np.std(Sm,ddof=1)
    return Sm/s if s>1e-15 else None

def kl_hist(Zm, nb=80):
    if Zm is None: return 50.0
    h,e=np.histogram(Zm,bins=nb,range=(-5,5)); h=h+1; p=h/h.sum()
    bc=0.5*(e[:-1]+e[1:]); bw=e[1]-e[0]; q=stats.norm.pdf(bc)*bw; q=q/q.sum()
    return sum(p[i]*np.log(p[i]/q[i]) for i in range(len(p)) if p[i]>0 and q[i]>0)

def fit_alpha(m, v):
    ok=(v>0)&np.isfinite(v)
    if ok.sum()<3: return 0.,0.,0.
    s,i,r,_,se=stats.linregress(np.log(m[ok]),np.log(v[ok]))
    return s, r**2, se

def lyap_log(r, N=50000):
    x=0.4
    for _ in range(500): x=r*x*(1-x)
    l=0.
    for _ in range(N):
        x=r*x*(1-x); d=abs(r*(1-2*x))
        if d>0: l+=np.log(d)
    return l/N

# ---- Parameters ----
Nt=2000; Ns=2000; ms=800; mg=np.array([50,100,200,400,800,1200,1600]); nb=80

# ---- 1. Logistic ----
print("[1] Logistic map...")
rvals=[3.50,3.60,3.80,4.00]
lt={}
for r in rvals:
    t=np.zeros((Nt,Ns))
    for i in range(Nt): t[i]=logistic_map(r,np.random.uniform(0.01,0.99),Ns)
    lt[r]=t

gx=lambda x:x; gx2=lambda x:x**2
T1=[]
for r in rvals:
    lam=lyap_log(r)
    for gn,gf in [('x',gx),('x²',gx2)]:
        v=compute_var(lt[r],gf,mg); a,R2,ase=fit_alpha(mg,v)
        Zm=compute_Zm(lt[r],gf,ms); kl=kl_hist(Zm); w1=wasserstein_distance(Zm,np.random.randn(len(Zm))) if Zm is not None else 50.
        ad=anderson(Zm,'norm').statistic if Zm is not None else 999.
        nfi=kl+abs(a-1)
        T1.append(dict(r=r,g=gn,lam=lam,a=a,R2=R2,kl=kl,w1=w1,ad=ad,nfi=nfi,nfi_sh=kl,nfi_sc=abs(a-1)))
        print(f"  r={r} g={gn}: α={a:.4f} R²={R2:.4f} KL={kl:.4f} W1={w1:.4f} AD={ad:.1f} NFI={nfi:.4f}")

# ---- 2. Hénon ----
print("[2] Hénon map...")
Nth=1500
hp=[(1.4,0.3),(1.4,0.1),(1.3,0.3)]
ht={}
for a,b in hp:
    xs=np.zeros((Nth,Ns))
    ok=0
    for i in range(Nth):
        try:
            xh,_=henon_map(a,b,np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5),Ns)
            if np.all(np.abs(xh)<100): xs[ok]=xh; ok+=1
        except: pass
    ht[(a,b)]=xs[:ok]
    print(f"  ({a},{b}): {ok} valid")

T2=[]
for a,b in hp:
    tr=ht[(a,b)]
    for gn,gf in [('x',gx)] + ([('x²',gx2)] if (a,b)==(1.4,0.3) else []):
        v=compute_var(tr,gf,mg); al,R2,_=fit_alpha(mg,v)
        Zm=compute_Zm(tr,gf,ms); kl=kl_hist(Zm)
        nfi=kl+abs(al-1)
        T2.append(dict(ab=f'({a},{b})',g=gn,a=al,R2=R2,kl=kl,nfi=nfi))
        print(f"  ({a},{b}) g={gn}: α={al:.4f} R²={R2:.4f} KL={kl:.4f} NFI={nfi:.4f}")

# ---- 3. Pomeau-Manneville ----
print("[3] Pomeau-Manneville...")
Ntp=1500; pm_mg=np.array([50,100,200,400,800])
T3=[]
pmt={}
for z in [1.2,1.5,1.8,2.0,2.5]:
    t=np.zeros((Ntp,Ns))
    for i in range(Ntp): t[i]=pomeau_manneville(z,np.random.uniform(0.01,0.99),Ns)
    pmt[z]=t
    v=compute_var(t,gx,pm_mg); al,R2,_=fit_alpha(pm_mg,v)
    Zm=compute_Zm(t,gx,ms); kl=kl_hist(Zm)
    w1=wasserstein_distance(Zm,np.random.randn(len(Zm))) if Zm is not None else 50.
    nfi=kl+abs(al-1)
    ath=min(2.,1./(z-1)) if z>1 else 1.
    T3.append(dict(z=z,a=al,ath=ath,R2=R2,kl=kl,w1=w1,nfi=nfi))
    print(f"  z={z}: α={al:.3f}(th≈{ath:.2f}) R²={R2:.3f} KL={kl:.4f} NFI={nfi:.4f}")

# Save all
pickle.dump(dict(lt=lt,ht=ht,pmt=pmt,T1=T1,T2=T2,T3=T3,mg=mg,pm_mg=pm_mg,ms=ms,nb=nb,rvals=rvals),
            open('/home/claude/sim_data.pkl','wb'))
print("\n[OK] Core simulations done, data saved.")
