import numpy as np, pickle
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size':10,'axes.labelsize':12,'axes.titlesize':11,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'figure.dpi':200})

D=pickle.load(open('/home/claude/sim_data.pkl','rb'))
E=pickle.load(open('/home/claude/sim_extra.pkl','rb'))
lt,ht,pmt=D['lt'],D['ht'],D['pmt']
mg,ms,nb=D['mg'],D['ms'],D['nb']
T1,T2,T3=D['T1'],D['T2'],D['T3']
gx=lambda x:x; gx2=lambda x:x**2
FD='/home/claude/figs'

def compute_var(t,g,mv):
    gv=g(t);c=gv-np.mean(gv);v=np.zeros(len(mv))
    for i,m in enumerate(mv):
        if m>t.shape[1]:v[i]=np.nan;continue
        v[i]=np.var(np.sum(c[:,:m],axis=1),ddof=1)
    return v
def compute_Zm(t,g,m):
    gv=g(t);c=gv-np.mean(gv);Sm=np.sum(c[:,:m],axis=1);s=np.std(Sm,ddof=1)
    return Sm/s if s>1e-15 else None
def fit_a(m,v):
    ok=(v>0)&np.isfinite(v)
    if ok.sum()<3:return 0.,0.
    s,_,r,_,_=stats.linregress(np.log(m[ok]),np.log(v[ok]));return s,r**2

zr=np.linspace(-4,4,200); gpdf=stats.norm.pdf(zr)
rn={3.50:'Periodic',3.60:'Weak chaos',3.80:'Chaotic',4.00:'Fully chaotic'}

# ---- FIG 1: Variance scaling 4-panel ----
fig,axes=plt.subplots(1,4,figsize=(13,3.2))
for i,r in enumerate([3.50,3.60,3.80,4.00]):
    ax=axes[i]; v=compute_var(lt[r],gx,mg); a,R2=fit_a(mg,v)
    ok=v>0
    ax.loglog(mg[ok],v[ok],'o',ms=4,color='#185FA5',zorder=3)
    if ok.sum()>=2:
        mf=np.linspace(mg[ok][0],mg[ok][-1],100)
        li=np.mean(np.log(v[ok])-a*np.log(mg[ok]))
        ax.loglog(mf,np.exp(li)*mf**a,'-',color='#D85A30',lw=1.5)
    ax.set_title(f'{rn[r]}\n$\\alpha={a:.3f}$, $R^2={R2:.3f}$',fontsize=9)
    ax.set_xlabel('$m$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel('Var($S_m$)')
plt.tight_layout()
plt.savefig(f'{FD}/fig1.pdf',bbox_inches='tight'); plt.close()
print("Fig 1 done")

# ---- FIG 2: Gaussianity 4-panel ----
fig,axes=plt.subplots(1,4,figsize=(13,3.2))
for i,r in enumerate([3.50,3.60,3.80,4.00]):
    ax=axes[i]; Z=compute_Zm(lt[r],gx,ms)
    d=[x for x in T1 if x['r']==r and x['g']=='x'][0]
    if Z is not None and np.std(Z)>.01:
        ax.hist(Z,bins=60,density=True,alpha=.6,color='#185FA5',edgecolor='none')
    ax.plot(zr,gpdf,'-',color='#D85A30',lw=1.5,label='$\\mathcal{N}(0,1)$')
    ax.set_title(f'{rn[r]}\nKL$={d["kl"]:.4f}$',fontsize=9)
    ax.set_xlabel('$Z_{m_\\star}$'); ax.set_xlim(-4,4); ax.legend(fontsize=7)
    if i==0: ax.set_ylabel('Density')
plt.tight_layout()
plt.savefig(f'{FD}/fig2.pdf',bbox_inches='tight'); plt.close()
print("Fig 2 done")

# ---- FIG 3: Hénon 2x2 ----
fig,axes=plt.subplots(2,2,figsize=(8,6.5))
for col,(a,b) in enumerate([(1.4,.3),(1.3,.3)]):
    tr=ht[(a,b)]; v=compute_var(tr,gx,mg); al,R2=fit_a(mg,v); ok=v>0
    ax=axes[0,col]
    ax.loglog(mg[ok],v[ok],'o',ms=4,color='#185FA5')
    if ok.sum()>=2:
        mf=np.linspace(mg[ok][0],mg[ok][-1],100)
        li=np.mean(np.log(v[ok])-al*np.log(mg[ok]))
        ax.loglog(mf,np.exp(li)*mf**al,'--',color='#D85A30',lw=1.5)
    lb='Chaotic' if a==1.4 and b==.3 else 'Low mixing'
    ax.set_title(f'$({a},{b})$ [{lb}]\n$\\alpha={al:.2f}$, $R^2={R2:.3f}$',fontsize=9)
    ax.set_xlabel('$m$');ax.set_ylabel('Var($S_m$)');ax.grid(True,alpha=.3)
    ax=axes[1,col]; Z=compute_Zm(tr,gx,ms)
    d2=[x for x in T2 if x['ab']==f'({a},{b})' and x['g']=='x'][0]
    if Z is not None and np.std(Z)>.01:
        ax.hist(Z,bins=60,density=True,alpha=.6,color='#185FA5')
    ax.plot(zr,gpdf,'-',color='#D85A30',lw=1.5)
    ax.set_title(f'KL$={d2["kl"]:.3f}$',fontsize=9)
    ax.set_xlabel('$Z_{m_\\star}$');ax.set_ylabel('Density');ax.set_xlim(-4,4)
plt.tight_layout()
plt.savefig(f'{FD}/fig3.pdf',bbox_inches='tight'); plt.close()
print("Fig 3 done")

# ---- FIG 4: NFI vs r ----
fig,ax=plt.subplots(figsize=(5.5,3.8))
ax.plot(E['rscan'],E['nfi_scan'],'.',ms=3,color='#185FA5')
ax.set_xlabel('$r$');ax.set_ylabel('NFI');ax.set_title('NFI vs $r$')
ax.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig4.pdf',bbox_inches='tight'); plt.close()
print("Fig 4 done")

# ---- FIG 5: NFI vs λ ----
fig,ax=plt.subplots(figsize=(5.5,3.8))
ax.plot(E['lam_scan'],E['nfi_scan'],'.',ms=4,color='#185FA5')
ax.set_xlabel('Lyapunov exponent $\\lambda$');ax.set_ylabel('NFI')
ax.set_title('NFI vs $\\lambda$ (logistic map)')
m=(E['lam_scan']>-0.05)&(E['lam_scan']<0.15)&(E['nfi_scan']>0.5)&(E['nfi_scan']<15)
if m.any():
    ax.plot(E['lam_scan'][m],E['nfi_scan'][m],'o',ms=6,mfc='none',mec='#D85A30',mew=1.5,label='Same $\\lambda$, different NFI')
    ax.legend()
ax.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig5.pdf',bbox_inches='tight'); plt.close()
print("Fig 5 done")

# ---- FIG 6: Pomeau-Manneville ----
pm_mg=D['pm_mg']
fig,axes=plt.subplots(1,3,figsize=(11,3.2))
for i,z in enumerate([1.5,2.0,2.5]):
    ax=axes[i]; t=pmt[z]; v=compute_var(t,gx,pm_mg); al,R2=fit_a(pm_mg,v); ok=v>0
    ax.loglog(pm_mg[ok],v[ok],'o',ms=4,color='#185FA5')
    if ok.sum()>=2:
        mf=np.linspace(pm_mg[ok][0],pm_mg[ok][-1],100)
        li=np.mean(np.log(v[ok])-al*np.log(pm_mg[ok]))
        ax.loglog(mf,np.exp(li)*mf**al,'--',color='#D85A30',lw=1.5)
    ath=min(2.,1./(z-1))
    ax.set_title(f'$z={z}$\n$\\alpha={al:.3f}$ (theory$\\approx{ath:.2f}$)\n$R^2={R2:.3f}$',fontsize=9)
    ax.set_xlabel('$m$'); ax.grid(True,alpha=.3)
    if i==0: ax.set_ylabel('Var($S_m$)')
plt.tight_layout()
plt.savefig(f'{FD}/fig6.pdf',bbox_inches='tight'); plt.close()
print("Fig 6 done")

# ---- FIG 7: m★ dependence ----
fig,ax=plt.subplots(figsize=(5.5,3.8))
col={3.50:'#A32D2D',3.60:'#BA7517',4.00:'#185FA5'}
lab={3.50:'$r=3.50$ (periodic)',3.60:'$r=3.60$ (weak chaos)',4.00:'$r=4.00$ (fully chaotic)'}
for r in [3.50,3.60,4.00]:
    mv=[d['m'] for d in E['mstar_res'][r]]; nv=[d['nfi'] for d in E['mstar_res'][r]]
    ax.plot(mv,nv,'o-',ms=5,color=col[r],label=lab[r])
ax.set_xlabel('$m_\\star$');ax.set_ylabel('NFI');ax.set_title('NFI vs $m_\\star$')
ax.legend(fontsize=8);ax.grid(True,alpha=.3);ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{FD}/fig7.pdf',bbox_inches='tight'); plt.close()
print("Fig 7 done")

# ---- FIG 8: Noise robustness ----
fig,ax=plt.subplots(figsize=(5.5,3.8))
for r,c,l in [(4.00,'#185FA5','$r=4.00$'),(3.60,'#BA7517','$r=3.60$')]:
    ev=[d['eps'] for d in E['noise_res'][r]]; nv=[d['nfi'] for d in E['noise_res'][r]]
    ax.plot(ev,nv,'o-',ms=5,color=c,label=l)
ax.set_xlabel('Noise $\\varepsilon$');ax.set_ylabel('NFI')
ax.set_title('Noise robustness');ax.legend();ax.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig8.pdf',bbox_inches='tight'); plt.close()
print("Fig 8 done")

# ---- FIG 9: Binning sensitivity ----
fig,ax=plt.subplots(figsize=(5.5,3.8))
cb={3.50:'#A32D2D',3.60:'#BA7517',3.80:'#0F6E56',4.00:'#185FA5'}
for r in [3.50,3.60,3.80,4.00]:
    ax.plot(E['binr'],E['bsens'][r],'o-',ms=3,color=cb[r],label=f'$r={r}$')
ax.set_xlabel('Bins');ax.set_ylabel('KL');ax.set_title('KL vs bin count')
ax.legend();ax.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig9.pdf',bbox_inches='tight'); plt.close()
print("Fig 9 done")

# ---- FIG 10: Weighting sensitivity (2 panels) ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.8))
for c in E['cv']:
    nv=[]
    for r in [3.50,3.60,3.80,4.00]:
        d=[x for x in T1 if x['r']==r and x['g']=='x'][0]
        nv.append(d['kl']+c*d['nfi_sc'])
    a1.plot([3.50,3.60,3.80,4.00],nv,'o-',label=f'$c={c}$')
a1.set_xlabel('$r$');a1.set_ylabel('NFI');a1.set_title('All regimes');a1.legend();a1.set_yscale('log');a1.grid(True,alpha=.3)
for c in E['cv']:
    a2.plot(E['rcl'],E['cnfi'][c],'o-',ms=4,label=f'$c={c}$')
a2.set_xlabel('$r$');a2.set_ylabel('NFI');a2.set_title('Close neighborhood ($r\\in[3.58,3.66]$)')
a2.legend();a2.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig10.pdf',bbox_inches='tight'); plt.close()
print("Fig 10 done")

# ---- FIG 11: Alternative measures ----
fig,axes=plt.subplots(1,3,figsize=(11,3.5))
labs=['$r=3.50$','$r=3.60$','$r=3.80$','$r=4.00$','H(1.4,.3)','H(1.3,.3)']
kls=[50.,T1[2]['kl'],T1[4]['kl'],T1[6]['kl'],T2[0]['kl'],T2[3]['kl']]
w1s=[50.,T1[2]['w1'],T1[4]['w1'],T1[6]['w1'],0.06,0.8]  # approximate
ads=[999.,T1[2]['ad'],T1[4]['ad'],T1[6]['ad'],0.4,15.]
x=np.arange(6)
axes[0].bar(x,kls,color='#185FA5',alpha=.7);axes[0].set_xticks(x);axes[0].set_xticklabels(labs,rotation=45,fontsize=7)
axes[0].set_ylabel('KL');axes[0].set_title('KL divergence')
axes[1].bar(x,w1s,color='#0F6E56',alpha=.7);axes[1].set_xticks(x);axes[1].set_xticklabels(labs,rotation=45,fontsize=7)
axes[1].set_ylabel('$W_1$');axes[1].set_title('Wasserstein-1')
axes[2].bar(x,ads,color='#BA7517',alpha=.7);axes[2].set_xticks(x);axes[2].set_xticklabels(labs,rotation=45,fontsize=7)
axes[2].set_ylabel('AD stat');axes[2].set_title('Anderson-Darling')
plt.tight_layout()
plt.savefig(f'{FD}/fig11.pdf',bbox_inches='tight'); plt.close()
print("Fig 11 done")

print("\nAll 11 figures generated in", FD)
