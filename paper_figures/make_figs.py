import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
# usage: python make_figs.py <results_dir> <output_dir>
RES=sys.argv[1]; OUT=sys.argv[2]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e',
 'axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','font.family':'DejaVu Sans'})
BLUE,ORANGE,GRAY,INK='#2a78d6','#eb6834','#9a9993','#0b0b0b'
a=pd.read_csv(f'{RES}/ecg_robustness/runs_pass1.csv')
b=pd.read_csv(f'{RES}/ecg_robustness/runs_pass2.csv')
r=pd.concat([a,b]).drop_duplicates(['arm','attack','seed'],keep='last')
arms=['FEDAVG_CLIP','MEDIAN','TRIMMED_MEAN','MULTI_KRUM','FLTRUST','LSB','LSB_V2']
lab={'FEDAVG_CLIP':'FedAvg (clipped)','MEDIAN':'Coord. median','TRIMMED_MEAN':'Trimmed mean','MULTI_KRUM':'Multi-Krum',
     'FLTRUST':'FLTrust','LSB':'LSB eHealth (v1)','LSB_V2':'LSB eHealth'}
att=[('NONE','Clean'),('SIGNFLIP6','Sign flip (x-6)'),('NORM_MATCHED_FLIP','Norm-matched flip'),
     ('ALIE','ALIE'),('IPM','IPM'),('LABEL_FLIP30','Label flip 30%')]
col={'LSB_V2':BLUE,'LSB':ORANGE}
fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),sharex=True,sharey=True)
for ax,(k,t) in zip(axs.flat,att):
    for i,arm in enumerate(arms):
        v=r[(r.attack==k)&(r.arm==arm)].macro_f1.values; m=v.mean(); h=stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        c=col.get(arm,GRAY); y=len(arms)-1-i
        ax.plot([m-h,m+h],[y,y],color=c,lw=2,solid_capstyle='round')
        ax.plot(m,y,'o',ms=6,color=c,mec='#fcfcfb',mew=1.2,zorder=3)
        ax.plot(v.min(),y,'x',ms=5,color=c,mew=1.4)
    ax.set_title(t,fontsize=9,color=INK,loc='left'); ax.grid(axis='x',color='#e6e5e0',lw=.6); ax.set_axisbelow(True)
    ax.set_xlim(0,1)
for ax in axs[:,0]:
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([lab[a] for a in arms][::-1])
for ax in axs[1]: ax.set_xlabel('Macro-F1 (higher is better)')
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],marker='o',ls='-',color=GRAY),Line2D([],[],marker='x',ls='',color=GRAY)],
 ['mean with 95% CI (6 seeds)','worst seed'],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.55,-.01))
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(f'{OUT}/fig_ecg_robust_baselines.png',dpi=300); plt.close(fig)

# per-seed sign flip
fig,ax=plt.subplots(figsize=(5.6,2.6))
sel=[('FEDAVG_CLIP',GRAY,'s'),('MULTI_KRUM','#52514e','^'),('LSB_V2',BLUE,'o')]
seeds=sorted(r.seed.unique())
for j,(arm,c,mk) in enumerate(sel):
    v=r[(r.attack=='SIGNFLIP6')&(r.arm==arm)].set_index('seed').loc[seeds].macro_f1
    ax.plot(np.arange(len(seeds))+(j-1)*.18,v.values,mk,color=c,ms=7,mec='#fcfcfb',label=lab[arm],ls='')
ax.axhline(0.4789,color='#c3c2b7',lw=1,ls='--'); ax.text(2.5,0.40,'all-normal prediction (F1 = 0.479)',fontsize=7,color='#52514e',ha='center',va='top')
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f'seed {s}' for s in seeds]); ax.set_ylabel('Macro-F1'); ax.set_ylim(0,1)
ax.grid(axis='y',color='#e6e5e0',lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.0),fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_ecg_signflip_per_seed.png',dpi=300); plt.close(fig)

# docker
s=pd.read_csv(f'{RES}/consensus_testbed/summary.csv'); s=s[s.delay_ms==5]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.5))
modes=[('dbh','DBH-Chain PoV',ORANGE),('lsb','LSB eHealth',BLUE)]
def bars(ax,q,metric,ci,yl,title):
    for j,(m,l,c) in enumerate(modes):
        d=q[q['mode']==m].set_index('nodes').loc[[20,50]]
        x=np.arange(2)+(j-.5)*.36
        ax.bar(x,d[metric],.34,color=c,label=l,edgecolor='#fcfcfb',linewidth=1)
        lo=np.minimum(d[ci].values,d[metric].values); ax.errorbar(x,d[metric],yerr=[lo,d[ci].values],fmt='none',ecolor=INK,elinewidth=.8,capsize=2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['20 nodes','50 nodes']); ax.set_ylabel(yl); ax.set_title(title,fontsize=9,loc='left')
    ax.grid(axis='y',color='#e6e5e0',lw=.6); ax.set_axisbelow(True)
n16=s[(s.scenario=='normal')&(s.concurrency==16)]
n1=s[(s.scenario=='normal')&(s.concurrency==1)]
bz=s[(s.scenario=='byzantine30')]
bars(axs[0],n1,'latency_ms_median','latency_ms_median_ci95','ms','(a) Median latency, 1 client')
bars(axs[1],n16,'throughput_tx_s','throughput_tx_s_ci95','tx/s','(b) Throughput, 16 clients')
bars(axs[2],bz,'unsafe_commits','unsafe_commits_ci95','per 1000 tx','(c) Unsafe commits, 30% Byz.')
axs[0].legend(frameon=False,fontsize=7,loc='lower left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_consensus_testbed.png',dpi=300); plt.close(fig)
print('ok')
