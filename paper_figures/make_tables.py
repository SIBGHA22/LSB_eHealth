import sys,pandas as pd,numpy as np
from scipy import stats
# usage: python make_tables.py <results_dir> <output_dir>
RES,OUT=sys.argv[1:3]
a=pd.read_csv(f'{RES}/ecg_robustness/runs_pass1.csv')
b=pd.read_csv(f'{RES}/ecg_robustness/runs_pass2.csv')
r=pd.concat([a,b]).drop_duplicates(['arm','attack','seed'],keep='last')
arms=['FEDAVG_CLIP','MEDIAN','TRIMMED_MEAN','MULTI_KRUM','FLTRUST','LSB','LSB_V2']
lab={'FEDAVG_CLIP':'Clipped FedAvg','MEDIAN':'Coord.\\ median','TRIMMED_MEAN':'Trimmed mean','MULTI_KRUM':'Multi-Krum','FLTRUST':'FLTrust','LSB':'\\sysname{} (v1)','LSB_V2':'\\textbf{\\sysname{}}'}
att=[('NONE','Clean'),('SIGNFLIP6','Sign flip $\\times(-6)$'),('NORM_MATCHED_FLIP','Norm-matched'),('ALIE','ALIE'),('IPM','IPM'),('LABEL_FLIP30','Label flip')]
def ci(v): return stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
L=[]
L.append(r"\begin{table}[!htbp]\centering")
L.append(r"\caption{ECG robustness (PhysioNet/CinC 2017, ten clients, two adversarial). Macro-F1, mean $\pm$ \SI{95}{\percent} CI over six paired seeds, with the worst seed in brackets. Bold: highest mean in each condition. All values favourable upward.}")
L.append(r"\label{tab:ecg}\scriptsize\setlength{\tabcolsep}{2.2pt}")
L.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}l"+"c"*6+r"@{}}\toprule")
L.append(r"\textbf{Aggregation} & "+" & ".join(f"\\textbf{{{t}}}" for _,t in att)+r" \\\midrule")
best={k:max(arms,key=lambda x:r[(r.attack==k)&(r.arm==x)].macro_f1.mean()) for k,_ in att}
for x in arms:
    cells=[]
    for k,_ in att:
        v=r[(r.attack==k)&(r.arm==x)].macro_f1.values
        m=f"{v.mean():.3f}"; m=f"\\textbf{{{m}}}" if best[k]==x else m
        cells.append(f"{m}$\\pm${ci(v):.3f} ({v.min():.3f})")
    if x=='LSB_V2': L.append(r"\midrule")
    L.append(lab[x]+" & "+" & ".join(cells)+r" \\")
L.append(r"\bottomrule\end{tabular}}\end{table}")
L.append("")
# significance
L.append(r"\begin{table}[!htbp]\centering")
L.append(r"\caption{Paired comparison of \sysname{} against each baseline, macro-F1. Each cell: mean difference (\sysname{} minus baseline), seeds won out of six, and two-sided Wilcoxon signed-rank $p$. With six pairs the smallest attainable $p$ is 0.031. Bold: $p<0.05$.}")
L.append(r"\label{tab:sig}\scriptsize\setlength{\tabcolsep}{2.5pt}")
L.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}l"+"c"*6+r"@{}}\toprule")
L.append(r"\textbf{Baseline} & "+" & ".join(f"\\textbf{{{t}}}" for _,t in att)+r" \\\midrule")
for x in arms[:-1]:
    cells=[]
    for k,_ in att:
        A=r[(r.attack==k)&(r.arm=='LSB_V2')].set_index('seed').macro_f1
        B=r[(r.attack==k)&(r.arm==x)].set_index('seed').macro_f1.loc[A.index]
        d=A-B; p=stats.wilcoxon(A,B).pvalue; w=int((d>0).sum())
        c=f"{d.mean():+.3f}, {w}/6, {p:.3f}"
        cells.append(f"\\textbf{{{c}}}" if p<0.05 else c)
    L.append(lab[x]+" & "+" & ".join(cells)+r" \\")
L.append(r"\bottomrule\end{tabular}}\end{table}")
L.append("")
# exclusion
L.append(r"\begin{table}[!htbp]\centering")
L.append(r"\caption{Screening behaviour of the rules that select clients: share of adversarial updates excluded / share of honest updates excluded, averaged over rounds and seeds.}")
L.append(r"\label{tab:excl}\footnotesize\setlength{\tabcolsep}{3pt}")
L.append(r"\begin{tabular}{@{}l"+"c"*6+r"@{}}\toprule")
L.append(r"\textbf{Rule} & "+" & ".join(f"\\textbf{{{t}}}" for _,t in att)+r" \\\midrule")
for x in ['MULTI_KRUM','FLTRUST','LSB','LSB_V2']:
    cells=[]
    for k,_ in att:
        q=r[(r.attack==k)&(r.arm==x)]
        av=q.adv_exclusion_rate.mean(); hv=q.honest_exclusion_rate.mean()
        cells.append(("--" if np.isnan(av) else f"{100*av:.0f}\\%")+" / "+f"{100*hv:.0f}\\%")
    L.append(lab[x]+" & "+" & ".join(cells)+r" \\")
L.append(r"\bottomrule\end{tabular}\end{table}")
txt="\n".join(L).replace(' -0.',' $-$0.')
open(f'{OUT}/ecg_tables.tex','w').write(txt)
# docker table
s=pd.read_csv(f'{RES}/consensus_testbed/summary.csv'); s=s[s.delay_ms==5]
def g(n,sc,c,m,col):
    q=s[(s.nodes==n)&(s.scenario==sc)&(s.concurrency==c)&(s['mode']==m)].iloc[0]; return q[col],q.get(col+'_ci95',np.nan)
rows=[('Median latency, 1 client (ms)','normal',1,'latency_ms_median',2,True),
('95th pct latency, 1 client (ms)','normal',1,'latency_ms_p95',2,False),
('Median latency, 16 clients (ms)','normal',16,'latency_ms_median',1,True),
('Throughput, 16 clients (tx/s) $\\uparrow$','normal',16,'throughput_tx_s',1,True),
('Messages per transaction','normal',1,'messages_per_tx',1,False),
('CPU per transaction (ms)','normal',1,'cpu_ms_per_tx',2,False),
('Unsafe commits per 1000 tx, 30\\% Byzantine','byzantine30',1,'unsafe_commits',1,True),
('False rejects per 1000 tx, 30\\% Byzantine','byzantine30',1,'false_rejects',1,True),
('95th pct latency, 10\\% crashed (ms)','crash10',1,'latency_ms_p95',0,False),
('Timeouts per 1000 tx, 10\\% crashed','crash10',1,'timeouts',1,False)]
T=[r"\begin{table}[!htbp]\centering",
r"\caption{Consensus-layer comparison on Docker validator networks with \SI{5}{\milli\second} links. Mean over five seeds; $\pm$ gives the \SI{95}{\percent} CI where it is informative. Arrows: $\uparrow$ higher is better; all other rows lower is better. Committee size: DBH-Chain 5 ($N=20$) and 6 ($N=50$); \sysname{} 7.}",
r"\label{tab:consensus}\footnotesize\setlength{\tabcolsep}{3pt}",
r"\begin{tabular}{@{}lcccc@{}}\toprule",
r" & \multicolumn{2}{c}{\textbf{$N=20$}} & \multicolumn{2}{c}{\textbf{$N=50$}} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
r"\textbf{Metric} & DBH-Chain & \sysname{} & DBH-Chain & \sysname{} \\\midrule"]
for name,sc,c,col,dp,showci in rows:
    cells=[]
    for n in (20,50):
        for m in ('dbh','lsb'):
            v,h=g(n,sc,c,m,col); txt=f"{v:.{dp}f}"
            if showci and not np.isnan(h): txt+=f"$\\pm${h:.{dp}f}"
            cells.append(txt)
    T.append(name+" & "+" & ".join(cells)+r" \\")
T.append(r"\bottomrule\end{tabular}\end{table}")
open(f'{OUT}/consensus_table.tex','w').write("\n".join(T))
print(open(f'{OUT}/ecg_tables.tex').read()); print(open(f'{OUT}/consensus_table.tex').read())
