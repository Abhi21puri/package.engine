import sys, json
sys.path.insert(0,'/home/claude/v3')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from model import *
for f in fm.findSystemFonts():
    if 'Carlito' in f: fm.fontManager.addfont(f)
plt.rcParams['font.family']='Carlito'
INK='#1B2A3A'; BODY='#3A4A5E'

# ---------- recompute safe maps on a grid that includes the stress-test point (45 C, 2.5C) ----------
amb=[35,40,45,50]; crates=[0.5,1.0,1.5,2.0,2.5,3.0]
REC=dict(series=4,par=1,plate=True,fan=True,derate_start=53.0)
maps={}
for name,d in [("Baseline",DESIGNS['Baseline']),("Recommended",REC)]:
    M=np.zeros((len(crates),len(amb)))
    for i,c in enumerate(crates):
        for j,a in enumerate(amb):
            prof=[(480,6.0*c),(1320,3.0)] if c>0.5 else [(1800,3.0)]
            rng=np.random.default_rng(3); pp=sample_params(300,rng)
            o=simulate(d,pp,prof,T_amb_duty=float(a),charge=False)
            hl=((o['Tpeak']<55)&(o['Ah_del']/o['Ah_req']>=0.95)).mean()
            sf=(o['Tpeak']<60).mean()
            M[i,j]=2 if hl>=0.9 else (1 if sf>=0.95 else 0)
    maps[name]=M
json.dump({k:v.tolist() for k,v in maps.items()},open('maps_final.json','w'))
for k,M in maps.items():
    print(k); [print(c,list(r)) for c,r in zip(crates,M)]

# ---------- Figure 1: temperature curves ----------
curves=json.load(open('curves.json'))
t=np.arange(0,1800,10)/60.0
fig,ax=plt.subplots(figsize=(3.5,2.0),dpi=300)
col={'Baseline':'#C0392B','A: plate':'#8A9BA8','B: fan+BMS':'#8E6BBF','C: 4S2P':'#2E9E6B','Recommended':'#E86A1C'}
lab={'Baseline':'Baseline','A: plate':'A: plate','B: fan+BMS':'B: fan + BMS','C: 4S2P':'C: 2× cells','Recommended':'Recommended'}
for k in ['Baseline','Recommended']:
    ax.fill_between(t,curves[k]['p5'],curves[k]['p95'],color=col[k],alpha=0.16,lw=0)
for k in ['A: plate','B: fan+BMS','C: 4S2P','Baseline','Recommended']:
    ax.plot(t,curves[k]['med'],color=col[k],lw=2.2 if k in('Baseline','Recommended') else 1.3,label=lab[k])
ax.axhline(60,color='#C0392B',ls='--',lw=1); ax.axhline(55,color='#D99A1E',ls=':',lw=1.2)
ax.text(30,60.25,'60 °C datasheet limit',ha='right',va='bottom',fontsize=7,color='#C0392B')
ax.text(30,55.25,'55 °C: life drops (datasheet test point)',ha='right',va='bottom',fontsize=7,color='#9A6B00')
ax.set_xlim(0,30); ax.set_ylim(41.0,62)
ax.set_xlabel('minutes',fontsize=7.5,color=BODY); ax.set_ylabel('hottest cell (°C)',fontsize=7.5,color=BODY)
ax.tick_params(labelsize=7,colors=BODY,length=2)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.legend(fontsize=6,frameon=False,loc='lower right',ncol=3,bbox_to_anchor=(1.0,0.0),handlelength=1.4,columnspacing=0.9)
ax.set_title('Hottest cell on a 45 °C day (band = 90% of runs)',fontsize=8,color=INK,loc='left',fontweight='bold')
fig.tight_layout(pad=0.6); fig.savefig('fig_temp.png',transparent=False,facecolor='white'); plt.close(fig)

# ---------- Figure 2: safe-zone maps ----------
cmap=ListedColormap(['#E4817A','#F4C766','#7CC99A'])
fig,axs=plt.subplots(1,2,figsize=(2.85,2.0),dpi=300,sharey=True)
for ax,(name,M) in zip(axs,maps.items()):
    ax.imshow(M,origin='lower',cmap=cmap,vmin=0,vmax=2,aspect='auto')
    ax.set_xticks(range(len(amb))); ax.set_xticklabels(amb,fontsize=6.5,color=BODY)
    ax.set_yticks(range(len(crates))); ax.set_yticklabels([f'{c:g}C' for c in crates],fontsize=6.5,color=BODY)
    ax.set_title(name,fontsize=8,color=INK,fontweight='bold')
    ax.set_xlabel('outside °C',fontsize=7,color=BODY)
    ax.set_xticks(np.arange(-.5,len(amb),1),minor=True); ax.set_yticks(np.arange(-.5,len(crates),1),minor=True)
    ax.grid(which='minor',color='white',lw=1.2); ax.tick_params(which='both',length=0)
    # mark the stress test (45 C, 2.5C)
    ax.add_patch(plt.Rectangle((2-.5,4-.5),1,1,fill=False,ec=INK,lw=1.6))
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel('peak load (C-rate)',fontsize=7,color=BODY)
fig.tight_layout(pad=0.4); fig.savefig('fig_map.png',facecolor='white'); plt.close(fig)
print('figures done')
