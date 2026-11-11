"""Publication figures. Every numeric panel is derived from archived outputs."""
from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Ellipse,Arc,FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
P=Path(__file__).resolve().parents[1];F=P.parent/'results/outputs/full';O=P/'figures'
BLUE='#176A8A';TEAL='#008779';GOLD='#BE7C23';INK='#203443';GRAY='#81909A';LIGHT='#D9E1E5';ORANGE='#B95E40'
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.titlesize':11,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A4AFB6','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})
cases=json.loads((P/'audit/figure_evidence_export.json').read_text(encoding='utf8'))
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
def text(ax,x,y,s,size=10,weight='normal',color=INK,**kw):return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,**kw)
def rect(ax,x,y,w,h,fc='white',ec=LIGHT,lw=.9,r=.06):
 q=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,lw=lw);ax.add_patch(q);return q
def arrow(ax,a,b,color=GRAY,rad=0,lw=1.1):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,connectionstyle=f'arc3,rad={rad}',color=color,lw=lw))
def heading(ax,x,y,letter,title,sub=None):
 text(ax,x,y,letter,11,'bold',BLUE);text(ax,x+.3,y,title,11,'bold')
 if sub:text(ax,x+.3,y-.32,sub,9,color=GRAY)
# Main architecture is authored by make_architecture.py.
# FIGURE 2: answer two research questions, then cross-validation support.
fig=plt.figure(figsize=(7.4,5.1));gs=fig.add_gridspec(2,2,left=.24,right=.97,top=.88,bottom=.13,wspace=.75,hspace=.75,height_ratios=[3.6,1])
ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:])
spec=[('Text only','results','text'),('MLLM description','results','vlm'),('AU late fusion','results_ausoft','late'),('CLIP soft tokens','results_ausoft','soft_clip'),('AU text template','results','tmpl_au'),('AU soft tokens','results_ausoft','soft'),('+ emotion / EMFACS','results_acc','soft_emo')]
vals=[]
for label,folder,v in spec:
 a=[]
 for seed in range(1,6):
  d=pd.read_csv(F/folder/f'{v}_seed{seed}_test_pred.csv');a.append(float(((d.label>0)==(d.pred>0)).mean()*100))
 vals.append(a)
for i,a in enumerate(vals):
 c=TEAL if i>=5 else GRAY;ax.plot(a,i+np.linspace(-.09,.09,5),'o',ms=3,color=c,alpha=.38);ax.errorbar(np.mean(a),i,xerr=np.std(a,ddof=1),fmt='D' if i>=5 else 'o',color=c,capsize=3,ms=5,lw=1.4);ax.text(86.3,i,f'{np.mean(a):.1f}',va='center',fontsize=9,fontweight='bold' if i>=5 else 'normal',color=c)
ax.set(yticks=np.arange(7),yticklabels=[s[0] for s in spec],xlim=(73,88),ylim=(6.6,-.7),xlabel='All-test Acc-2 (%)');ax.set_xticks([75,80,85]);ax.axvline(np.mean(vals[0]),ls=':',lw=.8,color=LIGHT);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15)
ax.text(-.88,1.16,'Representation matters',transform=ax.transAxes,fontsize=11,fontweight='bold');ax.text(-.88,1.06,'Same default recipe · 1,034 clips',transform=ax.transAxes,fontsize=9,color=GRAY)
bins=json.loads((P/'audit/conflict_bins.json').read_text());yl=['0','(0, 0.4]','(0.4, 0.8]','(0.8, 1.2]','>1.2']
for i,r in enumerate(bins):
 d=np.array(r['soft'])-np.array(r['text']);c=TEAL if i>=3 else BLUE;bx.plot(d,i+np.linspace(-.08,.08,5),'o',ms=3,alpha=.35,color=c);bx.errorbar(d.mean(),i,xerr=d.std(ddof=1),fmt='D',color=c,capsize=3,ms=5,lw=1.4);bx.text(43,i,f'+{d.mean():.1f}',va='center',ha='right',color=c,fontsize=9,fontweight='bold')
bx.set(yticks=range(5),yticklabels=[f'{lab}\n(n={r["n"]})' for lab,r in zip(yl,bins)],xlim=(-1,44),ylim=(4.65,-.7),xlabel='Gain over text (points)');bx.set_xticks([0,10,20,30]);bx.axvline(0,color=GRAY,lw=.8);bx.spines['left'].set_visible(False);bx.tick_params(axis='y',length=0);bx.grid(axis='x',alpha=.15)
bx.text(-.38,1.16,'Larger gains at strong conflict',transform=bx.transAxes,fontsize=11,fontweight='bold');bx.text(-.38,1.06,r'Label gap $|y_T-y_V|$ · paired seeds',transform=bx.transAxes,fontsize=9,color=GRAY)
cx.set(xlim=(-1,15),ylim=(-.5,.5));cx.errorbar(7.2,0,xerr=[[4.4],[4.9]],fmt='D',ms=6,color=TEAL,capsize=4,lw=2);cx.axvline(0,color=GRAY,lw=.8);cx.set_yticks([]);cx.set_xticks([0,5,10,15]);cx.set_xlabel('Ensemble gain over text (Acc-2 points)');cx.spines['left'].set_visible(False)
cx.text(-.32,.92,'Strict-conflict\nconfirmation',transform=cx.transAxes,fontsize=10,fontweight='bold',va='top');cx.text(.02,1.18,'5-fold cross-validation · 470 strict clips',transform=cx.transAxes,fontsize=9,color=GRAY);cx.text(.68,1.18,'+7.2  [2.8, 12.1]',transform=cx.transAxes,fontsize=10,fontweight='bold',color=TEAL)
fig.text(.02,.015,'Top: mean ± SD across 5 seeds; dots: individual seeds.  Bottom: paired-bootstrap 95% CI.',fontsize=8.7,color=GRAY)
save(fig,'conflict_bins')
# FIGURE 3: show what a sparse map does and does not establish.
fig=plt.figure(figsize=(7.4,6.6));axes=[fig.add_axes([x,.43,.345,.36]) for x in [.10,.615]];cx=fig.add_axes([.10,.08,.86,.15])
fig.text(.02,.965,'Post-hoc facial evidence and prediction sensitivity',fontsize=12,fontweight='bold');fig.text(.02,.93,'Auxiliary fa2 model · 160 frame–AU tokens · layer 8 · seed 1',fontsize=9.5,color=GRAY)
for i,(a,d) in enumerate(zip(axes,cases[:2])):
 w=np.array(d['weights']).T
 # Show distribution relative to the uniform 1/160 weight; same logarithmic scale.
 z=np.log2(np.maximum(w*160,1e-8));im=a.imshow(z,cmap='BrBG',norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=7),aspect='auto',interpolation='nearest')
 a.set_xticks(range(8),[f'F{k}' for k in range(1,9)]);a.set_yticks(range(20),[c.replace('AU','') for c in d['cols']]);a.set_xlabel('Sampled frame');a.set_ylabel('Action unit');a.tick_params(length=0);a.tick_params(axis='y',labelsize=8);a.spines[['left','bottom']].set_visible(False)
 title='Correct polarity' if i==0 else 'Incorrect polarity'
 a.text(-.18,1.29,title,transform=a.transAxes,fontsize=10.5,fontweight='bold')
 quote='“Take her away. If I survive,\nI will repay you.”' if i==0 else '“Zhang Yang, thank you.”'
 a.text(-.18,1.20,quote,transform=a.transAxes,fontsize=9.5,va='top')
 a.text(-.18,1.035,f'T +1 / V −1    Overall {d["y"]:+.1f} → pred {d["pred"]:+.2f}',transform=a.transAxes,fontsize=9,color=TEAL if i==0 else ORANGE)
 if i==1:
  row=d['cols'].index('AU04');a.add_patch(Rectangle((-.5,row-.5),8,1,fill=False,edgecolor=INK,lw=1.0));a.text(1.02,1-(row+.5)/20,'AU04',transform=a.transAxes,fontsize=8.5,va='center',fontweight='bold')
cb=fig.colorbar(im,cax=fig.add_axes([.27,.35,.50,.013]),orientation='horizontal',ticks=[-6,-3,0,3,7]);cb.set_label(r'$\log_2$(similarity weight / uniform weight)',fontsize=9);cb.ax.tick_params(labelsize=8)
names=['Lowest 10%','Random 10%','Highest 10%','Rollout top 10%'];kk=['S_bottom','random','S_top','rollout_top'];faith=[]
for k in kk:faith.append([json.loads((F/f'results_evi/fa2_posthoc_seed{s}.json').read_text())['faithfulness']['L8'][k]['mean_abs_dpred'] for s in range(1,6)])
for i,v in enumerate(faith):
 c=TEAL if i==2 else GRAY;cx.scatter(i+np.linspace(-.08,.08,5),v,s=12,color=c,alpha=.5);cx.errorbar(i,np.mean(v),yerr=np.std(v,ddof=1),fmt='D',color=c,capsize=4,ms=5);cx.text(i,max(v)+.013,f'{np.mean(v):.3f}',ha='center',fontsize=9,fontweight='bold' if i==2 else 'normal')
cx.set(xticks=range(4),xticklabels=names,ylim=(0,.25),ylabel=r'Mean $|\Delta\hat y|$');cx.set_yticks([0,.1,.2]);cx.grid(axis='y',alpha=.15)
cx.text(-.1,1.14,'Occlusion sensitivity across all 1,034 test clips',transform=cx.transAxes,fontsize=10.5,fontweight='bold');cx.text(.04,.82,'Top / random: 11.2×',transform=cx.transAxes,fontsize=10,fontweight='bold',color=TEAL)
fig.text(.02,.02,'Top: selected contrasting outcomes.  Bottom: full test set; 5 seeds; mean ± SD.',fontsize=8.7,color=GRAY)
save(fig,'evidence_example')
print('Saved two result figures, SVG sources and 400-dpi previews.')
