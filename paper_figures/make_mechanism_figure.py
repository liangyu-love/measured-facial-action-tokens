"""Mechanism and representation comparison; all empirical values read from archives.
Attention blocks are a schematic of standard BERT self-attention, not a fitted map.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

P=Path(__file__).resolve().parents[1]; O=P/'figures'; F=P.parent/'results/outputs/full'
INK='#233745'; MUTED='#6F7E87'; LINE='#C8D2D8'; TEAL='#007D78'; BLUE='#417FA0'; GOLD='#BB853B'; PURPLE='#8068A0'
plt.rcParams.update({'font.family':'Arial','font.size':16,'text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
fig=plt.figure(figsize=(10.8,8.0));ax=fig.add_axes([.008,.012,.984,.976]);ax.set(xlim=(0,140),ylim=(0,104));ax.axis('off')
def text(x,y,s,fs=16,c=INK,bold=False,ha='center',va='center',**kw):
 return ax.text(x,y,s,fontsize=fs,color=c,fontweight='bold' if bold else 'normal',ha=ha,va=va,zorder=6,**kw)
def rect(x,y,w,h,fc='white',ec=LINE,r=.7,lw=1.0,**kw):
 q=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=1,**kw);ax.add_patch(q);return q
def arrow(points,c=INK,lw=1.3,style='-|>',ls='-'):
 if len(points)>2:
  xs,ys=zip(*points[:-1]);ax.plot(xs,ys,color=c,lw=lw,ls=ls,zorder=4)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle=style,mutation_scale=12,color=c,lw=lw,ls=ls,zorder=4))
def line(points,c=LINE,lw=1,ls='-'):
 xs,ys=zip(*points);ax.plot(xs,ys,color=c,lw=lw,ls=ls,zorder=4)
def metric(folder,v):
 vals=[]
 for seed in range(1,6):
  d=pd.read_csv(F/folder/f'{v}_seed{seed}_test_pred.csv');vals.append(float(((d.label>0)==(d.pred>0)).mean()*100))
 return np.mean(vals),np.std(vals,ddof=1)
mt,st=metric('results','tmpl_au');ma,sa=metric('results_ausoft','soft');me,se=metric('results_acc','soft_emo')
d=json.loads((P/'audit/figure_evidence_export.json').read_text(encoding='utf8'))[2];U=np.array(d['au']);row=3

text(2,100.4,'Continuous facial evidence, inside the language encoder',21,bold=True,ha='left')
text(2,95.8,'Same AU source · alternative input representations · separate model runs',15,c=MUTED,ha='left')

# Common measured source. One actual row is enlarged into the proposed route.
rect(2,48,23,41,'#F8FAFB','#D7E0E5')
text(13.5,85.8,'Py-Feat',17,bold=True)
for j,k in enumerate([1,4,8]):
 im=np.asarray(Image.open(O/f'assets/video_0006__0004/F{k}.jpg'))
 cols=np.flatnonzero(im.max(axis=(0,2))>20);im=im[:,cols[0]:cols[-1]+1]
 xx=4+j*6.4
 ax.imshow(im,extent=(xx,xx+5.8,71.5,81.4),aspect='auto',zorder=3)
text(13.5,68.6,'8 sampled frames',14.5,c=MUTED)
cm=LinearSegmentedColormap.from_list('AU',['#F4F8FA',BLUE])
ax.imshow(U,extent=(4,23,51.5,65.5),aspect='auto',cmap=cm,vmin=0,vmax=1,interpolation='nearest',zorder=3)
for rr in range(1,8):line([(4,51.5+rr*1.75),(23,51.5+rr*1.75)],'white',.6)
yy=65.5-(row+1)*1.75
ax.add_patch(Rectangle((3.7,yy),19.6,1.75,fill=False,ec=TEAL,lw=1.5,zorder=4))
text(13.5,45.6,'20 AU × 8 frames',14.5,c=BLUE)
text(13.5,42.3,'Continuous estimates',14,c=MUTED)
arrow([(25.5,yy+.875),(28.4,yy+.875),(28.4,82.7),(32,82.7)],MUTED,1.1)
arrow([(25.5,yy+.875),(32,yy+.875)],TEAL,1.6)

# Compact comparator. It is an alternative route, never a second branch in our model.
rect(32,76,106,14.5,'#F5F6F7','#D6DCE0')
text(34,88.2,'AU → text template',16,bold=True,ha='left',c=MUTED)
text(40.5,81.8,'Threshold\n& rank',14.5,c=MUTED)
arrow([(48,82),(51,82)],MUTED)
rect(51.5,77.7,33,7.5,'white','#C4CCD1',.5)
text(68,81.5,'“strong lip corner\ndepressor (AU15) …”',14.5,c=MUTED)
arrow([(85,82),(89,82)],MUTED)
text(94,82,'Word\ntokens',14.5,c=MUTED)
arrow([(100,82),(103,82)],MUTED)
rect(103.5,78.7,13,6.5,'white','#B6C2CA',.5)
text(110,82,'BERT',14.5,c=MUTED)
text(110,87.2,'+ transcript',12.5,c=GOLD)
arrow([(117,82),(120,82)],MUTED)
text(129,84.0,f'{mt:.2f}',20,c=MUTED,bold=True)
text(129,79.7,f'± {st:.2f}',14,c=MUTED)

# Main representation mechanism, with precise feature and token cardinalities.
rect(32,48,106,24,'#F0F8F5','#9ACCC1',.9,1.3)
text(34,69.0,'AU → continuous soft tokens',17,c=TEAL,bold=True,ha='left')
text(136,69.0,'Preserve magnitude + frame order',14.5,c=TEAL,ha='right')
text(46,63.7,r'Frame $f$: $\mathbf{a}_f=[\mathbf{u}_f;\delta_f]$',15.5)
vals=np.r_[U[row],1.]
for j,v in enumerate(vals):
 color=cm(v) if j<20 else MUTED
 ax.add_patch(Rectangle((34.7+j*1.03,56.1),.96,4.2,fc=color,ec='none',zorder=3))
text(35.0,53.7,'AU04',13.5,c=BLUE,ha='left');text(43.8,53.7,'AU12',13.5,c=BLUE,ha='left');text(54.8,53.7,'face',13.5,c=MUTED)
text(38.2,50.8,f'{U[row,2]:.3f}',14.5,c=BLUE);text(47.0,50.8,f'{U[row,9]:.3f}',14.5,c=BLUE);text(54.8,50.8,'1',14.5,c=MUTED)
arrow([(57,58.2),(61,58.2)],TEAL)
rect(61.5,53.1,23,10.6,'white',TEAL,.7,1.1)
text(73,60.4,'Linear projection',14.5,bold=True)
text(73,56.2,'21 → 768 · LN',15.5)
arrow([(85,58.2),(89,58.2)],TEAL)
ax.add_patch(Circle((91,58.2),1.7,fc='white',ec=TEAL,lw=1.2,zorder=4));text(91,58.2,'+',18,c=TEAL)
text(91,64.5,r'$\mathbf{q}_f$',17,c=TEAL)
arrow([(91,62.6),(91,60.2)],TEAL)
arrow([(93,58.2),(97,58.2)],TEAL)
for k in range(8):
 xx=98+k*4.55
 rect(xx,54.7,3.4,7.3,'white',TEAL,.25,.9)
 for y in np.linspace(55.5,61.2,5):line([(xx+.4,y),(xx+3,y)],'#9CCFC6',.65)
text(116,51.0,r'$\mathbf{e}_1\quad\mathbf{e}_2\quad\cdots\quad\mathbf{e}_8$',16,c=TEAL)

# Enhanced model is explicitly separated from the representation comparison.
rect(32,38.7,106,6.3,'#F8F6FB','#D6CEE3',.6,.9)
text(34,42.0,'Full model',14.5,c=PURPLE,bold=True,ha='left')
text(52,42.0,'+7 emotion probs +5 AU products → 33D',14,c=PURPLE,ha='left')
text(136,42.0,f'{me:.2f} ± {se:.2f}',15,c=PURPLE,bold=True,ha='right')

# A joint token sequence feeds ordinary, bidirectional BERT self-attention.
rect(2,13,23,17,'#FCF6EC','#E5D1AD')
text(13.5,26.4,'Transcript',16,bold=True)
text(13.5,21.2,r'$w_1\quad\cdots\quad w_n$',16,c=GOLD)
text(13.5,16.4,'Word embeddings',14.5,c=GOLD)
arrow([(25.5,21),(28.5,21),(28.5,34.7),(34,34.7)],GOLD)
xx=34.8
for lab,w in [('[CLS]',8),(r'$w_1$',5),(r'$w_i$',5),(r'$\cdots$',4),('[SEP]',8)]:
 rect(xx,32.6,w,4.2,'#F8ECD5',GOLD,.3,.8);text(xx+w/2,34.7,lab,14.5);xx+=w+.8
text(72.8,34.7,'|',16,c=MUTED)
for k in range(8):
 x=75+k*4.4;rect(x,32.6,3.4,4.2,'#DFEFE9',TEAL,.3,.8);text(x+1.7,34.7,str(k+1),12.5,c=TEAL)
arrow([(136,50),(139,50),(139,34.7),(110.8,34.7)],TEAL)

# Mechanism zoom: four attention blocks; no invented heatmap values.
rect(32,1.8,106,28.5,'#F4F8FB','#9CB6C8',.9,1.1)
text(34,27.2,'Inside the shared BERT',17,bold=True,ha='left')
text(136,27.2,'12 layers',14,c=BLUE,ha='right')
arrow([(51,32.2),(51,30.6)],GOLD);arrow([(94,32.2),(94,30.6)],TEAL)
mx,my,mw,mh=41,6,20,16
for x,y,w,h,c,label in [(mx,my+mh/2,mw/2,mh/2,'#F4E4C7','T → T'),(mx+mw/2,my+mh/2,mw/2,mh/2,'#B7DFD4','T → F'),(mx,my,mw/2,mh/2,'#E0EFEC','F → T'),(mx+mw/2,my,mw/2,mh/2,'#E9F1F5','F → F')]:
 ax.add_patch(Rectangle((x,y),w,h,fc=c,ec='white',lw=1,zorder=3));text(x+w/2,y+h/2,label,14,c=TEAL if label=='T → F' else MUTED,bold=label=='T → F')
text(mx+mw/2,23.3,'Keys: text | face',13.5,c=MUTED)
text(37.2,14,'Queries',13.5,c=MUTED,rotation=90)
text(51,3.8,'Schematic attention blocks',11.5,c=MUTED)
arrow([(61.5,18),(66,18)],TEAL,1.5)
text(86.8,22.1,'Each word reads facial evidence',15.5,c=TEAL,bold=True)
text(86.8,17.8,r'$\alpha_i=\mathrm{softmax}(q_iK^\top/\sqrt{d_k})$',17)
text(69,11.4,r'$z_i=$',17,ha='left')
text(85,11.4,r'$\sum_{j\in T}\alpha_{ij}v_j$',17,c=GOLD)
text(96.7,11.4,'+',17)
text(107.4,11.4,r'$\sum_{f=1}^{8}\alpha_{if}v_f$',17,c=TEAL)
text(86.8,5.6,'Residual + normalisation + feed-forward',13.5,c=MUTED)
line([(117.6,4),(117.6,23.4)],'#CBD9E3',.8)
text(128,22.2,'CLS → linear',13.5,c=BLUE)
text(128,16.9,f'{ma:.2f}',23,c=TEAL,bold=True)
text(128,12.4,f'± {sa:.2f}',14,c=TEAL)
text(128,7.6,f'+{ma-mt:.2f} pts',16,c=TEAL,bold=True)
text(128,4.5,'vs. AU text',13,c=MUTED)

text(2,-1.6,'Acc-2 (%) · CH-SIMS v2 test n=1,034 · default recipe · 5-seed mean ± SD',14,c=MUTED,ha='left')
for ext in ['pdf','svg','png']:fig.savefig(O/f'method_overview.{ext}',dpi=400,bbox_inches='tight',pad_inches=.08,facecolor='white')
plt.close(fig)
print(f'Template {mt:.4f}; AU soft {ma:.4f}; gain {ma-mt:.4f}; augmented {me:.4f}')
