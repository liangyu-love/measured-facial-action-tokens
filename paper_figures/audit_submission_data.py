from pathlib import Path
import pandas as pd, numpy as np, json, hashlib
ROOT=Path(__file__).resolve().parents[2]; F=ROOT/'results/outputs/full'; OUT=ROOT/'paper/audit'
clips=pd.read_csv(F/'clips.csv').set_index('key');te=clips[clips.split=='test']
specs=[('10 fps, face crops, 8 tokens','results_dense','crop8'),('10 fps, 8 segment-pooled tokens','results_dense','seg8'),('10 fps, 16 tokens','results_dense','dense16'),('per-video, relative to the video median','results_norm_rel','soft_emo'),('per-video $z$-score','results_norm_z','soft_emo'),('+ head pose / gaze tokens','results_ausoft','soft_hg'),('+ head pose / gaze / prosody tokens','results_ausoft','soft_hga'),('+ one clip-level statistics token','results_acc','soft_stat'),('salient-frame selection','results_ausoft','soft_salient'),('cross-attention pre-fusion','results_acc','xattn_pre'),('per-(frame, AU) tokens + a learned \\texttt{[EVI]} summary token (READ-style)','results_evi','fa_evi'),('AU text template (meaningless codes)','results','zh_code_noid'),('chinese-roberta-wwm-ext~\\cite{cui2021wwm}: text only','results_rob2','text'),('chinese-roberta-wwm-ext: + emotions / EMFACS','results_rob2','soft_emo')]
specs=[(n, 'results_xgate' if v=='zh_code_noid' else f,v) for n,f,v in specs]
rows=[]
for name,folder,variant in specs:
 fs=sorted((F/folder).glob(variant+'_seed*_test_pred.csv')); ms=[]
 for f in fs:
  d=pd.read_csv(f).set_index('key').loc[te.index]; y=d.label.to_numpy();p=d.pred.clip(-1,1).to_numpy();ok=(y>0)==(p>0)
  ms.append([ok.mean()*100,ok[te.category=='conflict'].mean()*100,ok[(te.label_T*te.label_V)<0].mean()*100,np.abs(y-p).mean(),np.corrcoef(y,p)[0,1]])
 if len(fs)!=5: print('MISSING',name,folder,variant,len(fs));continue
 vals=np.mean(ms,axis=0);rows.append(dict(name=name,seeds=len(fs),files=[str(f.relative_to(ROOT)) for f in fs],values=vals.tolist()))
 print(name, ' & '.join(f'{x:.2f}' for x in vals[:3]),f'{vals[3]:.3f} / {vals[4]:.3f}')
(OUT/'appendix_recomputed.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
# Bin means and across-seed SD; same default-recipe predictions as Table bins.
gap=np.abs(te.label_T-te.label_V).to_numpy(); masks=[gap<1e-8,(gap>1e-8)&(gap<=.40001),(gap>.40001)&(gap<=.80001),(gap>.80001)&(gap<=1.20001),gap>1.20001]
tr=clips[clips.split=='train'];coef=np.linalg.lstsq(np.c_[np.ones(len(tr)),tr[['label_T','label_V']]],tr.label,rcond=None)[0];oracle=np.c_[np.ones(len(te)),te[['label_T','label_V']]]@coef
res=[]
for mask in masks:
 r={'n':int(mask.sum()),'oracle':float(((oracle[mask]>0)==(te.label.to_numpy()[mask]>0)).mean()*100)}
 for key,folder,v in [('text','results','text'),('soft','results_acc','soft_emo')]:
  vals=[]
  for seed in range(1,6):
   d=pd.read_csv(F/folder/f'{v}_seed{seed}_test_pred.csv').set_index('key').loc[te.index];vals.append(float(((d.label.to_numpy()[mask]>0)==(d.pred.to_numpy()[mask]>0)).mean()*100))
  r[key]=vals
 res.append(r)
(OUT/'conflict_bins.json').write_text(json.dumps(res,indent=2),encoding='utf8')
print('BINS',[(r['n'],np.mean(r['text']),np.mean(r['soft'])) for r in res])
