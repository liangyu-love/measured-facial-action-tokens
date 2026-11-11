"""Read-only inference from archived fa2 checkpoint; export numeric maps, no training."""
import sys,json,os
import numpy as np,pandas as pd,torch
sys.path.insert(0,os.path.expanduser('~/work/msa-pilot/scripts'))
from au_soft import build_inputs
from evi_bert import EviModel
from train_bert import FULL,BERT
from transformers import AutoTokenizer
torch.set_num_threads(4)
df=pd.read_csv(os.path.join(FULL,'clips.csv'));fr,_,_=build_inputs(df,'soft')
keys=['video_0051/0032','video_0016/0040','video_0006/0004']
ix=[df.index[df.key==k][0] for k in keys]
tok=AutoTokenizer.from_pretrained(BERT)
model=EviModel('fa2',fr.shape[-1],20).cuda()
model.load_state_dict(torch.load(os.path.join(FULL,'results_evi/fa2_seed1.pt'),map_location='cpu'));model.eval()
te=df.index[df.split=='test'].to_numpy();cache={}
with torch.no_grad():
 for start in range(0,len(te),32):
  idx=te[start:start+32]
  if not set(idx)&set(ix):continue
  enc=tok(df.iloc[idx].text.tolist(),padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   y,S,R,ex=model(enc['input_ids'],enc['attention_mask'],enc['token_type_ids'],torch.tensor(fr[idx]).cuda(),attn=True,post_layers=(8,))
  for j,i in enumerate(idx):
   if i in ix:cache[i]=(float(y[j]),ex['post_L8'][j].float().cpu().reshape(8,20).tolist())
au=pd.read_csv(os.path.join(FULL,'au_frames.csv'));cols=sorted(c for c in au if c.startswith('AU') and c[2:].isdigit())
out=[]
for j,i in enumerate(ix):
 r=df.iloc[i];a=au[au.key==r.key].sort_values('frame');out.append(dict(key=r.key,text=r.text,y=float(r.label),yt=float(r.label_T),yv=float(r.label_V),pred=cache[i][0],weights=cache[i][1],au=a[cols].fillna(0).values.tolist(),cols=cols))
path=os.path.join(FULL,'figure_evidence_export.json');json.dump(out,open(path,'w'),ensure_ascii=False,indent=2);print(path,flush=True)
