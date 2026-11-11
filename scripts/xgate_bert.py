"""Single cross-encoder with reliability token + token-level evidence gating (CH-SIMS v2, official split).

Input: [CLS] transcript [SEP] [REL] Chinese-AU-evidence [SEP]   (bert-base-chinese, one pass -> token-level cross-attention)
  (a) [REL] embedding = word_emb([unused1]) + MLP_r(r), r = reliability feats (face rate, FaceScore, AU strength, #salient)
  (b) evidence-token gating: g_i = sigmoid(u . e_i + MLP_g(r)) multiplies each evidence token's input embedding
  (c) conflict head on [CLS] -> BCE vs sign(label_V) != sign(label_T)   (aux, weight 0.2)
Variants: zh_concat (plain concat, no REL/gate/aux), xgate (a+b+c), xgate_noaux (a+b).
Chinese evidence = our measured AU Face sentence translated with DEVA's Chinese AU phrases + intensity words.
Usage: python xgate_bert.py --variant xgate --seed 1
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from gate_bert import reliability_feats
from train_bert import BERT, FULL, jl, metrics

AU_ZH = {"AU01": "抬起内眉", "AU02": "抬起外眉", "AU04": "低眉", "AU05": "抬起上眼睑", "AU06": "抬起脸颊",
         "AU07": "紧闭眼皮", "AU09": "皱鼻头", "AU10": "扬起上唇", "AU12": "拉唇角", "AU14": "收紧嘴角(酒窝)",
         "AU15": "下压唇角", "AU17": "抬起下巴", "AU20": "拉伸嘴唇", "AU23": "抿紧嘴唇", "AU24": "压紧嘴唇",
         "AU25": "分开嘴唇", "AU26": "掉下巴", "AU28": "嘬嘴", "AU43": "闭眼"}
LV_ZH = {"strong": "强烈的", "moderate": "中等的", "slight": "轻微的"}
AUX_W = 0.2


# semantic ablation: same structure, action words replaced by meaningless codes or by other AUs' real words
_AUS = sorted(AU_ZH)
_GAN, _NUM = "甲乙丙丁戊己庚辛壬癸", "一二三四五六七八九十"
AU_CODE = {au: _GAN[i % 10] + _NUM[i // 10] for i, au in enumerate(_AUS)}
_seed = 7
_perm = list(np.random.default_rng(_seed).permutation(len(_AUS)))
while any(p == i for i, p in enumerate(_perm)):  # derangement: no AU keeps its own word
    _seed += 1
    _perm = list(np.random.default_rng(_seed).permutation(len(_AUS)))
AU_PERM = {au: AU_ZH[_AUS[_perm[i]]] for i, au in enumerate(_AUS)}


def zh_evidence(face_sentence, words="real", keep_id=True):
    """'Face: [F3-F5] strong lip corner puller (AU12); ...' -> '面部动作：[F3-F5] 强烈的拉唇角(AU12)；...'
    words: real | code (meaningless codes) | perm (other AUs' real words); keep_id: keep '(AUxx)'."""
    if not face_sentence or "no salient" in face_sentence:
        return "面部动作：没有明显的表情动作。"
    vocab = {"real": AU_ZH, "code": AU_CODE, "perm": AU_PERM}[words]
    parts = []
    for m in re.finditer(r"\[([^\]]+)\] (strong|moderate|slight) [^(;]*\((AU\d+)\)", face_sentence):
        fr, lv, au = m.groups()
        parts.append(f"[{fr}] {LV_ZH[lv]}{vocab.get(au, au)}" + (f"({au})" if keep_id else ""))
    miss = re.search(r"face not visible in ([F\d,\-]+)", face_sentence)
    if miss:
        parts.append(f"[{miss.group(1)}] 未检测到人脸")
    return "面部动作：" + ("；".join(parts) if parts else "没有明显的表情动作") + "。"


class XGate(nn.Module):
    def __init__(self, variant, n_rel=4):
        super().__init__()
        self.variant = variant
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.rel = nn.Sequential(nn.Linear(n_rel, 128), nn.GELU(), nn.Linear(128, d))
        self.u = nn.Linear(d, 1)
        self.gr = nn.Sequential(nn.Linear(n_rel, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.u.weight); nn.init.constant_(self.u.bias, 2.0)  # start with gates ~0.88 (near plain concat)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))
        self.conf = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, ids, att, tt, ev_mask, rel_pos, r):
        emb = self.enc.embeddings.word_embeddings(ids)
        g = torch.ones(ids.shape, device=ids.device, dtype=emb.dtype)
        if self.variant in ("xgate", "xgate_noaux"):
            b = torch.arange(ids.size(0), device=ids.device)
            emb = emb.clone()
            emb[b, rel_pos] = emb[b, rel_pos] + self.rel(r).to(emb.dtype)  # (a) reliability token
            gi = torch.sigmoid(self.u(emb).squeeze(-1) + self.gr(r).to(emb.dtype))  # (b) token-level gate
            g = torch.where(ev_mask, gi, g)
            emb = emb * g.unsqueeze(-1)
        h = self.enc(inputs_embeds=emb, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1), self.conf(h).squeeze(-1), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["zh_concat", "xgate", "xgate_noaux", "zh_noid", "zh_code_noid", "zh_perm_noid"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    out_dir = os.path.join(FULL, os.environ.get("XGATE_RESULTS_DIR", "results_xgate")); os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(out_dir, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    vis = jl(os.path.join(FULL, "measure_text.jsonl"), "visual_text")
    face = lambda t: (re.search(r"Face: .*?(?= (Head|Gaze): |$)", t) or [""])[0] if t else ""
    words = {"zh_code_noid": "code", "zh_perm_noid": "perm"}.get(a.variant, "real")
    keep_id = not a.variant.endswith("_noid")
    df["evid"] = [zh_evidence(face(vis.get(k, "")), words, keep_id) for k in df.key]
    zp = os.path.join(FULL, "measure_zh.jsonl")
    if a.variant == "zh_concat" and not os.path.exists(zp):
        with open(zp, "w") as f:
            for k, e in zip(df.key, df.evid):
                f.write(json.dumps({"key": k, "zh_evidence": e}, ensure_ascii=False) + "\n")
    R = torch.tensor(reliability_feats(df.key.tolist()))
    CT = torch.tensor((np.sign(df.label_V) != np.sign(df.label_T)).astype(np.float32).values)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(BERT)
    rel_tok = "[unused1]" if a.variant in ("xgate", "xgate_noaux") else ""
    enc = tok(list(df.text), [(rel_tok + e) for e in df.evid], truncation="only_second", max_length=256)
    rel_id = tok.convert_tokens_to_ids("[unused1]")

    def item(i):
        ids, tt = enc["input_ids"][i], enc["token_type_ids"][i]
        ev = [t == 1 and x not in (tok.sep_token_id, rel_id) for x, t in zip(ids, tt)]
        rp = ids.index(rel_id) if rel_id in ids else 0
        return ids, tt, ev, rp

    items = [item(i) for i in range(len(df))]

    def batch(idx):
        L = max(len(items[i][0]) for i in idx)
        pad = lambda seq, v: seq + [v] * (L - len(seq))
        ids = torch.tensor([pad(items[i][0], tok.pad_token_id) for i in idx])
        tt = torch.tensor([pad(items[i][1], 0) for i in idx])
        att = torch.tensor([pad([1] * len(items[i][0]), 0) for i in idx])
        ev = torch.tensor([pad(items[i][2], False) for i in idx])
        rp = torch.tensor([items[i][3] for i in idx])
        return ids.cuda(), att.cuda(), tt.cuda(), ev.cuda(), rp.cuda(), R[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = XGate(a.variant).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx):
        model.eval(); P, C, G = [], [], []
        for i in range(0, len(idx), 64):
            b = batch(idx[i:i + 64])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, c, g = model(*b)
            ev = b[3].float()
            P.append(y.float().cpu()); C.append(c.float().cpu())
            G.append(((g.float() * ev).sum(1) / ev.sum(1).clamp(min=1)).cpu())  # mean gate over evidence tokens
        model.train()
        return torch.cat(P).numpy(), torch.cat(C).numpy(), torch.cat(G).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]; b = batch(j)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, c, _ = model(*b)
            loss = nn.functional.mse_loss(y.float(), Y[j].cuda())
            if a.variant == "xgate":
                loss = loss + AUX_W * nn.functional.binary_cross_entropy_with_logits(c.float(), CT[j].cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        pv, _, _ = predict(sp["valid"])
        vm = metrics(df.label.values[sp["valid"]], pv); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
    model.load_state_dict(state)
    pt, ct, gt = predict(sp["test"]); te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    pd.DataFrame({"key": te.key, "label": te.label, "label_T": te.label_T, "label_V": te.label_V,
                  "category": te.category, "pred": pt, "conflict_logit": ct, "gate": gt}
                 ).to_csv(os.path.join(out_dir, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
