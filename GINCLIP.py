%%writefile train_clip_molgin_staged.py
#!/usr/bin/env python
# =========================================================
# train_clip_molgin_staged.py
# =========================================================
"""
2‑Stage CLIP training:
  Stage‑1  1 M SMILES‑IUPAC ペア (pretrain)
  Stage‑2  26 k 高品質キャプション (finetune)
  最終評価 MolT5/ChEBI‑20 validation.txt

依存:
  torch>=2.2.1+cu121, dgl, dgllife, rdkit-pypi, transformers 4.40

Colab 例:
  !python train_clip_molgin_staged.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math, warnings, argparse
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from rdkit import Chem
import dgl
from dgllife.utils import (
    smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
)
from transformers import (
    AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
)

# ---------- Dataset ----------
class MolCaptionTSV(Dataset):
    def __init__(self, tsv, augment=False, n_aug=1):
        df = pd.read_csv(tsv, sep="\t",
                         names=["cid","smiles","caption"],
                         dtype=str)
        df = df[df.smiles.apply(lambda s: Chem.MolFromSmiles(s) is not None)]
        self.smi, self.txt = df.smiles.tolist(), df.caption.tolist()
        self.augment, self.n_aug = augment, n_aug
        self.atom_f = PretrainAtomFeaturizer(); self.bond_f = PretrainBondFeaturizer()
    def __len__(self): return len(self.smi) * (self.n_aug if self.augment else 1)
    def __getitem__(self, idx):
        i = idx % len(self.smi)
        smi = self.smi[i]
        if self.augment:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),
                                   doRandom=True, canonical=False)
        return {"smiles": smi, "caption": self.txt[i]}

def collate(batch, atom_f, bond_f):
    graphs, caps = [], []
    for b in batch:
        try:
            g = smiles_to_bigraph(
                b["smiles"], add_self_loop=True,
                node_featurizer=atom_f, edge_featurizer=bond_f,
                canonical_atom_order=False)
            graphs.append(g); caps.append(b["caption"])
        except: pass
    return (dgl.batch(graphs) if graphs else None), caps

# ---------- Encoders ----------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MolecularGIN(nn.Module):
    def __init__(self, out_dim=512, num_patch=31, heads=8, layers=3):
        super().__init__()
        from dgllife.model import load_pretrained
        self.gnn = load_pretrained("gin_supervised_contextpred").to(_DEVICE)
        self.proj = nn.Linear(300, out_dim, bias=False)
        self.query_tok = nn.Parameter(torch.randn(num_patch, out_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.num_patch = num_patch
    @torch.no_grad()
    def _encode_nodes(self, bg):
        nfeat = [bg.ndata.pop("atomic_number"), bg.ndata.pop("chirality_type")]
        efeat = [bg.edata.pop("bond_type"), bg.edata.pop("bond_direction_type")]
        return self.gnn(bg, nfeat, efeat)
    def forward(self, bg):
        bg = bg.to(_DEVICE)
        x = self.proj(self._encode_nodes(bg))
        sizes = bg.batch_num_nodes().tolist()
        B, D = len(sizes), x.size(-1)
        out = torch.empty(B, 1+self.num_patch, D, device=_DEVICE)
        ptr = 0
        for i, n in enumerate(sizes):
            nodes = x[ptr:ptr+n]; ptr += n
            cls = nodes.mean(0, keepdim=True)
            toks = torch.cat([cls, self.query_tok], 0).unsqueeze(0)  # (1,1+P,D)
            out[i] = self.encoder(toks)[0]
        return out[:,0]  # (B,D)

class TextEncoder(nn.Module):
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2", out_dim=512):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.bert = AutoModel.from_pretrained(model_id).to(_DEVICE)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim, bias=False)
    def forward(self, caps):
        tok = self.tok(caps, padding=True, truncation=True,
                       return_tensors="pt").to(_DEVICE)
        emb = self.bert(**tok).last_hidden_state[:,0]
        return self.proj(emb)

class CLIPMol(nn.Module):
    def __init__(self, dim=512, init_temp=0.07):
        super().__init__()
        self.vision = MolecularGIN(out_dim=dim)
        self.text   = TextEncoder(out_dim=dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/init_temp)), requires_grad=True)
    def forward(self, bg, caps):
        v = self.vision(bg); t = self.text(caps)
        v = v / v.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * v @ t.t()
        gold   = torch.arange(len(v), device=_DEVICE)
        loss_i = nn.functional.cross_entropy(logits,   gold)
        loss_t = nn.functional.cross_entropy(logits.t(), gold)
        return (loss_i + loss_t) / 2, v.detach(), t.detach()

# ---------- Train / Eval ----------
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval(); v_all, t_all = [], []
    for bg, caps in val_loader:
        _, v, t = model(bg, caps)
        v_all.append(v); t_all.append(t)
    v_emb = torch.cat(v_all); t_emb = torch.cat(t_all)
    sims = v_emb @ t_emb.t()
    top1 = sims.topk(1, dim=1).indices.squeeze()
    gold = torch.arange(len(sims), device=_DEVICE)
    r1   = (top1 == gold).float().mean().item()
    top5 = sims.topk(5, dim=1).indices
    r5   = (top5 == gold.unsqueeze(1)).any(1).float().mean().item()
    return r1, r5

def run_stage(model, dataset_path, args, stage, epochs, lr_v, lr_t):
    ds = MolCaptionTSV(dataset_path, augment=True, n_aug=args.n_aug)
    coll = lambda b: collate(b, ds.atom_f, ds.bond_f)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=True,
                     num_workers=2, collate_fn=coll, drop_last=True)

    opt = torch.optim.AdamW([
        {"params": model.vision.parameters(), "lr": lr_v},
        {"params": model.text.parameters(),   "lr": lr_t},
        {"params": [model.logit_scale],       "lr": lr_v}
    ], weight_decay=1e-2)
    total = epochs * len(dl) // args.accum
    sch = get_cosine_schedule_with_warmup(opt, int(0.05*total), total)

    g_step = 0
    for ep in range(1, epochs+1):
        model.train()
        for ib, (bg, caps) in enumerate(dl):
            loss, *_ = model(bg, caps)
            (loss/args.accum).backward()
            if (ib+1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(); g_step += 1
                if g_step % 200 == 0:
                    print(f"[{stage}] step {g_step}/{total}  loss {loss.item():.4f}")
        print(f"★ [{stage}] epoch {ep}/{epochs}  last‑loss {loss.item():.4f}")

def main():
    args = argparse.Namespace(
        pretrain="/content/drive/MyDrive/clip_smiles_caption_1M.tsv",
        trainhq="MolT5/ChEBI-20_data/train.txt",
        val_path="MolT5/ChEBI-20_data/validation.txt",
        out_dir="./clip_ckpt",
        batch=32, accum=8, n_aug=2,
        dim=512,
        lr_pre=3e-4, lr_text_pre=6e-5, ep_pre=3,
        lr_ft=1e-4, lr_text_ft=2e-5, ep_ft=10
    )

    model = CLIPMol(dim=args.dim).to(_DEVICE)

    print("\n======= Stage‑1: Pre‑train on 1 M IUPAC =======")
    run_stage(model, args.pretrain, args, "pretrain",
              args.ep_pre, args.lr_pre, args.lr_text_pre)

    print("\n======= Stage‑2: Fine‑tune on 26 k HQ =======")
    run_stage(model, args.trainhq, args, "finetune",
              args.ep_ft, args.lr_ft, args.lr_text_ft)

    # ---- Evaluation ----
    val_ds = MolCaptionTSV(args.val_path, augment=False)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False,
                    num_workers=2,
                    collate_fn=lambda b: collate(b, val_ds.atom_f, val_ds.bond_f))
    r1, r5 = evaluate(model, vl)
    print(f"\n✅ Final  r@1 = {r1:.3f}   r@5 = {r5:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out_dir}/clip_molgin_final.pt")
    print(f"📝 Saved model to {args.out_dir}/clip_molgin_final.pt")

if __name__ == "__main__":
    main()
