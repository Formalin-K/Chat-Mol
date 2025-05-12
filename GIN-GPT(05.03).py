# Colab 起動直後にこのセルだけ走らせてください
!pip -q install "numpy<2"          # まず NumPy 1.x に戻す
import os, sys, json, textwrap
os.kill(os.getpid(), 9)            # Colab のカーネルを再起動


###

from google.colab import drive
drive.mount('/content/drive')

###

!pip -q install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
!pip -q install -U dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
!pip -q install dgllife==0.3.2 rdkit-pypi
!pip -q install transformers==4.40.0 bitsandbytes accelerate==0.28.0
!pip -q install sacrebleu rouge-score

###

!git clone https://github.com/blender-nlp/MolT5.git
!apt-get update -y
!apt-get install git-lfs -y
!git lfs install
!git -C MolT5 lfs pull

###

# ============================================================
# train_mini_mol_v3.py
# ============================================================
"""
MiniMolTiny v3
  - TinyLlama 1.1B Chat (完全凍結)
  - GIN ContextPred → MHA + proj + query_tok を学習可
  - 32 個のダミートークン <mol_patch_i>
      * tokenizer に登録するが埋め込みは使用しない
      * トークン列の埋め込みを mol_emb で **上書き** する
  - プロンプト：ChatML 準拠 1 ラウンド
"""

# ------------------------------- 基本設定 -------------------------------
from __future__ import annotations
import os, sys, math, argparse, warnings, random, json
warnings.filterwarnings("ignore")
os.environ["DGLBACKEND"] = "pytorch"
sys.argv = [sys.argv[0]]

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
import dgl
from dgllife.utils import (
    smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup
)
import sacrebleu

# ------------------------------- カスタム設定 -------------------------------
CONFIG = {
    "base_model":          "TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    "use_4bit":            False,          # Colab メモリが厳しければ True
    "train_query_tok":     True,           # MolGIN の query_tok・MHA を学習させる
    "train_token_embed":   False,          # <mol_patch_i> 埋め込み列を学習する
    "batch":               4,
    "epochs":              1,
    "lr":                  5e-4,
    "weight_decay":        1e-2,
    "eval_batches":        20,
    "train_tsv":           "MolT5/ChEBI-20_data/train.txt",
    "val_tsv":             "MolT5/ChEBI-20_data/validation.txt",
    "output_dir":          "./checkpoints_v3",
    "seed":                42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])

# ------------------------------- Dataset -------------------------------
class MolCaptionTSV(Dataset):
    def __init__(self, tsv: str, augment: bool = False):
        df = pd.read_csv(tsv, sep="\t",
                         names=["cid", "smiles", "caption"],
                         dtype=str)
        df = df[df.smiles.apply(lambda s: Chem.MolFromSmiles(s) is not None)]
        self.smi, self.txt = df.smiles.tolist(), df.caption.tolist()
        self.augment = augment
        self.atom_f, self.bond_f = PretrainAtomFeaturizer(), PretrainBondFeaturizer()
    def __len__(self): return len(self.smi)
    def __getitem__(self, i):
        smi = self.smi[i]
        if self.augment:
            smi = Chem.MolToSmiles(
                Chem.MolFromSmiles(smi), doRandom=True, canonical=False)
        return {"smiles": smi, "caption": self.txt[i]}

# ------------------------------- プロンプト -------------------------------
MOL_NUM_PATCH = 31
MOL_TOKEN_COUNT = MOL_NUM_PATCH + 1                       # 32
MOL_TOKENS = [f"<mol_patch_{i}>" for i in range(MOL_TOKEN_COUNT)]

CHAT_TMPL = (
    "<|im_start|>user\n"
    "{mol_token_str} [caption] describe this molecule\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def collate(batch, atom_f, bond_f, tok):
    graphs, prompts, labels = [], [], []
    mol_token_str = " ".join(MOL_TOKENS)
    for b in batch:
        try:
            g = smiles_to_bigraph(
                b["smiles"], add_self_loop=True,
                node_featurizer=atom_f, edge_featurizer=bond_f,
                canonical_atom_order=False
            )
            graphs.append(g)
            prompts.append(CHAT_TMPL.format(mol_token_str=mol_token_str))
            labels.append(b["caption"])
        except: pass
    if not graphs:
        return None, [], []
    return dgl.batch(graphs), prompts, labels

# ------------------------------- Molecular GIN -------------------------------
class MolecularGIN(nn.Module):
    def __init__(self, out_dim=2048, num_patch=MOL_NUM_PATCH, heads=8):
        super().__init__()
        from dgllife.model import load_pretrained
        self.gnn = load_pretrained("gin_supervised_contextpred")
        self.proj = nn.Linear(300, out_dim, bias=False)
        self.query_tok = nn.Parameter(torch.randn(num_patch, out_dim))
        self.mha = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.num_patch = num_patch

    @torch.no_grad()
    def _encode_nodes(self, bg):
        nfeat = [bg.ndata.pop("atomic_number"),
                 bg.ndata.pop("chirality_type")]
        efeat = [bg.edata.pop("bond_type"),
                 bg.edata.pop("bond_direction_type")]
        return self.gnn(bg, nfeat, efeat)     # (ΣN,300)

    def forward(self, bg):
        x = self._encode_nodes(bg)            # float32
        x = self.proj(x)                      # (ΣN,D)
        sizes = bg.batch_num_nodes().tolist()
        chunks = torch.split(x, sizes, 0)
        B, D = len(chunks), x.size(-1)
        out = torch.empty(B, 1+self.num_patch, D, device=x.device)
        for i, nodes in enumerate(chunks):
            cls = nodes.mean(0, keepdim=True)             # (1,D)
            patches, _ = self.mha(
                self.query_tok.unsqueeze(0),
                nodes.unsqueeze(0), nodes.unsqueeze(0))
            out[i] = torch.cat([cls, patches.squeeze(0)], 0)
        return out    # (B,32,D)

# ------------------------------- Full Model -------------------------------
class MiniMolTiny(nn.Module):
    def __init__(self):
        super().__init__()
        # ----- tokenizer -----
        self.tok = AutoTokenizer.from_pretrained(
            CONFIG["base_model"], use_fast=False, trust_remote_code=True)
        self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.add_tokens(MOL_TOKENS)

        # ----- LLM -----
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if CONFIG["use_4bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.float16
        self.llm = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"], **kwargs)
        self.llm.resize_token_embeddings(len(self.tok))

        # ----- freeze／unfreeze -----
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if CONFIG["train_token_embed"]:
            # 追加した行のみ学習可能に
            emb = self.llm.get_input_embeddings().weight
            emb[-MOL_TOKEN_COUNT:].requires_grad_(True)

        # ----- Vision -----
        D = self.llm.get_input_embeddings().embedding_dim
        self.vision = MolecularGIN(out_dim=D).to(self.llm.device)
        # 学習可否
        for name, p in self.vision.named_parameters():
            if name.startswith("proj"):
                p.requires_grad_(True)
            elif name.startswith(("mha", "query_tok")):
                p.requires_grad_(CONFIG["train_query_tok"])
            else:
                p.requires_grad_(False)

        self.to(DEVICE)

    # --------- helper ---------
    def encode_mol(self, bg):
        bg = bg.to(self.device)
        vis = self.vision(bg)             # (B,32,D) float32
        return vis.to(torch.float16)      # LLM と合わせる

    @property
    def device(self):
        return next(self.llm.parameters()).device

    # --------- forward ---------
    def forward(self, bg, prompts, labels=None, gen_cfg=None):
        B = bg.batch_size
        mol_emb = self.encode_mol(bg)                 # (B,32,D)

        tok = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        txt_emb = self.llm.get_input_embeddings()(tok.input_ids)  # (B,L,D)
        assert txt_emb.size(1) >= MOL_TOKEN_COUNT, "プロンプトが短すぎます"
        # ------- **** overwrite **** -------
        txt_emb[:, :MOL_TOKEN_COUNT, :] = mol_emb

        if labels is None:                            # generation
            gc = dict(max_new_tokens=64, top_p=0.9, temperature=0.7,
                      do_sample=True, eos_token_id=32002,
                      pad_token_id=self.tok.eos_token_id)
            if gen_cfg: gc.update(gen_cfg)
            out_ids = self.llm.generate(
                inputs_embeds=txt_emb,
                attention_mask=tok.attention_mask,
                **gc)
            return self.tok.batch_decode(out_ids, skip_special_tokens=True)

        # ------- training -------
        lbl = self.tok(labels, return_tensors="pt", padding=True).to(self.device)
        lbl_emb = self.llm.get_input_embeddings()(lbl.input_ids)
        # overwrite 先頭 32 トークンは “Mol トークン” なのでラベル -100
        all_emb  = torch.cat([txt_emb, lbl_emb], 1)
        all_mask = torch.cat([tok.attention_mask, lbl.attention_mask], 1)
        seq = all_emb.size(1)
        lbl_full = torch.full((B, seq), -100, dtype=torch.long, device=self.device)
        lbl_full[:, txt_emb.size(1):] = lbl.input_ids
        out = self.llm(inputs_embeds=all_emb,
                       attention_mask=all_mask,
                       labels=lbl_full)
        return out.loss

# ------------------------------- Fast BLEU -------------------------------
@torch.no_grad()
def fast_bleu(model, dataloader, max_batches=20):
    model.eval()
    hyps, refs = [], []
    for i, (bg, p, l) in enumerate(dataloader):
        outs = model(bg, p, gen_cfg=dict(max_new_tokens=48))
        hyps += [o.strip() for o in outs]
        refs += [[r.strip()] for r in l]
        if i+1 >= max_batches: break
    return sacrebleu.corpus_bleu(hyps, refs).score

# ------------------------------- Train -------------------------------
def train():
    train_ds = MolCaptionTSV(CONFIG["train_tsv"], augment=True)
    val_ds   = MolCaptionTSV(CONFIG["val_tsv"],   augment=False)
    coll = lambda b: collate(b, train_ds.atom_f, train_ds.bond_f, tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"], use_fast=False, trust_remote_code=True)
    tokenizer.add_tokens(MOL_TOKENS)

    tl = DataLoader(train_ds, batch_size=CONFIG["batch"], shuffle=True,
                    num_workers=2, collate_fn=coll, drop_last=True)
    vl = DataLoader(val_ds,   batch_size=CONFIG["batch"], shuffle=False,
                    num_workers=2, collate_fn=coll, drop_last=False)

    model = MiniMolTiny()
    # 学習対象パラメータを抽出
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    total_steps = CONFIG["epochs"] * len(tl)
    sch = get_cosine_schedule_with_warmup(
        opt, int(0.1*total_steps), total_steps)

    g_step = 0
    for epoch in range(1, CONFIG["epochs"]+1):
        model.train()
        for bg, p, l in tl:
            loss = model(bg, p, l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            g_step += 1
            if g_step % 50 == 0:
                print(f"step {g_step}/{total_steps}  loss {loss.item():.4f}")

        # -------- Validation --------
        model.eval(); v_loss, n = 0.0, 0
        with torch.no_grad():
            for bg, p, l in vl:
                v_loss += model(bg, p, l).item(); n += 1
        ppl = math.exp(v_loss/n)
        bleu = fast_bleu(model, vl, max_batches=CONFIG["eval_batches"])
        print(f"[epoch {epoch}]  val_loss {v_loss/n:.3f}  PPL {ppl:.1f}  BLEU {bleu:.1f}")

        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        ck = os.path.join(CONFIG["output_dir"], f"fastbleu_epoch{epoch}.pt")
        torch.save(model.state_dict(), ck)
        print("✓ saved", ck)

# ------------------------------- Main -------------------------------
if __name__ == "__main__":
    train()

###




# --- 推論用コード ---
# モデル読み込みと推論プロンプト修正
model = MiniMolTiny()
ckpt_path = "checkpoints_v3/fastbleu_epoch1.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=model.device))
model.eval()

example_smi = "c1ccccc1O"
mol_token_str = " ".join(MOL_TOKENS)
prompt_template = (
    f"<|im_start|>user\n{mol_token_str} [caption] describe this molecule\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)

mol = Chem.MolFromSmiles(example_smi)
g = smiles_to_bigraph(
    example_smi, add_self_loop=True,
    node_featurizer=PretrainAtomFeaturizer(),
    edge_featurizer=PretrainBondFeaturizer(),
    canonical_atom_order=False
)
batched_graph = dgl.batch([g])

with torch.no_grad():
    output = model(batched_graph, [prompt_template])
    print("=== Generated Caption ===")
    print(output[0])