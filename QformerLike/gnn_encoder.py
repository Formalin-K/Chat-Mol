

# ==== molecular vision encoder (L統一)====
import torch, torch.nn as nn
import dgl
from dgllife.model import load_pretrained
from dgllife.utils import (
    smiles_to_bigraph,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MolecularGIN(nn.Module):
    """
    GIN (ContextPred, hidden=300) ➜ proj(300→768) ➜
    Query‑Pooling (learnable K=256) で
        [CLS] + 256 patch  = 257 tokens, すべて 768 dim
    を出力する分子版 Vision Encoder。
    """

    def __init__(
        self,
        hidden_out: int = 768,
        num_patch: int = 256,
        num_heads: int = 8,
        freeze_gnn: bool = True,
    ):
        super().__init__()
        # 1) 凍結済み GIN encoder
        self.gnn = load_pretrained("gin_supervised_contextpred").to(_DEVICE)  # H=300
        if freeze_gnn:
            for p in self.gnn.parameters():
                p.requires_grad_(False)

        # 2) Featurizers
        self.atom_f = PretrainAtomFeaturizer()
        self.bond_f = PretrainBondFeaturizer()

        # 3) 300 → 768 射影
        self.proj = nn.Linear(300, hidden_out, bias=False)

        # 4) Query‑Pooling (= Set Transformer の PMA 相当)
        self.num_patch = num_patch
        self.query_tok = nn.Parameter(torch.randn(num_patch, hidden_out))
        self.mha       = nn.MultiheadAttention(hidden_out, num_heads, batch_first=True)

        # 5) CLS token
        self.cls_tok = nn.Parameter(torch.randn(1, hidden_out))

    @torch.no_grad()
    def _graph2node(self, smiles_list: list[str]):
        """SMILES → ノード埋め込み (各分子長さ可変)"""
        graphs = [
            smiles_to_bigraph(
                s,
                node_featurizer=self.atom_f,
                edge_featurizer=self.bond_f,
                add_self_loop=True,
            )
            for s in smiles_list
        ]
        bg = dgl.batch(graphs).to(_DEVICE)

        # raw feats (fwd に渡さないとエラー)
        nfeat = [bg.ndata.pop("atomic_number"), bg.ndata.pop("chirality_type")]
        efeat = [bg.edata.pop("bond_type"), bg.edata.pop("bond_direction_type")]

        node_repr = self.gnn(bg, nfeat, efeat)                  # [ΣNi, 300]
        node_repr = self.proj(node_repr)                        # [ΣNi, 768]

        # ── split back by molecule ──
        num_nodes = bg.batch_num_nodes().tolist()               # python list[B]
        out = torch.split(node_repr, num_nodes, dim=0)          # B 個 (Ni,768)
        return out                                              # list[Tensor]

    def forward(self, smiles_list: list[str]):
        """
        Parameters
        ----------
        smiles_list : list[str]   (len = B)

        Returns
        -------
        feats : torch.FloatTensor  [B, 1+K, 768]   (CLS + patch)
        attn  : torch.BoolTensor   [B, 1+K]        (すべて True)
        """
        B, K = len(smiles_list), self.num_patch
        node_embeds = self._graph2node(smiles_list)             # list[(Ni,768)]

        out_tokens = torch.empty(B, 1 + K, self.proj.out_features, device=_DEVICE)

        for i, x in enumerate(node_embeds):
            # --- CLS = mean pooling ---
            cls = x.mean(dim=0, keepdim=True)                   # (1,768)

            # --- patch tokens via MHA ---
            q = self.query_tok.unsqueeze(0)                     # (1,K,768)
            patches, _ = self.mha(q, x.unsqueeze(0), x.unsqueeze(0))  # (1,K,768)

            out_tokens[i] = torch.cat([cls, patches.squeeze(0)], dim=0)

        attn_mask = torch.ones(B, 1 + K, dtype=torch.bool, device=_DEVICE)
        return out_tokens, attn_mask





