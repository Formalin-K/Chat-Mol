import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from gnn_encoder import MoleculeGIN          # ★ here

from qformer import QFormer

class MolBLIP2OPT(nn.Module):
    def __init__(
        self,
        llm_id: str = "facebook/opt-2.7b",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)

        # 1) GIN Encoder
        self.gnn = MoleculeGIN().to(self.device)                   # out_dim=768

        # 2) Q‑Former
        self.qformer = QFormer().to(self.device)

        # 3) OPT (8‑bit)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=False)
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            device_map="auto",
            quantization_config=quant_cfg,
            torch_dtype=torch.float16,
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # 4) Projection 768 → OPT embed
        llm_dim = self.llm.get_input_embeddings().embedding_dim
        self.proj = nn.Linear(768, llm_dim, bias=False).to(self.device)

    # ---------- 以下、前回と同じ ----------
    def encode_mol(self, smiles):
        vis, mask = self.gnn(smiles)                   # [B,L,768], [B,L]
        q = self.qformer(vis, mask)                    # [B,Q,768]
        return self.proj(q.float())                    # [B,Q,LLMdim]

    def forward(
        self,
        smiles: list[str],
        prompt: str | list[str],
        labels: list[str] | None = None,
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        temperature: float = 0.7,
    ):
        B = len(smiles)
        prefix = self.encode_mol(smiles).to(torch.float16)         # [B,Q,D]

        if isinstance(prompt, str):
            prompt = [prompt] * B
        tok = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        txt_emb = self.llm.get_input_embeddings()(tok["input_ids"])

        inputs_embeds = torch.cat([prefix, txt_emb], dim=1)
        attn_mask = torch.cat(
            [
                torch.ones(
                    B, prefix.size(1), device=self.device, dtype=tok["attention_mask"].dtype
                ),
                tok["attention_mask"],
            ],
            dim=1,
        )

        if self.training:
            out = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=tok["input_ids"],
            )
            return out.loss

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)



