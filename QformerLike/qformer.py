import torch, torch.nn as nn
from transformers import BertConfig, BertModel

class QFormer(nn.Module):
    def __init__(self,
                 num_query_token: int = 32,
                 vision_width: int = 768,
                 cross_attention_freq: int = 2,
                 hidden_size: int = 768):
        super().__init__()
        cfg = BertConfig(
            hidden_size=hidden_size,
            encoder_width=vision_width,          # cross‑attention 用
            num_hidden_layers=12,
            is_decoder=False,
            add_cross_attention=True,
            cross_attention_freq=cross_attention_freq,
        )
        self.bert = BertModel(cfg)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_token, hidden_size)
        )

    def forward(self, vision_embeds, vision_attn_mask):
        """
        vision_embeds : [B, Lv, C]
        vision_attn_mask : [B, Lv]
        return : [B, num_query_token, C]
        """
        B = vision_embeds.size(0)
        queries = self.query_tokens.expand(B, -1, -1)    # [B,Q,C]
        return self.bert(
            inputs_embeds=queries,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_attn_mask,
        ).last_hidden_state                               # [B,Q,C]
