# train_stage2.py  – Stage-2: LoRA による LLM 部の微調整
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from mol_blip2_opt import MolBLIP2OPT

class MolTextPair(Dataset):
    def __init__(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        self.smiles = df.smiles.tolist()
        self.text = df.text.tolist()
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx): return self.smiles[idx], self.text[idx]

def collate_fn(batch):
    s, t = zip(*batch)
    return list(s), list(t)

def main(csv_path="train.csv", stage1_ckpt="stage1_ep2.pt", epochs=3, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # モデル準備（Stage-1 学習済の qformer + proj）
    model = MolBLIP2OPT(device=device)
    model.load_state_dict(torch.load(stage1_ckpt, map_location="cpu"), strict=False)
    
    # LLM 側に LoRA 注入（q_proj, v_proj）
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.llm = prepare_model_for_kbit_training(model.llm)
    model.llm = get_peft_model(model.llm, lora_cfg)
    model.llm.print_trainable_parameters()

    # モデル設定
    model.qformer.eval()   # QFormer は固定
    model.proj.eval()      # 射影も固定
    model.llm.train()

    optimizer = torch.optim.AdamW(model.llm.parameters(), lr=lr)
    dataset = MolTextPair(csv_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for ep in range(epochs):
        for step, (smiles, texts) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(smiles, texts[0])  # prompt = ground-truth
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"[ep {ep} step {step}] loss: {loss.item():.4f}")
        # 保存
        model.llm.save_pretrained(f"stage2_ep{ep}_llm_lora")
        torch.save(model.state_dict(), f"stage2_ep{ep}_full.pt")

if __name__ == "__main__":
    main()
