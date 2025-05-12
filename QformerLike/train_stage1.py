import torch, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mol_blip2_opt import MolBLIP2OPT

class MolTextPair(Dataset):
    """CSV: smiles,text の 2 カラム前提"""
    def __init__(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        self.smiles = df.smiles.tolist()
        self.text   = df.text.tolist()
    def __len__(self): return len(self.smiles)
    def __getitem__(self, i):
        return self.smiles[i], self.text[i]

def collate(batch):
    s,t = zip(*batch)
    return list(s), list(t)

def main(csv_path="train.csv", epochs=3, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MolBLIP2OPT(device=device)
    model.qformer.train(); model.proj.train()
    opt = optim.AdamW(model.qformer.parameters(), lr=lr)
    opt.add_param_group({"params": model.proj.parameters()})
    loader = DataLoader(MolTextPair(csv_path), batch_size=8,
                        shuffle=True, collate_fn=collate)
    for ep in range(epochs):
        for step,(smi,txt) in enumerate(loader):
            loss = model(smi, txt[0])  # prompt == ground‑truth text
            loss.backward(); opt.step(); opt.zero_grad()
            if step%100==0: print(f"ep{ep} step{step} loss {loss.item():.4f}")
        torch.save(model.state_dict(), f"stage1_ep{ep}.pt")

if __name__ == "__main__":
    main()
