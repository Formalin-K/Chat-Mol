colabで使うとき
# Colab 起動直後にこのセルだけ走らせてください
!pip -q install "numpy<2"          # まず NumPy 1.x に戻す
import os, sys, json, textwrap
os.kill(os.getpid(), 9)            # Colab のカーネルを再起動

# ==== Colab 先頭インストールセル ====
# PyTorch (CUDA 12.1) ＋ DGL (CUDA 12.1) ＋ dgllife ＋ RDKit
!pip -q install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
!pip -q install -U "dgl" -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html   # cu121 対応 wheel ✔︎ :contentReference[oaicite:0]{index=0}
!pip -q install dgllife==0.3.2 rdkit-pypi
!pip install -U bitsandbytes

↑の後にセルに張り付けてgnn_encoder.py→qformer.py→mol_blip2_opt.py実行すると最後まで動いた。