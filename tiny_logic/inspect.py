"""可解释性分析: linear probe + logit lens + attention 可视化.

回答三个问题:
  1. 模型什么时候(哪层/哪个 token 位置)线性可读出 condition 真假?  -> probe 热力图
  2. ANSWER 位置上, 从哪一层开始 logit lens 的 top token 就是正确答案?  -> logit-lens 曲线
  3. 注意力头把 ANSWER 位置的信息从哪里搬来?                        -> attention 图

所有图存到 runs/.  用法:  python -m tiny_logic.inspect --ckpt checkpoints/tiny_logic.pt
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_logic.model import TinyGPT
from tiny_logic.data import make_example
from tiny_logic.probe import train_probe
from tiny_logic.vocab import VOCAB_SIZE, ANSWER, name, decode, num_tok

# V1 定长序列里各 token 的位置 (input_ids, 长度 18)
POS_LABELS = ["BOS", "IF", "(", "A", "op", "B", ")", "LOG", "(", "C", ")",
              "ELSE", "LOG", "(", "D", ")", "ANSWER"]
ANSWER_POS = 16


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(vocab_size=VOCAB_SIZE, max_seq_len=32, d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"], n_heads=cfg["n_heads"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def collect(model, device, n=4000, seed=100):
    """跑 n 条样本, 收集每层残差流 + 标签. 返回 acts[layer] -> [N,T,d], metas."""
    rng = random.Random(seed)
    exs = [make_example(rng) for _ in range(n)]
    X = torch.tensor([e[0] for e in exs], device=device)          # [N,18]
    metas = [e[3] for e in exs]
    _, a = model(X, return_activations=True)
    layers = {"emb": a["embedding"].cpu()}
    for i, blk in enumerate(a["blocks"]):
        layers[f"L{i}"] = blk.cpu()
    return layers, metas, a["attn"], X.cpu()


def probe_heatmap(layers, metas, out):
    """每层 x 每位置 训练一个探针预测 cond, 画准确率热力图."""
    cond = torch.tensor([1 if m["cond"] else 0 for m in metas])
    lnames = list(layers.keys())
    T = len(POS_LABELS)
    grid = torch.zeros(len(lnames), T)
    for li, ln in enumerate(lnames):
        H = layers[ln]                                   # [N,T,d]
        for t in range(T):
            acc, _ = train_probe(H[:, t, :], cond, n_classes=2)
            grid[li, t] = acc

    fig, ax = plt.subplots(figsize=(11, 3 + 0.4 * len(lnames)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(T)); ax.set_xticklabels(POS_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(lnames))); ax.set_yticklabels(lnames)
    ax.set_title("Linear probe: is condition (true/false) linearly readable "
                 "from the hidden state?\n(1.0 = perfectly readable, 0.5 = chance)")
    for li in range(len(lnames)):
        for t in range(T):
            ax.text(t, li, f"{grid[li,t]:.2f}", ha="center", va="center",
                    color="white" if grid[li, t] < 0.85 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="probe acc")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return grid


def logit_lens(model, layers, metas, X, out):
    """在 ANSWER 位置, 把每层 hidden 过 ln_f+lm_head, 看 top token 何时=正确答案."""
    answers = torch.tensor([num_tok(m["answer"]) for m in metas])
    lnames = list(layers.keys())
    accs, probs = [], []
    with torch.no_grad():
        for ln in lnames:
            h = layers[ln][:, ANSWER_POS, :].to(next(model.parameters()).device)
            logits = model.lm_head(model.ln_f(h)).cpu()
            p = F.softmax(logits, dim=-1)
            accs.append((logits.argmax(-1) == answers).float().mean().item())
            probs.append(p[torch.arange(len(answers)), answers].mean().item())

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(lnames))
    ax.plot(xs, accs, "o-", label="fraction where top-1 == correct answer")
    ax.plot(xs, probs, "s--", label="mean prob of correct answer")
    ax.set_xticks(list(xs)); ax.set_xticklabels(lnames)
    ax.set_ylim(0, 1.05); ax.set_xlabel("layer read out (at ANSWER position)")
    ax.set_title("Logit lens @ ANSWER: which layer does the answer form in?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return accs


def attn_example(attn, X, out):
    """画第 0 条样本每层每头的注意力图."""
    n_layers = len(attn)
    n_heads = attn[0].shape[1]
    toks = [name(t) for t in X[0].tolist()]
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.2 * n_heads, 3.2 * n_layers), squeeze=False)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            A = attn[li][0, hi].cpu()
            ax.imshow(A, cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"L{li} H{hi}", fontsize=9)
            ax.set_xticks(range(len(toks))); ax.set_yticks(range(len(toks)))
            ax.set_xticklabels(toks, rotation=90, fontsize=5)
            ax.set_yticklabels(toks, fontsize=5)
    fig.suptitle(f"Attention (example 0): {decode(X[0].tolist())}", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tiny_logic.pt")
    p.add_argument("-n", type=int, default=4000)
    p.add_argument("--outdir", default="runs")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}  trace={cfg['trace']}  device={device}")

    layers, metas, attn, X = collect(model, device, n=args.n)

    print("\n[1/3] linear probe (cond 真假) ...")
    grid = probe_heatmap(layers, metas, f"{args.outdir}/probe_cond.png")
    # 打印每层最容易读出 cond 的位置
    lnames = list(layers.keys())
    for li, ln in enumerate(lnames):
        t = int(grid[li].argmax())
        print(f"  {ln:>4}: 最强位置 = {POS_LABELS[t]:<7} acc={grid[li,t]:.3f}")

    print("\n[2/3] logit lens @ ANSWER ...")
    accs = logit_lens(model, layers, metas, X, f"{args.outdir}/logit_lens.png")
    for ln, a in zip(lnames, accs):
        print(f"  {ln:>4}: top1=answer 比例 = {a:.3f}")

    print("\n[3/3] attention 图 (样本0) ...")
    attn_example(attn, X, f"{args.outdir}/attention.png")

    print(f"\n图已存到 {args.outdir}/: probe_cond.png, logit_lens.png, attention.png")


if __name__ == "__main__":
    main()
