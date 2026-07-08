"""Train TinyGPT on the V1 dataset (decoder-only causal LM, masked loss)."""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tiny_logic.data import TinyLogicDataset, in_holdout_pair
from tiny_logic.model import TinyGPT
from tiny_logic.vocab import VOCAB_SIZE, ANSWER


def masked_lm_loss(logits, labels, loss_mask):
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T),
                           reduction="none").view(B, T)
    loss = loss * loss_mask
    return loss.sum() / loss_mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 loss 和 answer-token accuracy (只看 ANSWER 后第一个预测)."""
    model.eval()
    total = correct = steps = 0
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        logits = model(input_ids)
        total_loss += masked_lm_loss(logits, labels, loss_mask).item()
        steps += 1

        # loss_mask 第一个 1 的位置 = 预测 answer 的位置
        ans_pos = loss_mask.argmax(dim=1)
        rows = torch.arange(input_ids.size(0), device=device)
        pred = logits[rows, ans_pos].argmax(dim=-1)
        gold = labels[rows, ans_pos]
        correct += (pred == gold).sum().item()
        total += input_ids.size(0)
    return {"loss": total_loss / steps, "acc": correct / total}


def main():
    p = argparse.ArgumentParser()
    # 默认值调到能可靠"学会比较"的档位 (小 lr / 少数据学不出条件, 会卡在 0.5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=200_000)
    p.add_argument("--val-size", type=int, default=5_000)
    p.add_argument("--trace", action="store_true", help="用 trace 版 completion")
    p.add_argument("--answer", choices=["copy", "plus1"], default="copy",
                   help="copy: 答案=选中的数(可复制); plus1: 答案=(选中的数+1)%%100(要算后继)")
    p.add_argument("--holdout-pair", action="store_true",
                   help="训练时排除 A∈[20,29]&B∈[30,39], 让 eval 的 holdout-pair 成为真泛化测试")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--out", default="checkpoints/tiny_logic.pt")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device = {device}")

    reject = in_holdout_pair if args.holdout_pair else None
    if reject:
        print("训练排除 holdout-pair 区域 (A∈[20,29] & B∈[30,39])")
    TRAIN_SEED = 1
    print(f"answer_mode = {args.answer}")
    train_ds = TinyLogicDataset(args.train_size, trace=args.trace, seed=TRAIN_SEED,
                                reject=reject, answer_mode=args.answer)
    val_ds = TinyLogicDataset(args.val_size, trace=args.trace, seed=2,
                              reject=reject, answer_mode=args.answer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = TinyGPT(vocab_size=VOCAB_SIZE, max_seq_len=32, d_model=args.d_model,
                    n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    print(f"params = {model.num_params():,}")
    # 朴素 Adam, 不加 weight decay -- 保持在 Illustrated Transformer 讲到的概念范围内
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            loss = masked_lm_loss(model(input_ids), labels, loss_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        m = evaluate(model, val_loader, device)
        print(f"epoch {epoch}  train_loss={running/len(train_loader):.4f}  "
              f"val_loss={m['loss']:.4f}  val_acc={m['acc']:.4f}")

    torch.save({"state_dict": model.state_dict(),
                "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                           "n_heads": args.n_heads, "trace": args.trace,
                           "holdout_pair": args.holdout_pair, "answer": args.answer,
                           "train_size": args.train_size, "train_seed": TRAIN_SEED}},
               args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
