"""Evaluate a checkpoint, with an explicit no-overfitting check.

过拟合的判定 (按项目目标):
  测试题的 (A,B,op,C,D) 只要没和训练集完全重合, 答对就算真本事, 不算记忆.
所以这里跑:
  train-seen      : 直接从训练集里抽的题 (上界参考)
  novel-tuple     : 保证 (A,B,op,C,D) 从未在训练集出现的题  <- 主要泛化指标
  holdout-pair    : 训练时整块排除的 A∈[20,29]&B∈[30,39] (结构泛化, 需 --holdout-pair 训练)
过拟合 = train-seen 远高于 novel-tuple. 目标: novel-tuple >= 0.80.
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader, Dataset

from tiny_logic.model import TinyGPT
from tiny_logic.data import (make_example, in_holdout_pair, meta_tuple,
                             training_tuple_set, TinyLogicDataset)
from tiny_logic.train import evaluate
from tiny_logic.vocab import VOCAB_SIZE


class ExampleList(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels, loss_mask, meta = self.examples[idx]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.float), "meta": meta}


def sample_where(n, predicate, trace, seed, ops=("GT", "LT"), answer_mode="copy"):
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 500:
        tries += 1
        ex = make_example(rng, ops=ops, trace=trace, answer_mode=answer_mode)
        if predicate(ex[3]):
            out.append(ex)
    return ExampleList(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tiny_logic.pt")
    p.add_argument("-n", type=int, default=3000)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(vocab_size=VOCAB_SIZE, max_seq_len=32, d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"], n_heads=cfg["n_heads"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    trace = cfg["trace"]
    held = cfg.get("holdout_pair", False)
    amode = cfg.get("answer", "copy")
    reject = in_holdout_pair if held else None

    # 还原训练集见过的所有题目 (A,B,op,C,D)
    train_size = cfg.get("train_size", 200_000)
    train_seed = cfg.get("train_seed", 1)
    print(f"loaded {args.ckpt}  trace={trace}  answer={amode}  holdout_pair_trained={held}")
    print(f"还原训练集题目 ({train_size} 条) ...")
    seen = training_tuple_set(train_size, seed=train_seed, reject=reject)
    print(f"训练集去重后 {len(seen)} 道不同的题\n" + "=" * 66)

    tests = {}
    # 1. 训练里见过的题 (上界)
    tests["train-seen"] = (TinyLogicDataset(args.n, seed=train_seed, reject=reject,
                           answer_mode=amode), "训练集抽样")
    # 2. 保证从未见过的题 -- 主指标
    tests["novel-tuple"] = (sample_where(
        args.n, lambda m: meta_tuple(m) not in seen, trace, seed=100,
        answer_mode=amode), "严格没见过")
    # 3. 结构 holdout (整块 (A,B) 区域排除)
    if held:
        tests["holdout-pair"] = (sample_where(
            args.n, in_holdout_pair, trace, seed=102, answer_mode=amode), "整块排除区域")

    accs = {}
    for name, (ds, kind) in tests.items():
        m = evaluate(model, DataLoader(ds, batch_size=512), device)
        accs[name] = m["acc"]
        print(f"{name:<14} acc={m['acc']:.4f}  loss={m['loss']:.4f}  "
              f"n={len(ds):<5} [{kind}]")

    print("=" * 66)
    gap = accs["train-seen"] - accs["novel-tuple"]
    print(f"过拟合 gap (train-seen - novel-tuple) = {gap:+.4f}  "
          f"(接近0 = 没过拟合)")
    verdict = "达标 ✅" if accs["novel-tuple"] >= 0.80 else "未达标 ❌"
    print(f"novel-tuple = {accs['novel-tuple']:.4f}  vs  目标 0.80  -> {verdict}")


if __name__ == "__main__":
    main()
