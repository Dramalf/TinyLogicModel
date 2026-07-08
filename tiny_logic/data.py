"""V1 dataset: single-level `if (A op B) log(C) else log(D)` programs.

每条样本是一个定长序列 (V1 不需要 padding):
    BOS  IF ( A op B )  LOG ( C )  ELSE  LOG ( D )  ANSWER  <completion>
completion:
    direct 版:  <answer> EOS
    trace  版:  <TRUE|FALSE> THEN <answer> EOS   (condition 显式吐出, 适合 logit lens)

训练是 decoder-only causal LM: 每个位置预测下一个 token, 但 loss 只算 ANSWER 之后的部分.
"""
import argparse
import random

import torch
from torch.utils.data import Dataset

from tiny_logic.vocab import (
    BOS, EOS, IF, ELSE, LOG, LPAREN, RPAREN, GT, LT, ANSWER,
    FALSE, TRUE, THEN, MAX_NUM, VOCAB_SIZE, num_tok, decode,
)

OPS = {"GT": GT, "LT": LT}


def in_holdout_pair(meta) -> bool:
    """真 holdout 区域: A∈[20,29] 且 B∈[30,39]. 训练时排除, 测试时只测这块.
    每个数字单独都见过, 但这个 (A,B) 组合没见过 -- 比 holdout 单个数字更公平的泛化测试."""
    return 20 <= meta["a"] <= 29 and 30 <= meta["b"] <= 39


def make_example(rng: random.Random, max_num=MAX_NUM, ops=("GT", "LT"),
                 trace=False, answer_mode="copy"):
    """生成一条样本. 返回 (input_ids, labels, loss_mask, meta).

    A/B/C/D 全部独立随机, 保证 answer 与位置无固定关系 (防止模型学 shortcut).
    answer_mode:
      "copy"  -> answer = 选中的那个数 (C 或 D), 答案就在输入里, 模型只需复制.
      "plus1" -> answer = (选中的数 + 1) % 100, 答案不在输入里, 模型要算"后继".
    """
    a = rng.randint(0, max_num)
    b = rng.randint(0, max_num)
    c = rng.randint(0, max_num)
    d = rng.randint(0, max_num)
    op_name = rng.choice(ops)
    op = OPS[op_name]
    cond = (a > b) if op == GT else (a < b)
    selected = c if cond else d
    answer = selected if answer_mode == "copy" else (selected + 1) % 100

    program = [
        BOS,
        IF, LPAREN, num_tok(a), op, num_tok(b), RPAREN,
        LOG, LPAREN, num_tok(c), RPAREN,
        ELSE,
        LOG, LPAREN, num_tok(d), RPAREN,
        ANSWER,
    ]
    if trace:
        completion = [TRUE if cond else FALSE, THEN, num_tok(answer), EOS]
    else:
        completion = [num_tok(answer), EOS]

    tokens = program + completion
    input_ids = tokens[:-1]
    labels = tokens[1:]

    # 只在 completion 上算 loss: labels[i] 对应预测 tokens[i+1].
    # ANSWER 在 tokens 里的下标 = ai, labels 从下标 ai 起就是要预测的答案部分.
    ai = tokens.index(ANSWER)
    loss_mask = [1 if i >= ai else 0 for i in range(len(labels))]

    meta = {"a": a, "b": b, "c": c, "d": d, "op": op_name,
            "cond": cond, "selected": selected, "answer": answer}
    return input_ids, labels, loss_mask, meta


def meta_tuple(m):
    """一条样本的"身份": (A,B,op,C,D). 两条样本这五元组相同才算"同一道题"."""
    return (m["a"], m["b"], m["op"], m["c"], m["d"])


def training_tuple_set(train_size, seed=1, reject=None, ops=("GT", "LT")):
    """精确重放训练集用过的所有 (A,B,op,C,D), 供 eval 判断测试题是否真的没见过.
    与 TinyLogicDataset 用同样的 rng 流程, 所以能还原出一模一样的训练题目集合."""
    rng = random.Random(seed)
    seen = set()
    made = 0
    while made < train_size:
        ex = make_example(rng, ops=ops)
        if reject is not None and reject(ex[3]):
            continue
        seen.add(meta_tuple(ex[3]))
        made += 1
    return seen


class TinyLogicDataset(Dataset):
    def __init__(self, n_examples, max_num=MAX_NUM, ops=("GT", "LT"),
                 trace=False, seed=0, reject=None, answer_mode="copy"):
        """reject: 可选 predicate(meta)->bool, 为真则丢弃重采 (用于训练时排除 holdout)."""
        rng = random.Random(seed)
        self.examples = []
        while len(self.examples) < n_examples:
            ex = make_example(rng, max_num, ops, trace, answer_mode)
            if reject is not None and reject(ex[3]):
                continue
            self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels, loss_mask, meta = self.examples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "meta": meta,
        }


def _sanity_check(n=8, trace=False):
    """打印 n 条样本, 确认 program 和 answer 对得上. (Day 1 dataset check)"""
    rng = random.Random(0)
    print(f"VOCAB_SIZE = {VOCAB_SIZE},  trace = {trace}\n" + "=" * 78)
    for _ in range(n):
        input_ids, labels, loss_mask, meta = make_example(rng, trace=trace)
        tokens = input_ids + [labels[-1]]
        ops = ">" if meta["op"] == "GT" else "<"
        print(f"if ({meta['a']} {ops} {meta['b']}) log({meta['c']}) "
              f"else log({meta['d']})  ->  cond={meta['cond']}, answer={meta['answer']}")
        print("  tokens:", decode(tokens))
        print("  loss on:", decode([labels[i] for i in range(len(labels)) if loss_mask[i]]))
        print("-" * 78)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    _sanity_check(args.n, args.trace)
