# TinyLogicLM

一个 toy 项目: 训练一个小的 **decoder-only** Transformer 去"执行"一门极简符号语言,
然后分析模型内部**什么时候**表示出 condition 真假 / branch 选择 / 最终答案。

## V1 语言

```
if (A op B) log(C) else log(D)      op ∈ { >, < }
```

`A B C D ∈ 0..99`。cond 为真输出 C,否则输出 D。

- **离散 token**:语法符号各占一个 id,数字用 `NUM_OFFSET+n`(一个数一个 token)。
  不用裸 0-9 / 0-1,避免把"学分词"混进"学逻辑"。
- **decoder-only causal LM**:每个位置预测下一个 token,loss 只算 `ANSWER` 之后的部分
  (等价于 SFT 的 prompt/completion)。
- softmax 出现在两处:注意力内部 `softmax(QKᵀ/√d)`,输出层 cross-entropy(内含 log_softmax)。

## 用到的概念全部来自 Illustrated Transformer

模型刻意只用 <https://jalammar.github.io/illustrated-transformer> 里出现过的组件,不加复杂技巧:

| 文章里的概念 | 代码位置 |
|---|---|
| 词向量 Embedding | `token_emb` |
| 位置向量(加到词向量上) | `pos_emb`(用可学习向量,概念同"每个位置一个向量") |
| Self-Attention: Q/K/V + 缩放点积 + softmax | `CausalSelfAttention` |
| Multi-Head Attention | `n_heads` 拆头 |
| Add & Normalize(残差 + LayerNorm,放子层后) | `TransformerBlock`(Post-LN) |
| 前馈网络 Linear→ReLU→Linear | `TransformerBlock.mlp` |
| 最后 Linear + Softmax 出词 | `lm_head` + cross-entropy |

优化器用朴素 **Adam**(不加 weight decay 等文章没讲的正则)。模型很小
(**12.7 万参数**,d=64 / 2 层 / 2 头),既好训练也好做可解释性。

## 用法

```bash
pip install -r requirements.txt

# 1. 看看数据长啥样 (Day 1 sanity check)
python -m tiny_logic.data -n 8
python -m tiny_logic.data -n 8 --trace       # condition 显式吐出的 trace 版

# 2. 训练 (默认超参已调到能可靠"学会比较", CPU/MPS 几分钟)
python -m tiny_logic.train                       # 默认: 25 epoch / 20万条 / lr 1e-3
python -m tiny_logic.train --holdout-pair        # 训练时真排除一块区域, 做泛化测试
python -m tiny_logic.train --trace               # trace 版, 供 logit lens 对照

# 3. 严格评估 (含"保证没见过"的题 + 过拟合 gap)
python -m tiny_logic.eval --ckpt checkpoints/tiny_logic_holdout.pt

# 4. 可解释性图 (probe / logit lens / attention)
python -m tiny_logic.inspect --ckpt checkpoints/tiny_logic_holdout.pt
```

## V1 结果(已验证,没过拟合)

- **训练出现 grokking 相变**:前几个 epoch 卡在 acc≈0.50(只会瞎猜 C/D),
  第 5 个 epoch 左右**突然**学会比较,跳到 0.95+,最终 val_acc≈0.99。
- **不是记忆,是泛化**(判定标准:测试题的 `(A,B,op,C,D)` 没在训练集完整出现过):

  | 测试口径 | 准确率 |
  |---|---|
  | train-seen(训练见过的题) | 0.998 |
  | **novel-tuple(保证从没见过)** | **0.992** |
  | holdout-pair(整块排除的区域) | 0.983 |
  | 过拟合 gap = train − novel | **+0.006 ≈ 0** |

  训练 20 万条里 199907 道互不相同,模型在严格没见过的题上仍 99%,train/test 差 0.6% → 学到的是算法不是记忆。目标 80%,远超达标。
- **内部机制**(见 `runs/` 三张图):`emb` 层读不出条件(≈0.5)→ **L0 算出 condition**
  (线性可读 0.99)→ **L1 把答案拍板**(logit lens 0.99),ANSWER 位有一个把答案数字拷过来的 copy head。

## 路线图

| 版本 | 输入形式 | 目标 | 状态 |
|---|---|---|---|
| **V1** | 语法 token + `NUM_n` | 学 if-else 执行 | ✅ 本仓库 |
| V1.5 | + `{}` 嵌套 | 学 branch skipping / 深度 | 预留 flag |
| V2 | 语法 token + digit(0-9) | 学数字结构 / 泛化 | 计划 |
| V3 | 纯 decimal digit 流 | 学 token 边界 + 执行 | 计划 |
| V4 | 纯 binary bit 流 | 学 bit 编码 + 执行 | stretch |

## 可解释性实验 (V1 跑通后)

- **linear probe**:每层 / 关键 token 位置的 hidden state → cond true/false,看哪层最先"知道"。
- **logit lens**:中间层 hidden 直接过 `lm_head`,看 top token 何时变成正确答案。
- **direct vs trace**:direct 版把 cond 压进 hidden(用探针读),trace 版显式生成(用 logit lens 看)。

研究问题写在 `notebooks/` 里逐个验证。
