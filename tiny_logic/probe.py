"""Linear probe: 用一个逻辑回归探针问"某层某位置的 hidden state 里,
能不能线性读出 condition 的真假 / 最终答案"。

刻意不引入 sklearn -- 直接用 torch 写一个极小的 logistic / softmax 回归,
训练几百步即可, 依赖更少.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    return (X - mu) / sd, mu, sd


def train_probe(X, y, n_classes=2, epochs=300, lr=0.05, val_frac=0.2, seed=0):
    """X:[N,d] float, y:[N] long. 返回 (val_acc, probe_state). 内部做标准化+留出验证."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    perm = torch.randperm(N, generator=g)
    X, y = X[perm], y[perm]
    n_val = int(N * val_frac)
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    Xtr, mu, sd = _standardize(Xtr)
    Xva, _, _ = _standardize(Xva, mu, sd)

    clf = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xva).argmax(-1) == yva).float().mean().item()
    return acc, {"w": clf.weight.detach(), "b": clf.bias.detach(),
                 "mu": mu, "sd": sd}
