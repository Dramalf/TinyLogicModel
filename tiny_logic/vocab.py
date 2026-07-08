"""Token vocabulary for TinyLogicLM V1.

设计原则: 语法符号各占一个独立 token id, 数字用 NUM_OFFSET+n 表示 (一个数一个 token).
不要让模型先学分词 -- 直接给离散的语法/数字 token, 让它专注于学 if-else 执行逻辑.
"""

# --- special / syntax tokens ---
PAD = 0
BOS = 1
EOS = 2
IF = 3
ELSE = 4
LOG = 5
LPAREN = 6
RPAREN = 7
GT = 8          # >
LT = 9          # <
ANSWER = 10
# --- trace-only tokens (仅在 trace 版 completion 里出现) ---
FALSE = 11
TRUE = 12
THEN = 13
# --- 预留给 V1.5 嵌套版的块定界符, V1 用不到 ---
LBRACE = 14     # {
RBRACE = 15     # }

NUM_OFFSET = 100   # NUM_00=100 ... NUM_99=199
MAX_NUM = 99
VOCAB_SIZE = 200

_NAME = {
    PAD: "PAD", BOS: "BOS", EOS: "EOS", IF: "IF", ELSE: "ELSE", LOG: "LOG",
    LPAREN: "(", RPAREN: ")", GT: ">", LT: "<", ANSWER: "ANSWER",
    FALSE: "FALSE", TRUE: "TRUE", THEN: "THEN", LBRACE: "{", RBRACE: "}",
}


def num_tok(n: int) -> int:
    assert 0 <= n <= MAX_NUM, n
    return NUM_OFFSET + n


def tok_num(t: int) -> int:
    return t - NUM_OFFSET


def is_num(t: int) -> bool:
    return t >= NUM_OFFSET


def name(t: int) -> str:
    """把 token id 变成可读字符串, 方便打印/调试."""
    return _NAME[t] if t < NUM_OFFSET else f"N{t - NUM_OFFSET}"


def decode(tokens) -> str:
    return " ".join(name(t) for t in tokens)
