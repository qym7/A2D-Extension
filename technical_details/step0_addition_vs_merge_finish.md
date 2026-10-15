# Dream MATH: addition > merge 只是因为写得完（2026-10-10，零 GPU）

MATH-500 前 200 题，@1024，lockstep 单独跑（`outputs/step0/lockstep1024/math500/*-free`）。写完 = 文本含 `\boxed`。

| 模型 | acc | 写完题数 | 写完的题里 acc |
|---|---:|---:|---:|
| Ins | 44.5 | 146 | 61.0 |
| M(.4,.6) best merge | 53.5 | 156 | **68.6** |
| M(0,1) over merge | 48.0 | 149 | 64.4 |
| A(1.1) best addition | 57.0 | 191 | 59.7 |
| A(2.0) over addition | 58.5 | 189 | 61.9 |

三者都写完的 150 题：M(.4,.6) 68.7，A(1.1) 64.0，A(2.0) 66.0。
结论：addition 的总 acc 优势来自 1024 token 内写完的题更多，不是解题更准；同样写完时 merge 更好。
注意：merge 是 B300、addition 是 H200·0.51；写完与否和题目难度相关（不是随机缺失）。
