# 写法约定（2026-10-10 起，idea 1 所有新表格和回复都用这个）

## 模型：一律写成 (τ_D, τ_A) 系数，前缀区分线
| 写法 | 含义 | 旧名 / checkpoint |
|---|---|---|
| **Ins** | Dream-v0-Instruct = (1, 0) | parent, `parent-free` |
| **M(.4,.6)** | convex 线，best merge | mrg0406, best merge, `dream-ins-04a_D-06a_A` |
| **M(.2,.8)** | convex 线 | `merge_l08` |
| **M(0,1)** | convex 线终点，over merge（τ_D=0） | l10, `merge_l10` |
| **A(1.1)** | addition 线 (1.0, 1.1)，best addition | add1.1, `add-a10-b11` |
| **A(2.0)** | addition 线 (1.0, 2.0) | add2.0, `add-a10-b20` |

M(a,b) = D_b + a·τ_D + b·τ_A 且 a+b=1；A(β) = D_b + 1.0·τ_D + β·τ_A。

## 门控 arm：student ⇐ teacher @θ
- **student** = 主解码模型（host / driver），**teacher** = 低置信位置被替换进来的模型（donor / other）。
- 例：`M(.4,.6) ⇐ M(0,1) @0.7`；`Ins ⇐ A(1.1) @0.4`。
- 单独跑（不门控）就只写模型名，如 `A(2.0)`。

## 指标
- acc 写成百分比并标明题集：前 200 / 后 300（id 200–499）/ 全 500；全部 @1024、H200 0.51。
- **2026-10-12 起默认只跑后 300 题**（用户：全集太贵；前 200 偏简单且已有基线）。新 arm 用 `run_gated_shard.sh` 分片跑 id 200–499，`merge_shards.py` 合并到 `*-merged`；和已有系统比较一律用后 300 题那一列。
- 配对比较写 `赢/输, p`（精确 McNemar）。
- 替换率 1 = 交给 teacher 的位置占比（student p<θ）；替换率 2 = teacher 的 token 与 student 不同的占比。

## 目录名对照（已有目录不改名，避免打断正在跑的 job）
| 目录 `details/logs/…/math500/` | arm |
|---|---|
| `lockstep1024/{parent,merge,add11,add20,l10}-free` | Ins, M(.4,.6), A(1.1), A(2.0), M(0,1) 单独跑 |
| `gated_ins_add-a10-b11_1024/tXX` | Ins ⇐ A(1.1) @0.XX |
| `gated_ins_add-a10-b20_1024/tXX` | Ins ⇐ A(2.0) @0.XX |
| `gated_ins_mrg0406_1024/tXX` | Ins ⇐ M(.4,.6) @0.XX |
| `gated_add-a10-b11_add-a10-b20_1024/t02` | A(1.1) ⇐ A(2.0) @0.2 |
| `gated1024/e1b-mrg10-tXX` | M(.4,.6) ⇐ M(0,1) @0.XX（500 题） |
| `gated1024n100/k3-l10-tXX`, `k3-l08-t07` | M(.4,.6) ⇐ M(0,1) / M(.2,.8)（100 题） |
| `gated_mrg0406_add-a10-b20_1024/tXX` | M(.4,.6) ⇐ A(2.0) @0.XX（t06/t08 于 10-10 22:00 按用户要求提交；t02/t04 是 10-09 删掉的半截 run） |

## RunAI job 名（2026-10-10 起）
`yqin-i1-<student>-to-<teacher>-t<θ×10>-h200f051`，模型缩写：ins, m46 = M(.4,.6), m28 = M(.2,.8), m01 = M(0,1), a11 = A(1.1), a20 = A(2.0)。
单独跑：`yqin-i1-<model>-alone-h200f051`。例：`yqin-i1-m46-to-m01-t06-h200f051`。旧名 gi11/gi20/gim/g1120/gml10 = ins-to-a11 / ins-to-a20 / ins-to-m46 / a11-to-a20 / m46-to-m01。

## 方法总称（用户命名，2026-10-10）
**over distillation**：从 overly merged（系数过头）的模型里 leverage 信息的方式的总称。目前的实例：
- token 级：置信门控替换（student ⇐ teacher @θ），teacher = over-merged 模型；
- 答案级：多个不同权重模型的投票（weight vote，`draft_scripts/weight_vote.py`）；
- （未做）真正的训练式蒸馏：用 over-merged 模型的输出/分布训练 student。
