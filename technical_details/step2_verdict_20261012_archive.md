# (ARCHIVED 2026-10-15; superseded by step2_verdict.md) STEP2 — Verdict and TODO: what died, what survives, and what comes next (2026-10-12)

> **TODO (in order)**
> 1. Fill the mechanism-ablation table in `step1_methods.md` §2 (both_gated / schedule_gated / unconditional random on Ins ⇐ A(2.0) and M(.4,.6) ⇐ M(0,1); M(0,1) θ = 0.6/0.8/0.9 on the last 300) — **done 2026-10-13**.
> 2. Decided: confidence selectivity is real (unconditional random at the gate's dose = student, −11 / −6 pts vs the gate, p ≤ 0.03; position-only = student; position+content ≤ content). The method stays "confidence-gated content from the teacher, student keeps the order".
> 3. Two 1× controls still unrun: noise injection (student samples ~5% of its own tokens at temperature) and student @2048 tokens — "does the +4 need a second model?"
> 4. Lazy teacher call (teacher forward only on low-confidence steps) → identical outputs at ≈1.15× compute; implement before any cost claim.
> 5. Pipeline with fixed rules for every hyperparameter (student, teacher line, teacher dose β, θ or f), then generalise to a second family (Dream-Coder/HumanEval or DreamReasoner/AIME) without re-tuning.
>    **Main line (user, 2026-10-13): one of the two clean pairs, not M(.4,.6) ⇐ A(2.0)** (that one changes both lines at once and is kept only as the best number).
>    Proposed: **Ins ⇐ A(β)** as the main line — the student is the official Instruct model (IFEval anchor, no A2D merge needed first), the pipeline has a single knob β past the addition peak, and it is the higher of the two (49.0 vs 47.7 on the last 300; 53.0 vs 50.0 on 500). Its gain must be reported split: ≈ +8 is the `\boxed` format the teacher brings, ≈ +4 is solving (`step0_setup_baselines.md` fact 2); the finish-rate mechanism is also the one self-RFT is most likely to reproduce, so the STEP3 control matters most here.
>    **M(.4,.6) ⇐ M(0,1)** stays as the second line: the precision mechanism (52 → 62 on finished problems, finish rate unchanged) is token-level knowledge the student cannot get from its own samples, which is what the distillation control is designed to detect. Both are carried through STEP3; the distilled 1× model is read on finish rate and acc-on-finished separately to tell which behaviour it learned.
> 6. STEP3 (distillation): distil the mixture's outputs into the student (sequence-level first, selective token weighting second); success = 1× ≥ 2× mixture on MATH and no IFEval loss.
> 7a. **Two-line cascade (2026-10-13, zero GPU, `technical_details/step2_cascade_two_lines.md`)**: run M(.4,.6) ⇐ M(0,1) @0.8 first, fall back to Ins ⇐ A(2.0) @0.8 when it does not finish → 57.4 on all 500 / 55.0 on the last 300 (vs Ins ⇐ A(2.0) 51/29, p=0.02; vs best single 54.2, p=0.13), ≈ 2.45× compute. Reaches AR strict (56.0), not AR lenient (71). The two lines repair each other's *detectable* failure (the precision line fails by not finishing, the finish line by finishing wrong).
> 7. If the goal stays "beat AR": find a benchmark/family where diffusion is within reach of AR under *lenient* scoring; on MATH-500 with Dream-7B it is not (71 vs ≤ 55).

所有数字：MATH-500，lockstep 解码 @1024 token / 1024 步，top_p 0.9，conf_temp 0.1。前 200 = id 0–199，后 300 = id 200–499，全 500 = 两者合并。
硬件：9/30 的 run 是 B300 整卡（标 †），其余 H200·0.51。空格 = 没跑。写法见 `technical_details/step0_naming.md`；生成脚本 `scripts/step1/main_table.py`，分片合并 `merge_shards.py`。
配对比较 = 赢/输（McNemar 精确 p）。

## 1. 单模型基线（1×）

| 模型 | 位置 | 前 200 | 后 300 | 全 500 |
|---|---|---:|---:|---:|
| Ins | Dream-v0-Instruct (τ_D=1, τ_A=0) | 44.5 | 36.7 | 39.8 |
| M(.4,.6) | best merge（convex，IFEval 选） | 53.5 | 42.0 | 46.6 |
| M(.2,.8) | convex | 54.0 | | |
| M(0,1) | over merge（convex 终点） | 48.0 | 44.0 | 45.6 |
| A(1.1) | best addition（IFEval 选） | 57.0 | | |
| A(2.0) | over addition | 58.5 | 46.7 | 51.4 |
| A(3.0) | 更过头 | | 34.0 | |
| (0.7,1.3) | M(.4,.6) 与 A(2.0) 的权重中点 | 54.0 | 44.3 | 48.2 |
| **AR Qwen2.5-7B-Instruct**（严格，只认 `\boxed`） | 参考线，lm-eval 贪心 1024 | 57.0 | 55.3 | 56.0 |
| **AR Qwen2.5-7B-Instruct**（宽松，结尾抽答案） | 同上 | | 70.7 | 71.0 |

AR 的 22% "未写完" 是没写 `\boxed` 的完整答案，不是截断；宽松打分对我们的系统只 +1（我们的未写完是真截断）。**和 AR 的真实差距约 19 个点。**

## 2. 门控替换 student ⇐ teacher @θ（2×）

| student ⇐ teacher | θ | 前 200 | 后 300 | 全 500 | 全 500 vs student | 全 500 vs teacher |
|---|---|---:|---:|---:|---|---|
| Ins ⇐ A(1.1) | 0.2 / 0.4 / 0.6 / 0.8 | 47.5 / 53.0 / 52.0 / 57.5 | | | | |
| Ins ⇐ A(2.0) | 0.2 / 0.4 | 44.5 / 53.0 | | | | |
| Ins ⇐ A(2.0) | 0.6 | 56.5 | 49.3 | 52.2 | 96/34, <0.01 | 59/55, 0.78 |
| Ins ⇐ A(2.0) | 0.8 | 59.0 | 49.0 | 53.0 | 106/40, <0.01 | 60/52, 0.51 |
| Ins ⇐ M(.4,.6) | 0.2 / 0.4 | 43.0 / 46.5 | | | | |
| Ins ⇐ M(.4,.6) | 0.6 | 53.5 | 44.7 | 48.2 | 78/36, <0.01 | 55/47, 0.49 |
| Ins ⇐ M(.4,.6) | 0.8 | 52.0 | 41.7 | 45.8 | 73/43, 0.01 | 48/52, 0.76 |
| A(1.1) ⇐ A(2.0) | 0.2 | 55.0 | | | | |
| M(.4,.6) ⇐ M(0,1) † | 0.3 | 54.0 | 43.3 | 47.6 | 33/28, 0.61 | 58/48, 0.38 |
| M(.4,.6) ⇐ M(0,1) | 0.6 / 0.8 | 52.0 / 53.5 | | | | |
| M(.4,.6) ⇐ M(0,1) † | 0.7 | 54.0 | 48.0 | 50.4 | 66/47, 0.09 | 63/39, 0.02 |
| M(.4,.6) ⇐ A(2.0) | 0.4 | | 43.7 | | | |
| M(.4,.6) ⇐ A(2.0) | 0.6 | 58.0 | 49.7 | 53.0 | 75/43, <0.01 | 60/52, 0.51 |
| **M(.4,.6) ⇐ A(2.0)** | **0.8** | **59.0** | **51.0** | **54.2** | 79/41, <0.01 | 60/46, 0.21 |
| M(.4,.6) ⇐ A(2.0) | 1.0（teacher 内容 + student 顺序） | | 46.3 | | | |
| M(.4,.6) ⇐ A(3.0) | 0.8 | | 42.3 | | | |

最好系统 M(.4,.6) ⇐ A(2.0) @0.8 的替换率（全 500）：交给 teacher 的位置 15.2%，两模型分歧的位置 6.0%，最终写 teacher 且与 student 不同的 token 5.2%（= 分歧位置的 85.4%）。

## 3. 对照与机制（后 300 题，student M(.4,.6)，teacher A(2.0)）

| 系统 | 分歧位置里换掉的比例 | acc | 写完率 | 写完的题里 acc | 平均长度（字符） | vs 门控 @0.8 |
|---|---:|---:|---:|---:|---:|---|
| M(.4,.6) 单独 | 0% | 42.0 | 81% | 51.9 | 1123 | |
| 随机位置替换 | 26% | 46.3 | 84% | 54.9 | 1440 | |
| 门控 @0.4 | 26% | 43.7 | 83% | 52.6 | 1319 | 23/45, 0.01 |
| 门控 @0.6 | 58% | 49.7 | 89% | 56.0 | 1474 | 24/28, 0.68 |
| 随机位置替换 | 85% | 50.0 | 93% | 53.6 | 2678 | 29/32, 0.80 |
| **门控 @0.8** | 85% | **51.0** | 89% | **57.3** | 1893 | — |
| @1.0（全部 teacher 内容，student 顺序） | 100% | 46.3 | 95% | 48.8 | 2816 | 15/29, 0.05 |
| A(2.0) 单独 | — | 46.7 | 94% | 49.5 | 2824 | |
| 控制：A(3.0) 单独 | — | 34.0 | 72% | 47.0 | 3126 | |
| 控制：M(.4,.6) ⇐ A(3.0) @0.8 | 83% | 42.3 | 77% | 54.7 | 2525 | 14/40, <0.01 |
| 控制：(0.7,1.3) 单独（权重中点，1×） | — | 44.3 | 89% | 49.8 | 3296 | 31/51, 0.04 |
| 控制：Ins 单独 / Ins ⇐ A(2.0) @0.8 | 0% / 85% | 36.7 / 49.0 | 76% / 93% | 48.5 / 52.9 | 918 / 2133 | |
| 控制：M(0,1) 单独 / M(.4,.6) ⇐ M(0,1) @0.7 † | — / — | 44.0 / 48.0 | 74% / 77% | 59.2 / 62.1 | 1734 / 1433 | |
| AR（严格 / 宽松） | — | 55.3 / 70.7 | 78% | 70.9 | 1393 | |

随机 26% 的平均长度按 300 题重算。

## 4. 答案级投票（仅记录，已放弃作为方法：需要完整序列，不是 online 的）

| 体系 | 成员 | 解码次数 | 全 500 |
|---|---|---:|---:|
| M(.4,.6) ⇐ A(2.0) | @0.6 + @0.8 | 2 | 56.1 |
| | @0.6 + @0.8 + A(2.0) | 3 | 59.6 |
| | M(.4,.6) + @0.6 + @0.8 + A(2.0) | 4 | 61.6 |
| Ins ⇐ A(2.0) | @0.6 + @0.8 + A(2.0) | 3 | 58.7 |
| | 全 4 | 4 | 61.0 |
| oracle（任选对的）：最好 2 系统 / 3 系统 / 全部 13 系统 | | | 64.6 / 69.2 / 78.4 |

全 500 题里 108 题（22%）没有任何系统做对。

## 5. 判定

**什么死了（每条都有对照）**
1. 置信门控的"选择性"：26% 和 85% 两个剂量下，随机位置都 ≈ 门控。student 的置信度不知道 teacher 在哪里有用。
2. 解码顺序：teacher 的全部内容配 student 的顺序 = teacher 单独（46.3 vs 46.7），且显著低于 @0.8。
3. ~~over merge 当 teacher 没有收益~~ → **更正（用户 10-12 指出）**：M(.4,.6) ⇐ M(0,1) @0.7 在后 300 题是 48.0 vs student 42.0（46/28, p=0.05），全 500 是 50.4 vs 46.6（p=0.09）；只是前 200 题上平（54.0 vs 53.5），所以在 200 题筛选时被误判。它的收益走的是另一条路：M(0,1) 写不完（74%），门控把写完题里的精度从 51.9 提到 62.1，写完率不动（77%）。A(2.0) 则相反（提写完率）。两种 teacher 互补，但 M(0,1) 的总收益小于 A(2.0)（48.0 vs 51.0），且该 run 是 9/30 B300 旧 run，@0.6/@0.8 在后 300 题没跑过。
4. 更过头的 teacher：A(3.0) 已崩（34.0），门控只能救回 student。
5. 权重中点替身：(0.7,1.3) 拿到写完率（89%）但拿不到精度（49.8），比混合低 6 个点（p=0.01）。
6. student 是谁：Ins 和 M(.4,.6) 门控后只差 1 个点。

**什么活着**
- 两个模型在分歧位置按约 85% 混合，比任一端点高 4–5 个点、比权重中点高 6 个点，前 200 / 后 300 / 全 500 方向一致，换 student 也成立。
- 机制可分解：teacher 的 token 让答案在 1024 内写完（81% → 89–95%），少量 student 的 token 保住精度（写完题里 51.9 → 57.3；100% 时塌回 teacher 的 49.5）。85% 是两者乘积的最高点。位置不重要，所以这是互补，不是"teacher 更强"。
- 可预测的边界：teacher 必须"写得完但不够准"（A(2.0)），不能"写不完"（M(0,1)）或"已崩"（A(3.0)）。

**Main takeaway**
我们找到了一个直接的、可解释的改进：在 A2D 的最好点之上，用一个过头的 addition 模型做 token 级混合，MATH-500 全 500 题从 46.6（best merge）/ 51.4（over addition）提到 54.2，比论文里任何单点都高，机制清楚（写完 × 写对），边界条件可预测。
但它可能还不够：(a) 要 2× 算力、两个模型；(b) 它不是"从过头模型里提取信息"的选择性蒸馏，而是一个 token 级集成；(c) 在 MATH 上追不上 AR 源模型——AR 的真实水平是 71，我们的 oracle 上限 78、任意 3 系统 oracle 69、真实方法约 55 封顶。
所以这条线的价值只能是两个之一：把 +4 蒸成 1×（STEP3），或把同样的混合搬到 diffusion 本来就接近 AR 的 benchmark / 模型族上，在那里争取超过 AR。两者都需要先回答"这 +4 需不需要第二个模型"（噪声注入对照、student@2048 对照），两个都是 1×、便宜、还没跑。

完整的四对并排表（中英文）在 `step1_methods.md` §3.1。
