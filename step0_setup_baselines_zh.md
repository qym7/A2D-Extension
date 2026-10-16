> 中文版（2026-10-16 译自 `step0_setup_baselines.md`，以英文版为准）

# STEP0 — 设置、基线，以及如何读这些数字

## 模型（Dream-v0-7B ↔ Qwen2.5-7B；记号见 `technical_details/step0_naming.md`）
τ_D = Dream-Instruct − Dream-Base（扩散后训练向量），τ_A = Qwen-Instruct − Qwen-Base（AR 后训练向量），D_b = Dream-Base。

| name | weights | role |
|---|---|---|
| Ins | D_b + τ_D | Dream-v0-Instruct |
| M(a,b) | D_b + a·τ_D + b·τ_A, a+b=1 | 凸组合线（A2D）。**M(.4,.6) = best merge**（按 IFEval 选出）；**M(0,1) = over merge**（丢弃 τ_D） |
| A(β) | D_b + τ_D + β·τ_A | 加法线。**A(1.1) = best addition**（按 IFEval 选出）；**A(2.0) = over addition**；A(3.0) 崩溃 |
| (0.7,1.3) | D_b + 0.7τ_D + 1.3τ_A | M(.4,.6) 与 A(2.0) 的权重空间中点，用作 1× 对照 |

"student ⇐ teacher @θ" = 由 student 解码；在其原始 token 概率 p < θ 的步上写入 teacher 的 token（step1_methods.md）。

## 协议
- MATH-500，lockstep 解码器：1024 tokens，1024 steps（每步一个 token），top_p 0.9，confidence temperature 0.1，argmax token（确定性：改 seed 不改变任何结果）。
- 子集：**first 200**（ids 0–199，较易，用于筛选），**last 300**（ids 200–499，较难，自 2026-10-12 起每个新 arm 的默认集），**all 500**。
  随机 100 题子集只能保持差距 ≥ 6 pp 的排序（`technical_details/step0_subset_checks/check_500.md`）；差距 < 3 / 3–6 / ≥ 6 pp 时，200-vs-500 的排序分别有 65% / 82% / 98% 的概率保持一致。
- 硬件：9/30 的运行在 B300 上（标 †），10/09 起的全部在 H200 上以 0.51 fraction 运行。每张表均有标注。
- 评分：对最后一个 `\boxed{}` 用 minerva `is_equiv`；"finish" = 输出含 `\boxed`。对门控系统和 A(2.0)，未完成的输出是真正在 1024 tokens 处被截断（last 300：无 boxed 的输出中分别有 27/33 和 13/17 长度 ≥1000 tokens，其余为无答案的提前停止）。**Ins 是例外**：其 73 个无 boxed 输出中有 39 个是完整的无 `\boxed` 短答案（"The answer is: 28"），一个粗略的宽松抽取器判其中 25 个正确（Ins lenient = 45.0 vs strict 36.7，last 300）。同一抽取器下 M(.4,.6) +2.3，M(0,1) +1.0；A(2.0) 与各门控 +0。
- 配对：McNemar win/loss，精确双侧 p。

## 基线

| model | first 200 | last 300 | all 500 | finish (last 300) | acc on finished (last 300) | HW |
|---|---:|---:|---:|---:|---:|---|
| Ins | 44.5 | 36.7 | 39.8 | 76% | 48.5 | B300 |
| M(.4,.6) best merge | 53.5 | 42.0 | 46.6 | 81% | 51.9 | B300 |
| M(.2,.8) | 54.0 | | | | | B300 |
| M(0,1) over merge | 48.0 | 44.0 | 45.6 | 74% | 59.2 | B300 |
| A(1.1) best addition | 57.0 | | | | | H200 |
| A(2.0) over addition | 58.5 | 46.7 | 51.4 | 94% | 49.5 | H200 |
| A(3.0) | | 34.0 | | 72% | 47.0 | H200 |
| (0.7,1.3) midpoint | 54.0 | 44.3 | 48.2 | 89% | 49.8 | H200 |
| **AR Qwen2.5-7B-Instruct, strict**（lm-eval greedy 1024，仅 `\boxed`） | 57.0 | 55.3 | 56.0 | 78% | 70.9 | H200 |
| **AR Qwen2.5-7B-Instruct, lenient**（从最后一句抽取最终答案） | | 70.7 | 71.0 | | | H200 |

## 制约后续所有解读的两个事实
1. **在 Dream 上，addition > merge 是完成率效应，不是精度效应。** A(2.0) 完成率 94% vs M(.4,.6) 的 81%，但在两者都完成的题上，M(.4,.6) 更准（first 200：三者都完成的 150 题上 68.7 vs 66.0；`technical_details/step0_addition_vs_merge_finish.md`）。扣除截断后，这与 A2D 的"凸组合最优"结论一致。
2. **严格评分器低估了 AR 参考，也低估了 Ins。** 其 22% 的答案完整但没有 `\boxed`；宽松抽取下是 71，不是 56。同一宽松评分器使我们最好的系统变化 ≤ 1（它的失误是真截断），却使 Ins 变化 +8：Ins ⇐ A(2.0) 的 +12.3 中约 8 来自 teacher 教会 `\boxed` 格式，解题增益 ≈ +4；M(.4,.6) ⇐ A(2.0) 在宽松评分下仍保持 +6.7。MATH-500 上与 AR 的真实差距 ≈ 19 分，不是 4。

## 我们为什么要做这件事（余量）
![motivation](technical_details/step0_figs/motivation_addition_line.svg)
*主线（Ins ⇐ A(β)，last 300）。加入 AR 向量使准确率上升直至 A(2.0)，在 A(3.0) 崩溃 (a)；全部增益来自完成率 (b)，每个完成答案的精度不变 (c)。门控在 student 有信心处保留其 token，在不确定处取 teacher 的：它继承了 teacher 的完成率（93%），且比两者都更精确（52.9）→ 49.0。为什么是不确定的位置 (d)：A2D 的 teacher-forced 测量表明，AR 向量几乎只在模型不确定的地方改写 token（96% 的改动发生在 p < 0.7；在 p ≥ 0.9 处，即全部 token 的 61%，每 15,000 个才改一个）——所以 student 的置信度正是 AR 侧信息所在之处。(e) last 300 上的 θ 扫描；同样剂量放在随机位置则停留在 student 水平（38.0 / 38.3）。`code/step0/fig_motivation_addition.py`。*

![motivation merge line](technical_details/step0_figs/motivation_merge_line.svg)
*第二条线（M(.4,.6) ⇐ M(0,1)，all 500）：从 merge 线看同一门控，over-merged 模型无法完成但每个答案更精确；门控保留完成率并继承精度（50.0 vs 46.6 / 45.6）。`code/step0/fig_motivation.py`。*

在已保存的输出上，"student 对 OR teacher 对"在若干 A2D 设置下比更好的单一模型高出 6–8 分（`technical_details/step0_headroom_retry_selective.md`），例如 last 300 上 M(.4,.6) ∪ A(2.0) = 58 vs 46.7。其中一部分是解码噪声（同协议重跑会翻转个别题），所以只有打败匹配的随机对照的 router 才算数。
