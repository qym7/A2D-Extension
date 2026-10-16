> 中文版（2026-10-16 译自 `step3_distillation.md`，以英文版为准）

# STEP3 — 蒸馏：用一个模型做到门控所做的事（2026-10-16，进行中）

STEP2 已关闭（`step2_verdict.md`）：解码时门控 Ins ⇐ A(2.0) @0.8 是天花板（49.0，2× 成本），任何组合都没有超过它。
STEP3 把这一行为训练进单个 1× 模型。配方：`technical_details/step3_training_formulation.md`；计划与读数规则：
`technical_details/step3_plan_frontier_kd.md`。

## 要点（1k 测试，2026-10-16）
1. **1× 模型达到了 2× 门控的准确率**：在 1 k 训练 prompt 上训练 185 个优化器步（半小时训练）后，
   在同样 200 道 MATH-500 题上 50.5 vs Ins 38.5，p < 0.001。
2. **起作用的是教师。** 用同样的 rollout、同样的训练器做 self-SFT（C2）停留在 Ins 水平（38.0，vs Ins 的 p = 1.0）。
3. 增益带有门控的特征：完成率 77 → 94 % **并且** 已完成题目上的准确率 50 → 54（门控：93 % / 52.9）。

## 三行

| row | what is evaluated | how it was made | training data | test-time cost |
|---|---|---|---|---|
| **Ins** | `Dream-org/Dream-v0-Instruct-7B`，未改动 | — | — | 1× |
| **C2** | Ins + LoRA，合并为一个 checkpoint | self-SFT：在每个回放步上对 Ins 自己写出的 token 做交叉熵（相同帧、相同优化器，**无教师**） | 这 1 k 条 rollout | 1× |
| **M** | Ins + LoRA，合并为一个 checkpoint | 门控蒸馏：在低置信度 commit 之前的帧上，对 A(2.0) 做反向 KL；在高置信帧上，对未改动的 Ins 做 KL（λ 0.3） | 这 1 k 条 rollout | 1× |

三行都**由单个模型独立**解码，采样器完全相同（`code/step1/lockstep_gated.py`，模式 `free`，1024 步，top-p 0.9，
conf-temp 0.1）：测试时无教师、无门控、无第二次前向。教师 A(2.0) = D_b + τ_D + 2.0·τ_A
（`outputs/checkpoints/add-a10-b20`）只在训练损失内部使用。

超参数（M 与 C2 除损失外完全相同）：LoRA r 64 / α 128，作用于全部 attention + MLP 投影，lr 1e-4（10 % warm-up，线性
衰减），每步 32 帧，micro-batch 4，在 1 000 个 canvas × 3 帧 = 6 000 帧上训练 2 个 epoch = 185 步，θ = 0.8。
Checkpoint：`outputs/checkpoints/step3-t1k-{M,C2}/{lora,merged}`，日志 `train_log.jsonl`，参数 `args.json`。

## 训练数据（学生看到了什么）
`outputs/step3/prompts.jsonl` ids 0–999，是从 17 k 的 STEP3 prompt 池中固定种子采样得到（`code/step3/build_prompts.py`，seed 20261015）：

| source | n (of 1 000) | what |
|---|---|---|
| Tulu-3 SFT mixture（`allenai/tulu-3-sft-mixture`，除 wildjailbreak / wildguard / hard_coded 外的全部子集） | 489 | 通用指令数据，含其数学子集（open_math_2_gsm8k、personas-math 等） |
| MATH **train** split（`EleutherAI/hendrycks_math`，train） | 335 | 竞赛数学，包装为 `Problem:\n…\n\nSolution:` |
| Magicoder-OSS-Instruct | 122 | 代码 |
| no_robots | 54 | 对话 |

只使用 **prompt**。每个学生自己把每个 prompt 解码一次（`code/step3/rollout.py`，官方 Dream 采样器，512 token /
512 步，T 0.1，top-p 0.9；`outputs/step3/rollouts/ins/`），记录每个写出 token 的步数和原始置信度。没有参考答案、
没有 ground truth、没有按正确性过滤的内容进入训练。训练信号是教师在学生自身低置信步上的 next-token 分布（M），
或学生自己的 token（C2）。

## 测试数据与泄漏
测试集 = MATH-500（`HuggingFaceH4/MATH-500`，MATH **test** split 的标准 500 题子集），ids 200–499（最后 300 题，用户规则
2026-10-12）。已检查（`tmp.md` 2026-10-16）：1 000 条训练 prompt vs 300 道测试题，文本归一化后 —
0 个完全匹配，0 个包含关系，1 个共享 13-gram，即选择题模板 "(A) Line (B) Circle … Enter the letter of the
correct option."，出现在两道不同的题上（train 875 vs test 255）。MATH 的 train 与 test 按构造互斥；Tulu-3 已由其作者
针对 MATH test 去污染。→ **训练 prompt 到测试题没有泄漏。** 学生 rollout 只在训练 prompt 上生成，从未在 MATH-500 上生成。

## 泛化：已经展示和尚未展示的
- 已展示：增益从**混合训练 prompt**（其中一半不是数学）迁移到**留出的竞赛数学**，且学生独立解码。测试集与 STEP0–2
  衡量 Ins、各合并模型、各教师和门控所用的留出集相同，因此所有数字可比（硬件注意事项见下）。
- 尚未展示：三个 guard（GSM8K、HumanEval+、IFEval；官方 harness，`pipeline/config.yaml: guard_arms`）— 它们决定 M
  是否在别处**有所损失**。它们在 MATH 读数之后、在过了 MATH 门槛的那个 arm 上运行。
- 尚未展示：第二个种子 / 第二个模型家族；以及用于判断 1 k 是否已是平台期的 8 k / 在线运行。

## 当前数字的注意事项
- 300 题中已有 200 题（M 的 shard 0 和 2 因一个任务跟踪 bug 正在重跑，2026-10-16 上午）；完整 300 列和
  `technical_details/step3_results_t1k.md` 随后给出。
- Ins 行在 **B300** 上解码（2026-09-30），M 和 C2 在 **H200·0.51** 上。规则是每张表只有一个硬件列；在对外引用任何数字之前，
  下一个要申请的任务是在 H200 上对 Ins 重跑最后 300 题。
- C2 的输出更长且部分退化（重复 "The final answer is …"；平均 3 029 字符 vs Ins 888）；对于普通 self-SFT，lr 1e-4 偏高。
  这不改变其准确率（38.0），但意味着 C2 是一个保守的对照，而非调优过的对照。
- 计划中的读数规则仍然适用：成功 = ≥ Ins + 6 **且** > C2 + 3 **且** 无 guard 回退。

## 下一步
1. 完成 300 题（自动）。2. Ins 在 H200 上跑最后 300 题（一个任务）。3. 对 M 跑 guard。4. 8 k 离线 vs 4 × 2 k 在线轮次（`tmp.md` 2026-10-15 计划），
M06（θ 0.6）和 Mm（M(.4,.6) 初始化）作为下一批 arm。
