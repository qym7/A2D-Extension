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

**为什么是这个混合，这是常见的 instruction-following 配方吗？** 数据来源是标准的，用法不是。
- *Tulu-3 SFT mixture* 是公开的标准 post-training 数据集（AI2），也正是 Dream-v0-Instruct 自己微调所用的数据（官方
  `examples/run_sft_tulu3.sh`；Dream 报告：在 Tulu 3 + SmolLM2 上 SFT）。所以这些 prompt 就是学生自身的训练分布，
  "在 Tulu-3 上训练、在 GSM8K / MATH-500 / IFEval / HumanEval 上读数"是通行协议。它的数学子集保留（标准的 GSM8K/MATH 信号就来自这里），
  不额外加 GSM8K train。
- *MATH train* 是扩散语言模型上做数学 RL / 蒸馏的常用 prompt 来源（d1、TOPD 等）：对目标指标是同领域，且与 MATH-500 按构造不相交。
- *Magicoder-OSS-Instruct* 和 *no_robots* 是标准的代码集和对话集；放进来是为了让学生在训练中持续看到代码和对话 prompt，
  这正是 HumanEval+ / IFEval 两个 guard 检查的东西。
- **不标准**的地方：完全不使用参考回答。标准 SFT 拟合数据集里的答案；这里数据集只提供 prompt，学生自己写回答，
  唯一的外部信号是教师在学生不确定的步上的分布。比例（≈ ½ Tulu-3、⅓ MATH、其余代码/对话）是我们自定的，没有调过。

**2026-10-16 决定（用户）：按最规范的配方 = 只用 Tulu-3、原生比例。** Tulu-3 内部本来就有我们手工外加的那些子结构：在 100 k 样本中
≈ 36% 数学（personahub_math 15.9%、NuminaMath-TIR 6.9%、personas-math-grade 5.4%、open_math_2_gsm8k 5.3%、intermediate-algebra 2.1%），
≈ 15% 代码（evol_codealpaca 11.4%、personahub_code 3.7%），≈ 40% 对话/通用（wildchat、aya、flan、oasst、no_robots 1%），3% IF 数据，
≈ 12% 安全（wildjailbreak、wildguard、coconot）。所以额外加 MATH train、Magicoder、no_robots 是多余的（no_robots 本身就是 Tulu-3 的子集），
而且按来源等份分层也扭曲了 Tulu 自己的比例。下一个 prompt 池是 `outputs/step3/prompts_tulu3.jsonl`（`build_prompts.py --mix tulu3-native`，现在是默认：
从全部 Tulu-3 来源一次性均匀打乱抽 17 k，不排除任何子集，prompt ≤ 512 token；`n_prompts: N` 的 run 取 id 0–N−1，即原生比例的 N 条样本；
`pipeline/config.yaml: prompt_set: tulu3`）。上面的 t1k 数字来自之前的 mix17k 池，保持这个标注。

**n1k run**（2026-10-16 启动）用这个池的 id 0–999：数学 415（personahub_math 179、NuminaMath-TIR 84、personas-math-grade 64、
open_math_2_gsm8k 63、intermediate-algebra 25），代码 123（evol_codealpaca 98、personahub_code 25），对话/通用 289（aya 97、wildchat 89、
flan 85、oasst 9、no_robots 4、sciriff 3、table_gpt 2），IF 42，安全 131（wildguard 68、wildjailbreak 51、coconot 12）。数学占 41% 而不是
原始的 36%，是因为 ≤ 512 token 的过滤去掉的长对话 prompt 比数学 prompt 多。这 1 000 条与 MATH-500 id 200–499 的泄漏检查：0 条包含、0 个共享 13-gram。

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

## 结果（1k 测试；表由 `code/step3/results_table.py` 生成 → `technical_details/step3_results_t1k.md`）

下表每一行都是**单模型**在同一批 MATH-500 题上解码（后 300 ∩ 所有行都有的题；M 的最后两个 shard 还在跑，目前是 200 题），
只有两行 gate 是 2×（解码时两个模型）。STEP0 的两个**端点基线**都在表里：best merge M(.4,.6)（convex 线）和 over addition A(2.0)
（addition 线；它同时也是教师）。McNemar = 行赢次数 / 列赢次数，精确 p。

| row | acc | finish | acc\|finished | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | HW |
|---|---:|---:|---:|---|---|---|---|---|
| Ins（学生，未改动） | 38.5 | 77% | 50.0 | — | 26/25, p=1.00 | 17/26, p=0.22 | 22/41, p=0.02 | B300 |
| M(.4,.6) best merge（端点 1） | 43.0 | 84% | 50.9 | 26/17, p=0.22 | 28/18, p=0.18 | — | 23/33, p=0.23 | B300 |
| A(2.0) over addition（端点 2 = 教师） | 48.0 | 94% | 50.8 | 41/22, p=0.02 | 36/16, p=0.01 | 33/23, p=0.23 | — | H200 |
| gate Ins ⇐ A(2.0) @0.8（2×） | 50.5 | 93% | 54.3 | 38/14, p=0.00 | 40/15, p=0.00 | 30/15, p=0.04 | 27/22, p=0.57 | H200 |
| gate M(.4,.6) ⇐ A(2.0) @0.8（2×） | 53.0 | 91% | 58.2 | 41/12, p=0.00 | 41/11, p=0.00 | 30/10, p=0.00 | 27/17, p=0.17 | H200 |
| **M = 蒸馏后的 Ins（1×）** | **50.5** | **94%** | **54.0** | 39/15, p=0.00 | 40/15, p=0.00 | 32/17, p=0.04 | 28/23, p=0.58 | H200 |
| **C2 = Ins 自蒸馏（1×）** | 38.0 | 81% | 46.9 | 25/26, p=1.00 | — | 18/28, p=0.18 | 16/36, p=0.01 | H200 |

按可信度排序，这张表说明：
1. **M ≫ Ins，M ≫ C2**（都是 p < 0.001，39–40 赢 vs 15）：蒸馏后的 1× 模型优于起点，原因是教师（自蒸馏什么都没带来）。
2. **M ≥ 两个端点。** M 胜过 best merge（32/17，p = 0.04），与 over addition / 教师至少持平（28/23，p = 0.58），
   而它只是 Ins 权重 + 一个 LoRA 的 1× 模型——测试时没有 merge、没有向量运算。
3. **M = 它所蒸馏的 2× gate**（50.5 vs 50.5；gate 就是我们一开始要达到的天花板）。merge 起点的 gate（53.0）是下一个目标（臂 Mm）。
4. 它具有 gate 的两个特征：写完率 77 → 94%（教师水平），写完题的正确率 50 → 54（高于教师的 50.8）。

这两次训练的来源：`outputs/checkpoints/step3-t1k-{M,C2}/{args.json,train_log.jsonl,run.log}`；W&B 项目 `a2d-step3`，
run `step3-t1k-M` / `step3-t1k-C2`（由日志回填）。注意：M 的容器在完成前重启了两次（`run.log` 有三次启动、一次 `[done]`）；
checkpoint 来自唯一完整的那次，日志文件里也含有中断的尝试。

## 真正有用的评测需要什么（以及我们已有哪些）
| 要求 | 现状 | 动作 |
|---|---|---|
| 每行同一硬件 | Ins 和 M(.4,.6) 是 B300 的数（2026-09-30）；STEP3 各行是 H200 | 在 H200 上重测 Ins 和 M(.4,.6) 的后 300（2 个 job，≈ 5 GPU·h） |
| 整 300 题而不是 200 | M 的 shard 0、2 在跑 | 自动；表会重新生成 |
| 模型没有在别处变差 | 未测 | 对 M 跑 GSM8K、HumanEval+、IFEval 官方 harness（`guard_arms`），≈ 20 GPU·h |
| 不是 seed 的偶然 | 一个 seed | M 换一个训练 seed（0.5 h + 7 GPU·h 评测） |
| 不是 1k 的偶然 | 一个数据量 | 8k offline（或 4 × 2k online）——同时回答 1k 是否已到平台 |
| 统计尊重配对 | 已做：相同题 id 上的精确 McNemar | 保持；不同题集之间永远不比 acc |
| 无泄漏 | 已查（上一节） | prompt 池一变就重查 |
| 能杀掉其他解释的对照 | C2（需要教师吗）已做；C1（需要选择吗）、C3（需要 over 吗）、C4（需要回放态吗）、C5（KL 方向）未跑 | guards 之后在同一批 rollout 上跑（每个 0.5 h 训练） |

## 训练代码是怎么来的
为这一步专门写的，不是改自某个 trainer：`code/step3/train_frontier_kd.py` 约 300 行纯 PyTorch + HF `transformers` + `peft`（LoRA），
单卡，没有 Lightning、没有 verl、没有 accelerate。原样复用的部分：Dream 官方采样器做 rollout（`diffusion_generate`，与 MATH-500 协议同设置）、
Dream 的 logit 约定（位置 p 由隐层 p−1 预测，与官方 SFT trainer 和我们的解码器一致）、peft 的 LoRA 用法与官方 `run_sft_tulu3.sh` 相同、
STEP1 的 lockstep 解码器做评测。新写的部分：轨迹回放作为破坏方式、按置信度分流的 loss、self-anchor、帧采样器、OOM micro-batch 探测、
以及合并/保存路径（`merge_lora.py`，绕过 Dream 的 `generation_config.validate`）。没有用官方 verl FSDP trainer，因为它的 loss
（对参考答案做随机 mask CE）正是这一步要替换掉的东西，而且它需要 4 张卡。

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
