> 中文版（2026-10-16 译自 `step3_distillation.md`，以英文版为准）

# STEP3 — 蒸馏：用一个模型做到门控所做的事（2026-10-16，进行中）

STEP2 已关闭（`step2_verdict.md`）：解码时门控 Ins ⇐ A(2.0) @0.8 是天花板（49.0，2× 成本），任何组合都没有超过它。
STEP3 把这一行为训练进单个 1× 模型。配方：`technical_details/step3_training_formulation.md`；计划与读数规则：
`technical_details/step3_plan_frontier_kd.md`。

## 要点（1k 测试，2026-10-16；后 300 题完整数字）
1. **1× 模型达到了 2× 门控的准确率**：在 1 k 训练 prompt 上训练 185 个优化器步（13 分钟训练）后，
   在 MATH-500 后 300 题上 50.3 vs Ins 36.7，p < 0.001；它所蒸馏的 gate 是 49.0。
2. **起作用的是教师。** 用同样的 rollout、同样的训练器做 self-SFT（C2）停留在 Ins 水平（36.3，vs Ins 的 p = 1.0）。
3. 增益带有门控的特征：写完率 76 → 92 % **并且** 已写完题目上的准确率 48.5 → 54.7（门控：93 % / 52.9）。
4. **只用 Tulu-3 的 run（tulu3-1k）还没有做成** —— 2026-10-16 那次无效（两个流水线 bug，见“无效的 tulu3-1k run”），已重新排队。
   它唯一可用的观察：只在 **119 条 canvas 上训 22 步** 的模型也拿到 50.3 —— 效应很早就饱和。
5. **泛化（官方 IFEval / HumanEval+ harness，IFEval 随机 100）：** 蒸馏模型保持了 Ins 水平的指令跟随（64 vs 62）和最佳合并水平的代码
   （60.4 vs 61.0），而教师 A(2.0) 自己的 IF 很差（51）。它拿到了教师的数学，没拿到教师的损伤。
   2× gate 自己在 IFEval 上 70、HumanEval+ 上 59.8 —— gate 本身也不伤 IF / 代码。
6. **AR 教师基线（2026-10-17）：** 在前缀 mask 的 rollout 上把 Qwen2.5-7B-Instruct 蒸馏进 Ins（同样的数据、训练器、预算），MATH 46.7
   （diffusion over 教师 50.3），**IFEval 掉到 52**（vs 蒸馏 M p = 0.01）。diffusion 教师在两个轴上都赢。

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
Checkpoint：`outputs/checkpoints/step3-mixpool-1k-{M,C2}/{lora,merged}`，日志 `train_log.jsonl`，参数 `args.json`。

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
`pipeline/config.yaml: prompt_set: tulu3`）。上面的 mixpool-1k 数字来自之前的 mix17k 池，保持这个标注。

**tulu3-1k run**（2026-10-16 启动）用这个池的 id 0–999：数学 415（personahub_math 179、NuminaMath-TIR 84、personas-math-grade 64、
open_math_2_gsm8k 63、intermediate-algebra 25），代码 123（evol_codealpaca 98、personahub_code 25），对话/通用 289（aya 97、wildchat 89、
flan 85、oasst 9、no_robots 4、sciriff 3、table_gpt 2），IF 42，安全 131（wildguard 68、wildjailbreak 51、coconot 12）。数学占 41% 而不是
原始的 36%，是因为 ≤ 512 token 的过滤去掉的长对话 prompt 比数学 prompt 多。这 1 000 条与 MATH-500 id 200–499 的泄漏检查：0 条包含、0 个共享 13-gram。

## 测试数据与泄漏
测试集 = MATH-500（`HuggingFaceH4/MATH-500`，MATH **test** split 的标准 500 题子集），ids 200–499（最后 300 题，用户规则
2026-10-12）。已检查（`tmp.md` 2026-10-16）：1 000 条训练 prompt vs 300 道测试题，文本归一化后 —
0 个完全匹配，0 个包含关系，1 个共享 13-gram，即选择题模板 "(A) Line (B) Circle … Enter the letter of the
correct option."，出现在两道不同的题上（train 875 vs test 255）。MATH 的 train 与 test 按构造互斥；Tulu-3 已由其作者
针对 MATH test 去污染。→ **训练 prompt 到测试题没有泄漏。** 学生 rollout 只在训练 prompt 上生成，从未在 MATH-500 上生成。

## 泛化：1× 蒸馏模型 vs 两个端点和最佳合并（2026-10-16）
每一行都是**单个模型独立解码**（gate 行是它被蒸馏自的 2× 解码期门控）。MATH 每列用相同的题目 id；IFEval 和 HumanEval+ 用**官方 harness
和官方解码设置**（IFEval 1280/1280，HumanEval+ 512/512 T0.1，除注明外都在 H200·0.51）。子集都是**固定随机**列表（从不取前/后 N 题）：MATH rand100 =
全 500 题 seed-0 随机 100；IFEval rand100 = 541 条 prompt 的 seed-0 随机 100（`code/step3/guards/ifeval_rand100.yaml`，官方打分器原样导出）。
Ins 和 M(.4,.6) 的 IFEval / HumanEval+ **没有重跑**：直接从已有的官方全集日志里读这 100 题（IFEval 全 541：Ins 56.6，M(.4,.6) 66.5 prompt-strict；
HumanEval+ Ins 为 2026-09-19 的结果）。McNemar 格：行赢/列赢（共同题目上），精确双侧 p。

| row | MATH last 300 | MATH rand100/500 | IFEval rand100 (strict) | HumanEval+ (164) | MATH100 vs Ins | IFEval vs Ins | HE+ vs Ins | MATH100 vs M(.4,.6) | IFEval vs M(.4,.6) | HE+ vs M(.4,.6) |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Ins (student, untouched) | 36.7 | 34 | 62 | 57.3 | — | — | — | 9/17, p=0.17 | 13/16, p=0.71 | 10/18, p=0.18 |
| M(.4,.6) best merge | 42.0 | 42 | 65 | 61.0 | 17/9, p=0.17 | 16/13, p=0.71 | 18/10, p=0.18 | — | — | — |
| A(2.0) over addition (= the teacher) | 46.7 | 55 | 51 | 56.7 | 29/8, p=0.00 | 8/19, p=0.05 | 19/18, p=1.00 | 22/9, p=0.03 | 8/22, p=0.02 | 16/23, p=0.34 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 49.0 | 51 | 70 | 59.8 | 22/5, p=0.00 | 17/9, p=0.17 | 21/16, p=0.51 | 18/9, p=0.12 | 12/7, p=0.36 | 16/19, p=0.74 |
| M distilled (mixpool-1k) | 50.3 | 58 | 64 | 60.4 | 30/6, p=0.00 | 14/12, p=0.85 | 17/12, p=0.46 | 24/8, p=0.01 | 7/8, p=1.00 | 14/17, p=0.72 |
| AR-teacher baseline: Ins ⇐ Qwen2.5-7B-Instruct, prefix mask (mixpool-1k) | 46.7 | 51 | 52 | 57.3 | 26/9, p=0.01 | 12/22, p=0.12 | 15/15, p=1.00 | 19/10, p=0.14 | 5/18, p=0.01 | 10/18, p=0.18 |

- **MATH**：蒸馏 M 在所有列都是最好的一行（后 300：50.3；全 500 随机 100：58），显著高于 Ins 和最佳合并，并达到它所蒸馏的 2× gate 的水平
  （gate 在 rand100 上 51），但只解码一次。
- **IFEval**：教师 A(2.0) 的指令跟随**很差**（51，vs Ins p = 0.05，vs M(.4,.6) p = 0.02）—— 正是"过度"端点已知的损伤。蒸馏学生**没有**继承它：
  64 vs Ins 62 / M(.4,.6) 65，p ≈ 1。2× gate 自己拿到 **70**（vs 它的 donor A(2.0) 25/6，p < 0.001；vs Ins 17/9，p = 0.17；vs 蒸馏 M 13/7，
  p = 0.26）—— 只把低置信 token 交给一个 IF 很差的 donor，反而得到一点 IF 增益；蒸馏模型保住了 Ins 水平，但没拿到 gate 的 70
  （用 `code/step3/guards/gate_ifeval.py` 解码：与 gate 的 MATH 行同一段 gate 代码、argmax 取 token，官方 prompt 和打分器，不是 lm_eval 的采样解码）。在全 541 题上最佳合并比 Ins 高 10 分（66.5 vs 56.6，p < 0.001）；在这随机 100 题上差距只有 3 分，p = 0.71 ——
  100 题的子集分辨不出 10 分的 IFEval 差异，这就是为什么下面的验证集是每个任务 100 题，但任何正式结论都要上全集。
- **HumanEval+**：蒸馏 M 60.4 ≈ gate 59.8 ≈ 最佳合并 61.0 > Ins 57.3 > 教师 56.7；两两差异都不显著（164 题）。教师在代码上并不更强，学生也没有变差。
- **AR 教师基线（2026-10-17，同事的要求："直接对 rollout 做 AR 式的 mask，用 AR 模型直接蒸馏"）**：同样的 Ins rollout、同样的训练器、同样的预算
  （每条 canvas 3 帧，185 步，0.46 h），但破坏方式是随机**前缀切分**（前缀可见、后缀全 mask，loss 只在第一个 mask 位置 = AR 模型唯一有定义的地方），
  教师是 **Qwen/Qwen2.5-7B-Instruct**（在干净序列上取因果 logits；与 Dream 同一词表）。选择性保留（学生 p < 0.8 → 教师，否则 anchor）。它**确实**能教数学
  —— 后 300 题 46.7，rand100 51（vs Ins 26/9，p = 0.01）—— 但不如 over-addition 的 diffusion 教师（蒸馏 M 50.3 / 58；vs AR 9/16，p = 0.23），而且**伤指令跟随**：
  IFEval 52，vs M(.4,.6) 5/18（p = 0.01），vs 蒸馏 M 4/16（p = 0.01），尽管 Qwen2.5-7B-Instruct 本身 IF 很强。代码不变（57.3 = Ins）。训练特征也不同：
  在前缀切分态上学生只有 17 % 的位置"不确定"（gate 回放态上是 60 %），所以 AR 教师被问到的位置少得多。只试了一个未调的配方（没有 JustOPD 那种
  熵 top-25 % 或 L2R 学生 rollout）；全位置变体（`ARall`）和教师强度匹配的对照（一个在 MATH 上和 Qwen 一样强的 7B diffusion 教师）还没做。
- **AR 基线输在哪（2026-10-17 读数）。** 写完率、写完题准确率、长度来自后 300 题；IFEval 长度来自 rand100：

  | row | MATH rand100 | MATH 写完率 | 写完题准确率 | MATH 长度 (tok) | IFEval rand100 | IFEval 长度 (字符) | HumanEval+ |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Ins | 34 | 76 % | 48.5 | 375 | 62 | 1622 | 57.3 |
  | M(.4,.6) best merge | 42 | 81 % | 51.9 | 448 | 65 | 1753 | 61.0 |
  | A(2.0) 教师 | 55 | 94 % | 49.5 | 1005 | 51 | 4627 | 56.7 |
  | gate Ins ⇐ A(2.0) @0.8 (2×) | 51 | 93 % | 52.9 | 795 | 70 | 2375 | 59.8 |
  | M 蒸馏 (mixpool-1k) | 58 | 92 % | 54.7 | 639 | 64 | 2826 | 60.4 |
  | AR 教师基线 | 51 | **75 %** | 61.9 | 664 | **52** | 2381 | 57.3 |

  AR 基线的写完率就是 Ins 的（75 vs 76 %）；它写完题上的高准确率是**选择效应**，不是数学更好：在它和蒸馏 M 都写完的 218 题上两者都是 **62.8**；
  在 M 写完而它没写完的 58 题上 M 只有 24 %（难题）。所以 AR 教师没教会任何 diffusion 教师没教的东西，而 gate 的主要增益（写完率 76 → 92 %）
  恰恰是合法前缀态教不了的 —— 卡住发生在任意顺序解码的画布中段。IFEval 上丢的是约束类指令（no_comma 5、首尾引号 3、复述 prompt 3、占位符 3、
  回复语言 1）：回答读起来像泛泛的 Qwen 风格文本，忽略了 prompt 里的格式要求。结论：在现有证据下，**联合**（diffusion + AR 教师）蒸馏没有可叠加的
  互补信号 —— 不计划做。
- 解读：在现有证据下，选择性损失拿到了教师的数学，把它的指令跟随 / 代码损伤留在了后面。注意：一个种子、一个训练集、IF 只有 100 题子集；
  GSM8K 没跑。
- 原始数据：`outputs/step3/guards/<ins|m46|a20|gate|mixpool1k|ar-mixpool1k>/{ifeval_rand100,humaneval}.json`（逐题），`guards/math500_rand100_seed0_results.json`；
  AR 臂：`outputs/checkpoints/step3-mixpool-1k-AR/`（args、train_log、W&B 49u7anm9），MATH `outputs/step3/step3-mixpool-1k-AR/math500/`。

## 结果（1k 测试；表由 `code/step3/results_table.py` 生成 → `technical_details/step3_results_t1k.md`）

下表每一行都是**单模型**在同样 300 道 MATH-500 题（id 200–499）上解码，只有 gate 那一行是 2×（解码时两个模型）。
STEP0 的两个**端点基线**都在表里：best merge M(.4,.6)（convex 线）和 over addition A(2.0)（addition 线；它同时也是教师）。McNemar = 行赢次数 / 列赢次数，精确 p。

| row | acc | finish | acc\|finished | avg len (tok) | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | HW |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Ins（学生，未改动） | 36.7 | 76% | 48.5 | 375 | — | 37/36, p=1.00 | 26/42, p=0.07 | 32/62, p=0.00 | B300 |
| M(.4,.6) best merge（端点 1） | 42.0 | 81% | 51.9 | 448 | 42/26, p=0.07 | 45/28, p=0.06 | — | 35/49, p=0.16 | B300 |
| A(2.0) over addition（端点 2 = 教师） | 46.7 | 94% | 49.5 | 1005 | 62/32, p=0.00 | 58/27, p=0.00 | 49/35, p=0.16 | — | H200 |
| gate Ins ⇐ A(2.0) @0.8（2×） | 49.0 | 93% | 52.9 | 795 | 62/25, p=0.00 | 64/26, p=0.00 | 51/30, p=0.03 | 40/33, p=0.48 | H200 |
| **M** = Ins ⇐ A(2.0) 蒸馏（mixpool-1k 池），**θ = 0.8**，λ_anchor 0.3，reverse KL，lr 1e-4，2 ep，LoRA r64（1×） | **50.3** | 92% | 54.7 | 639 | 63/22, p=0.00 | 59/17, p=0.00 | 51/26, p=0.01 | 46/35, p=0.27 | H200 |
| **C2** = Ins 自蒸馏（对自己 rollout 的 token 做 CE，无教师），lr 1e-4，2 ep，LoRA r64（1×） | 36.3 | 80% | 45.6 | 856 | 36/37, p=1.00 | — | 28/45, p=0.06 | 27/58, p=0.00 | H200 |

**M 里的 θ（阈值）**是学生的原始置信度门槛，低于它这一帧就交给教师：在采样器下一步要写的位置上，c = max_v p_S(v | state)；
c < θ → 对 A(2.0) 做 reverse KL；c ≥ θ → 对未改动的 Ins 做 KL（权重 λ_anchor）。同一个 θ 也用来选训练帧（每条 canvas 的 3 帧里有 2 帧取自
rollout 时置信度 < θ 的那次写入之前）。θ = 0.8 是解码时 gate 的取值（STEP1 扫描）；臂 M06（θ 0.6）和 Msch（0.8 → 0.5）是计划中的 θ 消融。
参数：`--theta`、`--theta-end`、`--lambda-anchor`、`--kl`。avg len = 解码文本到结尾的平均 Dream token 数（canvas 固定 1024）。

按可信度排序，这张表说明：
1. **M ≫ Ins，M ≫ C2**（都是 p < 0.001，≈ 60 赢 vs ≈ 20）：蒸馏后的 1× 模型优于起点，原因是教师（自蒸馏什么都没带来）。
2. **M ≥ 两个端点。** M 胜过 best merge（51/26，p = 0.01），与 over addition / 教师至少持平（46/35，p = 0.27），
   而它只是 Ins 权重 + 一个 LoRA 的 1× 模型——测试时没有 merge、没有向量运算。
3. **M ≥ 它所蒸馏的 2× gate**（50.3 vs 49.0；gate 就是我们一开始要达到的天花板）。
4. 它具有 gate 的两个特征：写完率 76 → 91–92%（教师水平），写完题的正确率 48.5 → 55（高于教师的 49.5）。

训练来源：`outputs/checkpoints/step3-mixpool-1k-{M,C2}/`（`args.json`、`train_log.jsonl`、`run.log`）；W&B 项目 `a2d-step3`，
run `step3-mixpool-1k-M` / `step3-mixpool-1k-C2`（MATH 记为 `eval/math500_last300_*`）。注意：mixpool-1k 的 M 容器在完成前重启了两次（`run.log` 有三次启动、
一次 `[done]`）；checkpoint 来自唯一完整的那次，日志文件里也含有中断的尝试。

## 真正有用的评测需要什么（以及我们已有哪些）
| 要求 | 现状 | 动作 |
|---|---|---|
| 每行同一硬件 | Ins 和 M(.4,.6) 是 B300 的数（2026-09-30）；STEP3 各行是 H200 | 在 H200 上重测 Ins 和 M(.4,.6) 的后 300（2 个 job，≈ 5 GPU·h） |
| 整 300 题而不是 200 | M 的 shard 0、2 在跑 | 自动；表会重新生成 |
| 模型没有在别处变差 | IFEval rand100 64（Ins 62），HumanEval+ 60.4（Ins 57.3）—— 见“泛化” | GSM8K 和全 541 题 IFEval 仍未做 |
| 不是 seed 的偶然 | 一个 seed | M 换一个训练 seed（0.5 h + 7 GPU·h 评测） |
| 不是 1k 的偶然 | 一个数据量 | 8k offline（或 4 × 2k online）——同时回答 1k 是否已到平台 |
| 统计尊重配对 | 已做：相同题 id 上的精确 McNemar | 保持；不同题集之间永远不比 acc |
| 无泄漏 | 已查（上一节） | prompt 池一变就重查 |
| 能杀掉其他解释的对照 | C2（需要教师吗）已做；C1（需要选择吗）、C3（需要 over 吗）、C4（需要回放态吗）、C5（KL 方向）未跑 | guards 之后在同一批 rollout 上跑（每个 0.5 h 训练） |

## 训练配方（`code/step3/train_frontier_kd.py` 为 M 实际跑的设置，来自 `args.json` + `run.log`）

| item | value |
|---|---|
| 学生 / 教师 | Ins（`Dream-org/Dream-v0-Instruct-7B`）/ A(2.0)（`outputs/checkpoints/add-a10-b20`），都是 bf16；教师冻结、常驻显存 |
| 可训练参数 | LoRA r 64、α 128、dropout 0、无 bias，挂在每层的 q/k/v/o/gate/up/down 投影上（≈ 160 M 参数）；底座冻结 |
| anchor 模型 | 同一个网络关掉 LoRA（`disable_adapter`），不另存一份 |
| 数据 | 1 000 条 Ins rollout → 988 条 canvas（丢 12 条：回答 < 4 token 或只有一个写入步）；每条 canvas 每 epoch 取 3 帧（2 帧在 κ < θ 的写入之前，1 帧均匀）；prompt + 回答 > 1024 token 的帧跳过 |
| 步数 | 2 epoch × 988 × 3 = 5 928 帧 / 每步 32 帧 = **185 个优化器步** |
| batch | 每步 32 帧 = micro-batch 4（自动探测：8 在 H200·0.51 上 OOM，峰值 69.7 GB）× 8 次梯度累积 |
| 损失 | θ = 0.8：c < θ → reverse KL(p_S ‖ p_T)；c ≥ θ → λ·KL(p_S ‖ p_{S₀})，λ = 0.3；温度 1.0，不做 top-k 截断；两组位置分别取均值 |
| optimizer | AdamW，β = (0.9, 0.95)，ε 默认（1e-8），**weight decay 0** |
| lr 计划 | 峰值 1e-4，前 10 % 的步数线性 warm-up（18 步），之后线性衰减到第 185 步为 0 |
| 梯度裁剪 | 全局范数 1.0（只作用于 LoRA 参数） |
| 精度 | bf16 权重、bf16 autocast 前向、AdamW 里 LoRA 主参数 fp32、开 gradient checkpointing |
| seed | 0（取帧和 torch） |
| 成本 | 一张 H200·0.51 上 0.22 h，峰值 69.7 GB；每帧 0.13 s |
| 输出 | `lora/`（adapter）和 `merged/`（LoRA 折进 bf16 权重 + Dream 远程代码）—— 所有评测都加载 `merged/` |

没有用的东西：dropout、label smoothing、EMA、KL warm-up、θ 的 curriculum（Msch 臂会加）、教师 top-k、长度归一化。
每步监控（`train_log.jsonl`，W&B `train/*`）：`kd`、`anchor`、`ce`（对 rollout token 的 CE，不优化）、`unsure_rate`、`empty_frames`、`lr`、`peak_gb`。

## 训练代码是怎么来的
为这一步专门写的，不是改自某个 trainer：`code/step3/train_frontier_kd.py` 约 300 行纯 PyTorch + HF `transformers` + `peft`（LoRA），
单卡，没有 Lightning、没有 verl、没有 accelerate。原样复用的部分：Dream 官方采样器做 rollout（`diffusion_generate`，与 MATH-500 协议同设置）、
Dream 的 logit 约定（位置 p 由隐层 p−1 预测，与官方 SFT trainer 和我们的解码器一致）、peft 的 LoRA 用法与官方 `run_sft_tulu3.sh` 相同、
STEP1 的 lockstep 解码器做评测。新写的部分：轨迹回放作为破坏方式、按置信度分流的 loss、self-anchor、帧采样器、OOM micro-batch 探测、
以及合并/保存路径（`merge_lora.py`，绕过 Dream 的 `generation_config.validate`）。没有用官方 verl FSDP trainer，因为它的 loss
（对参考答案做随机 mask CE）正是这一步要替换掉的东西，而且它需要 4 张卡。

## 无效的 tulu3-1k run（2026-10-16）—— 哪里错了，还能说明什么

本意：同一配方跑在 1 000 条只用 Tulu-3 的 prompt 上。两个流水线 bug 让它变成了别的东西：
1. prompt 文件路径是用环境变量传的，而环境变量到不了 job 容器 → rollout job 解码的是 **mix17k 池**（和 mixpool-1k 相同的 prompt）；
2. `rollouts.jsonl` 是增量追加的，snakemake 重启后已存在的文件被当成已完成 → `all.jsonl` 只拼了 **122 行**。
于是“M (tulu3-1k)”实际是在 **119 条 mix 池 canvas 上训了 22 个优化器步** —— 却仍在后 300 题上拿到 **50.3 / 写完 91 % / 813 token**
（vs M(mixpool-1k) 33/33，p = 1.0）。产物移到 `trash/n1k_invalid_20261016/`（有 README）；W&B 的 `step3-tulu3-1k-M` 就是这次无效 run。
修复（已提交）：prompt 文件改为 runner 的显式参数；rollout shard 写 `done` 标记、Snakefile 等这个标记；MATH shard 只有 `summary.json` 存在才算完成。
只用 Tulu-3 的 run 用修好的流水线、同一 tag 重新排队（15 个 job，≈ 20 GPU·h）。
这次事故说明的事：MATH 增益在 ~700 帧后就到了 —— 1 k / 185 步的预算对这个指标早已过饱和。所以下一次 run 里有信息量的是 guards（代码 / IF）和长度，不是 MATH。

## 相关：把 AR 教师蒸馏进 diffusion 学生（为什么我们不直接这么做；2026-10-16）

同事的问题：为什么不直接蒸馏 AR 模型（Qwen）？已经有人做了 —— 而且每一种方法都得退回到**合法前缀态**，因为 AR 模型只定义 p(x_j | x_<j)，
无法以一张部分填充、任意顺序的画布为条件：

| work | 怎样让 AR 教师用得上 | 结果 |
|---|---|---|
| JustOPD (arXiv 2610.17397) | 学生（Dream-7B）**训练时从左到右 rollout**，每个状态都是前缀；Qwen3-14B 教师在前缀上打分；只在教师熵最高的 25 % 位置算 loss；推理时恢复任意顺序 | Dream-7B MATH-500 @1024：36.8 → 48.0（JSD），43.8（reverse KL） |
| ForkLeft (arXiv 2609.32448) | 熵优先 rollout，然后把已写部分**固定为前缀**，教师在同一前缀下监督 | Efficient-DLM-4B MATH-500 72.6 → 79.6（教师 Qwen3-30B-A3B） |
| OPDLM (arXiv 2606.06712) | 学生是 block-diffusion；教师只看**最终序列 x₀ 的因果前缀**，从不看带 mask 的状态 | AR → DLM 转换，训练 token 少 15–7000× |
| TIDE (arXiv 2604.26951) | 跨 tokenizer 的 chunk 级似然匹配 | 8 个 benchmark 平均 +1.5 |
| Distilled DLMs / Target Concrete Score（ICLR'25 投稿） | 用 AR 模型构造 target concrete score | 早期 |

JustOPD 把这个约束说得很明白：中间去噪态"一般不对应 AR 训练分布下的合法前缀"。我们的教师 A(2.0) 本身就是 diffusion 模型，所以它能在
**学生真正会到达的状态**（gate 的状态，置信度顺序，双向上下文）上、在 gate 选出的位置上监督学生 —— 这是 AR 教师做不到的。序列级 AR 蒸馏
（对 AR 输出做 SFT）是另一条路，A2D 的失败清单和我们计划中的 C7 对照覆盖它。

可直接对比的数字：JustOPD 用 14B AR 教师把 Dream-7B 在 MATH-500 @1024 上从 36.8 推到 48.0；我们用 7B 的 over-addition 教师、1 k prompt、185 步
把 Ins（我们后 300 上 36.7）推到 50.3。并排引用前需要在同一题集上重跑（TODO）。

**我们自己也跑了（2026-10-17，`train_frontier_kd.py` 的 `AR` 臂）：** `--corruption l2r --teacher-type ar`，教师 Qwen2.5-7B-Instruct，与 M 相同的
Ins rollout / 选择 / anchor / 预算。结果（完整一行见"泛化"）：MATH 后 300 **46.7**（M 50.3，Ins 36.7），IFEval rand100 **52**（M 64，Ins 62），
HumanEval+ 57.3（M 60.4）。所以在我们的数据上，AR 教师教数学不如 over-addition 的 diffusion 教师，*而且*伤指令跟随，哪怕纸面上它是更强的教师。
注意：只有一个未调的配方；AR 教师只在 17 % 的位置被问到（学生在合法前缀态上很少不确定）。

## 当前数字的注意事项
- 每一行都已齐 300 题（mixpool-1k 的 merge 规则在两个重跑 shard 完成前就触发了；2026-10-16 13:30 手动合并）。
- Ins 行在 **B300** 上解码（2026-09-30），M 和 C2 在 **H200·0.51** 上。规则是每张表只有一个硬件列；在对外引用任何数字之前，
  下一个要申请的任务是在 H200 上对 Ins 重跑最后 300 题。
- C2 的输出更长且部分退化（重复 "The final answer is …"；平均 3 029 字符 vs Ins 888）；对于普通 self-SFT，lr 1e-4 偏高。
  这不改变其准确率（38.0），但意味着 C2 是一个保守的对照，而非调优过的对照。
- 计划中的读数规则仍然适用：成功 = ≥ Ins + 6 **且** > C2 + 3 **且** 无 guard 回退。

## TODO（2026-10-16，@Amel —— 请看 / 补充）

**Validation set —— 已完成 2026-10-16（用户："需要 validation 数据来确认多阶段每一轮是否有意义"）。**
三份固定的随机 100 id 列表在 `code/step3/guards/`，全部与报告用的测试集不相交；一个 checkpoint = 3 个并行 job，
`modules/submit_step3_validation.sh <ckpt> <label>`（≈ 6.5 GPU·h ≈ 3.3 CHF），结果在 `outputs/step3/validation/<label>/{math,ifeval,mbpp}.json`：

| family | list | protocol | GPU·h |
|---|---|---|---:|
| 数学 | `val_math_test_minus500_rand500_seed0.jsonl` 第 0–99 行 —— **MATH test 集去掉 MATH-500**（4 500 题；MATH-500 本来就是从这个 split 均匀随机抽的，难度分布一样：L1–L5 = 9/18/21/26/27 % vs 池子的 10/19/21/25/25 %；和所有报告过的题目、和 rollout 池（MATH *train*）都不交叉）里随机 100。由 `build_math_val.py` 生成（官方 minerva gold 抽取，在 MATH-500 全部 500 题上核对过；prompt 模板与官方逐字节一致）。**替代** 2026-10-16 的 MATH-500 id 0–199 列表 —— 用户 2026-10-17 否决，因为 rand100-of-500 测试列也抽到 id 0–199（`trash/val_math500_first200_rejected_20261017/`） | 测试用的同一解码器（lockstep free @1024），`modules/run_math_val.sh` | 1.5 |
| 指令遵循 | `val_ifeval_rand100_seed1_disjoint.json` —— 不在报告用 `ifeval_rand100` 里的 441 题中随机 100 | 官方 1280/1280，task `ifeval_val100` | 3.5 |
| 代码 | `val_mbppplus_rand100_seed0.json` —— MBPP+ 378 题中随机 100（HumanEval+ 保持为测试集） | 官方 evalplus dllm 512/512，`evalplus_subset.py` | 0.6 |

新 math val100 上的参考点正在解码（Ins、蒸馏 M；2026-10-17）；旧列表上的免费数字（Ins 46、M(.4,.6) 51、A(2.0) 61、gate 65）不再适用。
**规则（用户 2026-10-17）：以后所有训练变体都只在这套 validation 上调 —— 配方搜索、θ、帧数、多阶段每一轮、tulu3-1k、对照。** 配方搜索（2026-10-17，`modules/submit_step3_recipe.sh` = 训练 + validation）：每次相对 M 只改一处 —— epoch 4、随机一半 rollout（500，seed 0）、θ 0.6、θ 0.4。测试集只在 validation 选定配方后碰一次。
没有采用：GSM8K（太简单，merge 下反向变化）。手写 sanity prompt：待加。

**Amel 的第一个任务，优先于下面所有条目：** 在只用 Tulu-3 的池子上训练蒸馏模型（`tulu3-1k`：直接从 Tulu-3 按原生比例抽 1 000 条 prompt，不加别的），
配方原样不动（Ins ⇐ A(2.0)，θ = 0.8，λ = 0.3，2 epoch），然后在三类任务上测泛化：**数学**（MATH-500 后 300、GSM8K）、**代码**（HumanEval+）、
**指令遵循**（IFEval）。问题只有一个：只用 Tulu 的学生是否保住 mixpool-1k 的 MATH 增益，*并且*在代码 / IF 上没有损失。流水线：`code/step3/pipeline`
（tag `tulu3-1k`，臂 M；guards 通过 `guard_arms`）。2026-10-16 状态：还没做成 —— 第一次 tulu3-1k 训在了错误的池上（见“无效的 tulu3-1k run”）；修好的流水线可以重启。列表里其余条目都在这个答案之后。

按"对结论改变什么"排序。成本为 H200·0.51 GPU·h。两个大项（θ 搜索、多阶段蒸馏）在下面单独成小节。
0. **先定一个配方**（用户 2026-10-16）—— 现在的配方（1 k prompt、2 epoch、1024 token 序列、θ 0.8、λ 0.3）从来没调过，之后所有 run 都应该
   用同一个不变的配方。序列长度固定 1024（官方 MATH-500 设置；不要动）。只改训练量，一次一个旋钮，同样的 MATH 后 300 + guards：
   (a) **epoch** 2 → 4（同一批 rollout；370 步，≈ 0.5 h 训练 + 7 GPU·h 评测）；
   (b) **rollout 序列数** 1 k → 2 k prompt（新 rollout，2 epoch；≈ 10 GPU·h rollout + 评测）；
   (c) **每条 rollout 的帧数** 3 → 6。一条 rollout 不是当成一整个序列来训：从它的 512 个去噪步里挑 3 个中间状态（“帧”：低置信度 commit 前的 2 个 +
   均匀随机 1 个），每一帧是一个训练样本。翻倍 = 同一批 rollout 出更多样本，不用重新解码（`--low-frames 4 --uniform-frames 2`，≈ 0.5 h 训练 + 7 GPU·h 评测）。
   顺序 / 跑 (a)–(c) 中的哪些由 Amel 决定。都没比 50.3 高出噪声以上，185 步配方就是最终配方；有一个更好就用那个。这一步在 θ 搜索和多阶段之前做。
1. **干净训练集**（`tulu3-1k`）—— 2026-10-16 第一次尝试无效（见“无效的 tulu3-1k run”）；用修好的流水线重新排队，≈ 20 GPU·h。然后 guards。
2. **θ 搜索** —— 见下面 §TODO-θ。≈ 8 GPU·h/点。
3. **多阶段蒸馏** —— 见下面 §TODO-multi-stage。GPU·h 与 offline 相同，墙钟串行。
4. **基线同硬件** —— Ins 与 M(.4,.6) 是 B300 数字；在 H200 重测（后 300，2 个 job，≈ 5 GPU·h）。
5. **Guards** —— IFEval rand100 / HumanEval+ 已做（见“泛化”）；还没做：GSM8K、IFEval 全 541 题、长度读数（M 比 Ins 多 80% token）。
6. **其余对照**，共用 rollout（每个 ≈ 8 GPU·h）：C1 不分裂、C3 教师 A(1.1)、C4 随机 mask、C5 forward KL；C2 换小 lr。
7. **第二个 seed + 8 k prompts** —— 1 k 是否已是平台，50.5 是否稳定。
8. **Trainer 断点续训** —— mixpool-1k 的 M 容器静默重启了两次；任何数小时 run 之前加周期性 LoRA + optimizer 存档。
9. **第二条学生线** —— M(.4,.6) 起点（臂 Mm），之后换模型族（DreamReasoner / LLaDA-Instruct）。

### TODO-θ：阈值搜索
目前原样沿用 gate 的 θ = 0.8。按 0.2 步长扫（0.1 太贵）：θ ∈ {0.4, 0.6, 0.8}，再加衰减计划 0.8 → 0.5（`--theta-end`，臂 Msch）。
rollout 共用，每个点 = 0.5 h 训练 + 7 GPU·h MATH。看：acc、写完率、平均长度、`unsure_rate`（交给教师的位置占比）—— 后者告诉我们更低的 θ 是否真的意味着更少教师。

### TODO-multi-stage：online 蒸馏（用更新后的学生 rollout）

**一句话。** 现在（offline）：S₀ = Ins 把 1 000 条 prompt 全部 rollout 一次，然后训练一次 → S₁。Online：把 prompt 切成 K 块；
第 1 块用 S₀ rollout，训练 → S₁；第 2 块**用 S₁** rollout，接着训练 → S₂；…… → S_K。教师 T、损失、θ 都不变，变的只有*谁来写轨迹*。

**记号。** S₀ = Ins；T = A(2.0)，冻结；S_k = 第 k 轮之后的学生（S₀ 加一个持续训练的 LoRA，所以 S_k 从 S_{k−1} 出发）；
Q_k = 第 k 块 prompt；p_S(·|x, j) = 模型 S 在 canvas x 位置 j 的分布；c_j = max_v p_S(v|x, j) = 学生在该处的置信度。

**第 k = 1, …, K 轮**
1. **Rollout。** S_{k−1} 用官方采样器（512 token / 512 步）解码 Q_k 里的每条 prompt；记录 canvas，以及每个位置的写入步和写入那一刻的置信度。
2. **取帧。** 每条 rollout 取 3 个训练状态 (x, NEXT)：2 个取在置信度 < θ 的写入之前，1 个取在均匀随机的一步 —— 与 offline 相同。
3. **训练。** 从 S_{k−1} 出发，在这些帧上

   S_k = argmin_S  E_{(x, NEXT)} Σ_{j ∈ NEXT} [ 1[c_j < θ] · KL( p_S(·|x,j) ‖ p_T(·|x,j) )  +  λ · 1[c_j ≥ θ] · KL( p_S(·|x,j) ‖ p_{S₀}(·|x,j) ) ]

   即 S 不确定的地方向 T 靠，S 确定的地方按在 S₀ 上。c_j 用正在训练的 S 计算。

第 K 轮结束后合并 S_K，评测方式与 offline 模型相同。K = 1 就精确等于 offline run。

**Initial plan。** 1 000 条 Tulu-3 prompt（tulu3-1k 池），K = 4 块、每块 250，θ = 0.8，λ = 0.3，每轮 2 epoch，lr 1e-4、只在第 1 轮 warm-up；
在同一 MATH 后 300（+ guards）上比较 S₄ 与 offline 的 S₁。GPU·h ≈ offline（每条 prompt 解码一次、每帧训练一次）；墙钟 ≈ 4 × (1.4 h rollout + 0.1 h 训练)。
期望收益：从第 2 轮起，"不确定的位置"是学生*现在仍然*会错的，而不是已经修好的（TOPD 的 on-policy 论证）。代价：对照组 C1 … C5 只能共用第 1 轮的 rollout。

**每轮的 validation 关卡（用户 2026-10-16）。** 每一轮 k 结束后对 S_k 跑 validation（`submit_step3_validation.sh`，≈ 6.5 GPU·h）。
只有当 MATH-val(S_k) ≥ MATH-val(S_{k−1}) 且 IFEval-val、MBPP+-val 都没掉超过 3 分，才进入第 k+1 轮；否则停下、保留 S_{k−1}。
测试集只碰一次，用 validation 选出的那一轮。

**以后 / 需要**
- 变体：移动教师 T_k = 在 S_k 之上重新施加 over 步。
- 可选：递减的 θ_k（TODO-θ）；先分开测。
- 代码：`rollout.py --lora`（带 adapter 解码）、trainer 的 `--resume-lora`、Snakefile 按轮依赖。

## 下一步
1. 完成 300 题（自动）。2. Ins 在 H200 上跑最后 300 题（一个任务）。3. 对 M 跑 guard。4. 8 k 离线 vs 4 × 2 k 在线轮次（`tmp.md` 2026-10-15 计划），
M06（θ 0.6）和 Mm（M(.4,.6) 初始化）作为下一批 arm。
