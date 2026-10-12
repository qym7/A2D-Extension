# DreamReasoner-8B：over-merge 到底牺牲了什么？—— 用论文自己的 AR-help / AR-harm 分解

零 GPU。全部数字来自 `dream_study/step7_reasoning_merge/outputs/` 下已存在的结果。
分析脚本（临时，不在 dream_study 下）：
`/tmp/claude-256458/-mnt-lts4-scratch-home-yqin-DllmAR/11cea7f4-d429-4fd7-914c-a728e9fb8a6c/scratchpad/hh{,2,3,4}.py`
凸组合 `D_b+(1−λ)τ_D+λτ_A` ⇒ **τ_D = 1−λ**。未提交任何 GPU 作业，未修改 `dream_study/` 下任何文件。

---

## 1. 定义的出处（原文引用，不是猜）

项目里存在**两个不同**的 help/harm 定义，必须分清。

### 1A. donor-conditioned「AR-help / AR-harm adoption」—— 这是论文 Fig.1（`fig:curves`）用的

定义脚本 `dream_study/step2_instruct/scripts/behavioral_inheritance.py:105-113`：

```python
ar_help = [key for key in task_ids if diffusion[key] == 0 and ar[key] == 1]
ar_harm = [key for key in task_ids if diffusion[key] == 1 and ar[key] == 0]
...
help_adoption = _mean([scores[key] for key in ar_help])
harm_adoption = _mean([1 - scores[key] for key in ar_harm])
```

文字定义 `dream_study/iclr/figures/curves_inherit/README.md:4`：
> "help adoption = fraction of donor-right/parent-wrong items the merge answers correctly; harm adoption = fraction of parent-right/donor-wrong items it answers incorrectly."

`dream_study/step3_selective_transfer/README.md:17-20`：
```text
AR-help = Dream wrong, AR correct
AR-harm = Dream correct, AR wrong
```
并明确要求保留 `(D, A, M)` 八态，否则会把 merge-only synergy 误记到 AR 头上。

报告位置：
- 图：`dream_study/iclr/main.tex:326`（`\label{fig:curves}`），脚本 `dream_study/iclr/figures/curves_inherit/plot_curves_inherit.py:27-28`。
- 表：`dream_study/step2_instruct/results/README.md:106-125`（IFEval / HumanEval 的 `AR-help adoption | AR-harm adoption | AR-agreement` 栏）。
- 正文论断 `dream_study/iclr/main.tex:385`："the control adopts wins and errors *symmetrically* (base rate), while the merge is asymmetric --- it takes substantially more of the donor's wins while taking fewer of its errors."

**「两者同时增长」这句话的确切出处是 `dream_study/README_ZH.md:27-28`**：
> "4. **增加 AR 权重会增加 behavior inheritance，但不一定增加有用行为。** AR-help 与 AR-harm 会同时被引入，而且 AR agreement 可能在 accuracy 过峰值后继续上升。"

以及 `dream_study/step2_instruct/results/README.md:123-124`：
> "AR agreement can continue increasing after task accuracy peaks. A single global beta therefore cannot select AR-help without also importing AR-harm; this motivates Step 3."

> ⚠️ **重要澄清**：我在 `iclr/overleaf/*.tex` 全文 grep `harm` 只有一处无关命中
> （`appendix_models_evals.tex:21`，讲 format breaking）。**"AR help 与 AR harm 都随系数增长" 这句话并不在投稿的
> appendix tex 里**，它在 `README_ZH.md` / `step2_instruct/results/README.md` 的 ledger 里，图 `fig:curves`
> 只画曲线不写这句结论。而且该 ledger 的证据**全部是 Dream-v0 的 IFEval 与 DreamCoder 的 HumanEval，
> 没有 DreamReasoner**。见下表（抄自 `step2_instruct/results/README.md:106-121`）：

| 域 | β | acc | AR-help adoption | AR-harm adoption |
|---|---:|---:|---:|---:|
| IFEval | 0.20 | 0.606 | 0.433 | 0.212 |
| IFEval | 0.60 | **0.665** | 0.535 | 0.231 |
| IFEval | 0.80 | 0.651 | **0.559** | 0.308 |
| IFEval | 1.00 | 0.490 | 0.409 | **0.519** |
| HumanEval | 0.25 | **0.854** | 0.615 | 0.222 |
| HumanEval | 0.40 | 0.823 | **0.692** | **0.333** |

即：在 IFEval/HumanEval 上，help 升到 β=0.8 还在升、harm 升得更慢但更晚爆炸。**本报告要验证的就是这条在
DreamReasoner 上是否成立。**

### 1B. parent-conditioned help/harm 计数 —— 本报告用的主定义

定义脚本 `dream_study/step3_selective_transfer/scripts/ar_help_harm_curve_instruct_coding.py:5` 与
`.../ar_help_harm_curve_math.py:7-9`：

```text
help(b) = #{ parent WRONG & merge RIGHT };  harm(b) = #{ parent RIGHT & merge WRONG }
```

`dream_study/iclr/behavior_analysis.tex:48` 明确区分两者：
> "Note that these rates are conditioned on the diffusion parent; the donor-conditioned adoption curves of Fig.~\ref{fig:curves} ..."

本报告主表用 1B（它直接回答"over-merge 牺牲了什么"），§4 另给 1A 的 MATH-500 版本。

---

## 2. 协议分组（先读，否则表读不了）

同一模型换输出目录后缀差 2.6–2.8 分，这个跨度比整条 λ 扫描还大
（已记录于 `new_ideas/idea1_selective_distill/results/DREAMREASONER.md:8-17`）。根因是 **token 预算**，
从 `results.json` 的 `args.max_new_tokens` 直接可读：

| arm | 目录 | max_new_tokens | no_thinking | acc |
|---|---|---:|---|---:|
| parent | `math500/parent`（无后缀） | 4096 | – | 89.2 |
| parent | `math500/parent-m500-8k` | 8192 | False | **93.6** |
| parent | `math500/parent-m500-24k` | 24576 | False | 94.4 |
| parent | `math500/parent-m500-nothink-8k-h200` | 8192 | **True** | 88.8 |

本报告**只用两组**，组内 `args` 完全一致（已逐一核对）：
- **MATH-500 组**：`-m500-8k` 后缀，8192 tok，greedy（T=0, top_p=1），block=32，conf=0.95，thinking-on。
- **AIME 组**：`aime/{parent,rmg-b010..b060}`，8192 tok，greedy，block=32，conf=0.95，thinking-on，
  6 个 arm 的 `args` 字典除 `model`/`out` 外完全相同。
- 绝不混 `nothink`（parent 88.8 vs 93.6）。
- `drtr-*` / `drbase-*` / `rmg-add-*` 是 **Base-anchored（`D_b+β·τ_A`，τ_D=0）的 transfer-only 基线**，
  单列、不参与 over-merge 结论：`drtr-m500-8k` 76.0、`drbase-m500-raw-8k-b300` 68.0、
  `rmg-add-a05-m500-8k-b300` 76.0（全为 8192 tok）。Base 本身不会自己停，任何从它们推出的
  "over-merge" 结论都被这个混淆污染。

---

## 3. 主结果：parent-conditioned help / harm

### 3.1 AIME24+25（n=60，`aime/` 组）—— 有信号的那个

数据源：每 arm 的 `eval.log`，行格式 `[aime24 7/30] gold=X pred=Y OK|x tok=N`（配对到 (task, i)）。
显著性：**配对 exact McNemar**（只用两 arm 都答了的题，这里 60/60 全配对）。

| λ | τ_D | n | acc | help | harm | net | McNemar n | p（双侧 exact） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0（parent） | 1.0 | 60 | 51.7 | – | – | – | – | – |
| 0.1 | 0.9 | 60 | 63.3 | 11 | 4 | **+7** | 15 | 0.119 |
| **0.2** | 0.8 | 60 | **68.3** | **13** | **3** | **+10** | 16 | **0.021** |
| 0.3 | 0.7 | 60 | 46.7 | 4 | 7 | −3 | 11 | 0.549 |
| 0.4 | 0.6 | 60 | 56.7 | 8 | 5 | +3 | 13 | 0.581 |
| 0.6 | 0.4 | 60 | 36.7 | 5 | **14** | **−9** | 19 | 0.064 |

按年份拆（证明不是单年噪声）：

| λ | aime24 acc | aime25 acc | help24 | harm24 | help25 | harm25 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 53.3 | 50.0 | – | – | – | – |
| 0.1 | 73.3 | 53.3 | 6 | 0 | 5 | 4 |
| 0.2 | 80.0 | 56.7 | 8 | 0 | 5 | 3 |
| 0.3 | 50.0 | 43.3 | 3 | 4 | 1 | 3 |
| 0.4 | 60.0 | 53.3 | 4 | 2 | 4 | 3 |
| 0.6 | 40.0 | 33.3 | 4 | 8 | 1 | 6 |

**结论（对论文论断的裁决）：在 DreamReasoner / AIME 上，"help 持续上升、harm 上升更快"这条
被证伪（refute）。** harm 确实单调性地变坏（4→3→7→5→**14**），但 **help 不是持续上升：它在 λ=0.2 达到
13 之后掉到 4–8，λ=0.6 只剩 5**。也就是说 over-merge 不是"好处和坏处一起放大"，而是
**好处先饱和再消失、坏处单独爆炸**。这与 IFEval（help 一路升到 β=0.8）定性不同。

旁证（与既有记录一致）：`DREAMREASONER.md:34-48` 以 λ=0.2 为参照的配对结果
λ=0.3 −21.7 (p=0.001)、λ=0.6 −27.3 (p=0.001)，Bonferroni 0.05/5 下两者都存活；
而对 parent 的配对（我这里的 λ=0.2 行，p=0.021）不过 Bonferroni。两套数字互相印证。

### 3.2 MATH-500（n=500，`-m500-8k` 组）—— 饱和、无信号

| λ | τ_D | n | acc | 写完acc | 不可解析 pred | help | harm | net | McNemar n | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0（parent） | 1.0 | 500 | 93.6 | 95.9 | 12 | – | – | – | – | – |
| 0.1 | 0.9 | 500 | 93.2 | 95.9 | 14 | 15 | 17 | −2 | 32 | 0.860 |
| 0.2 | 0.8 | 500 | 94.8 | 96.3 | 8 | 15 | 9 | +6 | 24 | 0.307 |
| 0.3 | 0.7 | 500 | 94.0 | 96.5 | 13 | 19 | 17 | +2 | 36 | 0.868 |
| 0.4 | 0.6 | 500 | 94.2 | 96.5 | 12 | 19 | 16 | +3 | 35 | 0.736 |

**MATH-500 饱和**：acc 93.2–94.8、写完acc 95.9–96.5，每个 λ 的配对 p ≥ 0.31。
help 15→15→19→19 微升、harm 17→9→17→16 持平 —— 方向上勉强不反对论文论断，但**整条曲线在
一个标准误内是 null，不能当证据**。λ 扫描在 MATH-500 上也没有 λ≥0.5 的点。
→ **AIME 是唯一信息量 benchmark；MATH-500 只能当"这里没有可牺牲的东西"的对照。**

唯一单调的通道是长度（下表，`resp` 字符数，来自 `results.partial.jsonl`）：

| λ | 中位 resp 字符 | p90 | 不可解析 | dup-shingle≥0.3 |
|---|---:|---:|---:|---:|
| 0 | 1518 | 8058 | 12 | 5 |
| 0.1 | 1646 | 7781 | 14 | 6 |
| 0.2 | 1743 | 8075 | 8 | 3 |
| 0.3 | 1725 | 7915 | 13 | 3 |
| 0.4 | 1907 | 9078 | 12 | 1 |

长度 +26%（AR 确实让它写长），但 8192 预算下还没有把写完率压下去，所以**在 MATH-500 上长度通道
是"已点火但未起效"**。退化重复（dup-shingle≥0.3）并不随 λ 增加（5→1），**MATH-500 的 λ≤0.4 范围内
没有退化重复问题**。

---

## 4. donor-conditioned 版本（论文 Fig.1 的定义）在 DreamReasoner 上

donor = Qwen3-8B，唯一有逐题标签的文件是
`step7_reasoning_merge/outputs/math500/qwen3-math-full-peritem-h200/results.json`（`results[0].per_q[*].frac`，
avg@4，T=0.6/top_p=0.95/32768 tok）。donor-correct 取 `frac ≥ 0.75`。
**协议与 diffusion 侧完全不同（采样 vs greedy、32768 vs 8192 tok），所以这组数字只能当定性参考。**

AR-help 集 |D=0,A=1| = 20；AR-harm 集 |D=1,A=0| = 11。

| λ | help adoption | harm adoption |
|---|---:|---:|
| 0 | 0.000 | 0.000 |
| 0.1 | 0.650 | 0.364 |
| 0.2 | 0.600 | 0.273 |
| 0.3 | 0.800 | 0.273 |
| 0.4 | **0.900** | 0.364 |

定性上与论文的"asymmetric adoption"一致（help adoption 随 λ 升到 0.90，harm adoption 停在 0.27–0.36），
**但 n=20 / n=11、且跨协议**，不能据此做任何定量论断。AIME 上**无法做这件事**：
`aime/qwen3-aime-full/results.json` 只有 `avg@64` 的汇总 accuracy，**没有 `per_q`**。

---

## 5. 被帮助 / 被损害的题长什么样

### 5.1 MATH-500（有 `resp` 全文）

merge 自己的输出，pooled λ=0.1–0.4（127 个 discordant 事件）：

| 集合 | n | 中位字符 | p90 字符 | ≥7000 字符 | 不可解析 | dup-shingle≥0.3 |
|---|---:|---:|---:|---:|---:|---:|
| 被帮助（merge 输出） | 68 | **5364** | 15904 | 28 | **0** | **0** |
| 被损害（merge 输出） | 59 | **13401** | 22663 | 41 | **28** | 9 |

逐 λ：

| λ | 集合 | n | 中位 resp 字符 | dup≥0.3 | 不可解析 pred |
|---|---|---:|---:|---:|---:|
| 0.1 | help | 15 | 6189 | 0 | 0 |
| 0.1 | harm | 17 | 13431 | 5 | 8 |
| 0.2 | help | 15 | 5725 | 0 | 0 |
| 0.2 | harm | 9 | 13913 | 2 | 5 |
| 0.3 | help | 19 | 4403 | 0 | 0 |
| 0.3 | harm | 17 | 11850 | 2 | 9 |
| 0.4 | help | 19 | 5061 | 0 | 0 |
| 0.4 | harm | 16 | 12954 | 0 | 6 |

（parent 全局中位 1518 字符、12 题不可解析、5 题 dup≥0.3。）

**关键的方向性事实**：
- 被帮助的题：parent 中位 **11127** 字符 → merge 中位 **5364**，merge **更短**的占 47/68。
  即 help = "parent 原本在长篇乱绕 / 绕不出来，merge 让它更快收敛并写出答案"。
- 被损害的题：parent 中位 **8060** → merge 中位 **13401**，merge **更长**的占 48/59。
  即 harm = "parent 本来能收住，merge 让它写不完 / 绕飞"。
- 被损害的 59 个事件里，**28 个是答案完全解析不出来**（未终止 / 截断），31 个是解析出了但答错
  （真正的推理损失）。**所以 MATH-500 上的 over-merge 伤害大约一半是"不终止"，一半是"算错"。**
- "被损害的题" vs "没被动的题"在 parent 侧的长度差极大：中位 8578 vs 1352 字符。

按 **parent 输出长度**分桶（pooled λ=0.1–0.4，每格是事件数）：

| parent 字符 | 题数 | parent acc | help 事件 | harm 事件 | help 占比 |
|---|---:|---:|---:|---:|---:|
| 0–1000 | 168 | 98.8 | 6 | 1 | 0.86 |
| 1000–2000 | 125 | 99.2 | 2 | 5 | 0.29 |
| 2000–4000 | 100 | 97.0 | 8 | 8 | 0.50 |
| 4000–8000 | 57 | 89.5 | 7 | 15 | 0.32 |
| 8000–16000 | 33 | 72.7 | 25 | 20 | 0.56 |
| ≥16000 | 17 | 35.3 | 20 | 10 | 0.67 |

**读法**：长输出区（≥8000 字符，50/500 题）集中了 45/68 的 help 和 30/59 的 harm —— 它是
**"易变区"，而不是"help 区"**。在易变区内部，help:harm ≈ 56:44 / 67:33，**没有把两者分开**。

### 5.2 AIME（**没有 `resp`，只有 pred**）

`aime/{parent,rmg-b0*}/` 目录只有 `eval.log` + `results.json`（2026-08-27 跑的），
**没有 `results.partial.jsonl`，没有保存任何生成文本**。保存 `resp` 的功能是
`step7_reasoning_merge/scripts/dreamreasoner_aime.py:274` 加的（文件日期 2026-09-22，
另有 `.bak_20260917`），晚于这批 run。所以 AIME 上**无法**做：
长度分析、dup-shingle 退化重复检测、未终止 vs 算错的区分。**这是本报告最大的空洞。**
`eval.log` 里的 `tok=8192` 对所有 arm 所有题都是 8192（= 预算），是预算不是实际长度，**不能当长度用**。

能从 pred 本身看出来的（这是唯一可用的通道）：

| λ | pred=='0' | pred 位数>3 | pred ∉ [0,999]（AIME 合法区间外，含空/None） |
|---|---:|---:|---:|
| 0（parent） | 3 | 3 | 3 |
| 0.1 | 1 | 0 | 0 |
| 0.2 | 1 | 1 | 1 |
| 0.3 | 0 | 2 | 3 |
| 0.4 | 2 | 4 | 6 |
| 0.6 | 2 | 6 | **7** |

被损害题的 pred 值（gold 对照），**harm 的"性格"随 λ 改变**：

| λ | 被损害的 (题, pred→gold) |
|---|---|
| 0.1 | a25#7 1→821；a25#22 **217→237**；a25#27 4→19；a25#29 6→104 |
| 0.2 | a25#7 12→821；a25#24 **159→149**；a25#29 2→104 |
| 0.3 | a24#4 56→809；a24#17 4→721；a24#19 42→468；a24#29 2→104；a25#7 41→821；a25#24 **150→149**；a25#29 **30576**→104 |
| 0.4 | a24#29 **18267**→104；a24#30 **221→321**；a25#2 **804**→588；a25#7 **1016**→821；a25#24 **130→149** |
| 0.6 | a24#1 89→33；a24#4 1→809；a24#8 **000**→601；a24#9 **None**→25；a24#15 **None**→294；a24#17 31→721；a24#25 23→73；a24#29 16→104；a25#1 7→70；a25#2 4→588；a25#7 **None**→821；a25#17 **None**→49；a25#21 **None**→293；a25#29 2→104 |

- λ=0.1–0.4：harm 以**"差一点"的三位数近答案**为主（217 vs 237、159 vs 149、150 vs 149、221 vs 321、
  1016 vs 821）＋ 偶发的"抓错数字"（30576、18267）。
- λ=0.6：harm 变成**个位数 / `000` / `None`**（4 题完全没有答案）→ **崩塌式失败**。
  这与写完率从 100% 掉到 91.7%（`DREAMREASONER.md:41`）一致。

**所以 over-merge 牺牲的东西是分段的**：λ ≤ 0.4 牺牲的是**算准**（near-miss 推理滑落），
λ = 0.6 牺牲的是**能不能产出一个答案**（终止/收敛）。

### 5.3 被帮助的题在 λ 之间稳不稳？

help 集合的 Jaccard（AIME）：

| | 0.1 | 0.2 | 0.3 | 0.4 | 0.6 |
|---|---|---|---|---|---|
| 0.1 | 1.00 | 0.50 | 0.36 | 0.46 | 0.23 |
| 0.2 | 0.50 | 1.00 | 0.21 | 0.40 | 0.29 |
| 0.3 | 0.36 | 0.21 | 1.00 | 0.33 | 0.50 |
| 0.4 | 0.46 | 0.40 | 0.33 | 1.00 | 0.30 |
| 0.6 | 0.23 | 0.29 | 0.50 | 0.30 | 1.00 |

最高也只有 0.50 —— **"被哪些题帮到"随 λ 大幅换人，不是一个稳定的题目子集**。

逐题 flip 画像（H=help，X=harm，`.`=不变；λ 顺序 0.1/0.2/0.3/0.4/0.6），只列有 flip 的题：

```
a24#2  x  H.H.H      a24#24 x  HHHHH      a25#7  OK XXXXX
a24#5  x  .H..H      a24#25 OK ....X      a25#24 OK .XXX.
a24#14 x  HHHH.      a24#28 x  HH.H.      a25#25 x  HH.H.
a24#16 x  .H.HH      a24#29 OK ..XXX      a25#26 x  HHHHH
a24#17 OK ..X.X      a25#2  OK ...XX      a25#29 OK XXX.X
```
（完整 37 行见脚本 `hh.py` 输出。）

- 5 个 λ 上一次都没 flip 的题：**24/60**。
- 至少被帮过一次：**17** 题；至少被害过一次：**19** 题。
- 被 ≥3 个 λ 稳定帮到：**7** 题；被 ≥3 个 λ 稳定害到：**4** 题。
- **逐题挑最优 λ 的 oracle = 48/60 = 80.0%**，而最好的单一 λ（0.2）是 41/60 = 68.3%。
  → **逐题/逐 λ 选择有 +11.7 分的 headroom**，这是本报告唯一的正面可用信号。

---

## 6. 直接回答：help 和 harm 可分离吗？

**按"题目本身的可测属性"分离：不行。**
- MATH-500 上唯一有预测力的题目侧特征是 parent 的输出长度（= 难度代理），但它同时选出 help 和 harm：
  长输出区（≥8000 字符，10% 的题）含 45/68 的 help 与 30/59 的 harm，桶内 help 占比 0.56–0.67 ——
  **只比随机好一点，不构成分离**。
- 被帮助 vs 未被帮助的题，在 parent 侧长度上互相覆盖（λ=0.3：10686 vs 12022；λ=0.4：9057 vs 18368，
  方向还会翻转）。
- AIME 上根本做不了题目侧刻画（无 `resp`，`tok` 是常数 8192），能看到的只有：
  helped/harmed 的题号随 λ 大换血（Jaccard ≤ 0.50）。**"某些题天生该被 AR 帮"这个假设在 AIME 上不成立。**

**按"输出的可测属性"事后分离：部分可以，但只能事后。**
- 规则「merge 输出比 parent 更长 ⇒ 判为 harm」（MATH-500，pooled λ=0.1–0.4）：
  召回 48/59 = 0.81，误判 21/68 的 help，**精度 0.70**。
- 规则「merge 输出 ≥7000 字符 ⇒ harm」：召回 41/59，误判 28/68，精度 0.59（更差）。
- 28/59 的 harm 是"答案解析不出来"，这一半**几乎零成本可检测**（pred 为空），
  可以直接触发回退到 parent；剩下 31/59 是解析出来但答错，**与 help 在任何已测特征上不可区分**。

**综合判断：** 在 DreamReasoner 上，help 与 harm **不能**由任何题目侧属性预先分离；
只能靠"生成完之后看输出长度/是否终止"事后部分分离（精度 ~0.70，且只在 MATH-500 验证过，
而 MATH-500 本身是 null benchmark）。更重要的是 **AIME 上 help 根本不随 λ 持续增长** ——
它在 λ=0.2 饱和后下降，harm 独自爆炸。**所以"保留有用的那部分"这个提法，在 DreamReasoner 上
不是"从一个单调上升的 help 里把 harm 剔掉"，而是"λ=0.2 之后 help 本身就没了"。**
唯一的正面出口是**逐题选 λ**（oracle 80.0% vs 68.3%），但那需要一个预测器，
而本报告的全部证据说明**题目侧特征做不出这个预测器**。

---

## 7. 要补什么 GPU run 才能把洞填上（本报告未提交任何作业）

按价值排序。默认池 H200 @ 0.2 GPU 分数（单 8B bf16 模型 ≈15 GB < 28 GB，够）；job 名加 `-h200`。

1. **重跑 6 个 AIME arm，保存 `resp`（最高优先）。**
   `step7_reasoning_merge/scripts/dreamreasoner_aime.py` 现在会写 `results.partial.jsonl`
   并在每条记录里带 `resp`（`:274`），而现有 `aime/rmg-b0*` 目录是 2026-08-27 跑的、没有这个文件。
   不重跑就**永远无法**在唯一有信号的 benchmark 上做长度 / 退化重复 / 未终止-vs-算错 分解。
   成本：6 arm × 60 题 × ~80–96 s/题 ≈ 8–10 GPU·h。
2. **轨迹保存**：同一批 run 加 `DREAM_SAVE_TRAJ=<dir>`
   （开关定义在 `step2_instruct/scripts/traj_utils.py:7,31`；harness 写 `<dir>/<name>.traj.jsonl`）。
   over-merge 的 DreamReasoner arm **目前一条轨迹都没有**，所以"置信度被压低、每 block 需要更多
   refinement 步"（`DREAMREASONER.md:51-53` 从 s/prob 80→96 s 推断）**只是推断，没有直接测量**。
   有了轨迹才能直接看 conf 阈值 0.95 的达成步数分布。
3. **AIME λ 扫描补点 λ=0.5 以及 MATH-500 补 λ≥0.5**：现在 MATH-500 组最大只到 0.4，
   AIME 组 0.4→0.6 之间是空的，而 help 的崩塌正发生在这一段。
4. **Qwen3-8B donor 的 AIME 逐题标签**：`aime/qwen3-aime-full/results.json` 只有 `avg@64` 汇总，
   没有 `per_q`。要在 AIME 上做论文 Fig.1 的 donor-conditioned adoption 曲线，必须重跑并存逐题。
   （`math500/qwen3-math-full-peritem-h200` 已经有 `per_q`，可照抄其形式。）
5. **多种子 / 多采样**：现有全部 arm 是 greedy 单样本，n=60。AIME 上 help=13、harm=3 这种量级，
   任何结论都受单次解码抖动影响。至少 4 个采样种子的 avg@k 才能把 §3.1 的曲线形状钉死。

---

## 8. 限制（limits）

1. **AIME 无 `resp`**：§5.2 的全部刻画只基于最终 pred 字符串。退化重复率、真实输出长度、
   截断/未终止在 AIME 上**未测量**。`tok=8192` 是预算常数，不是长度。
2. **MATH-500 是 null benchmark**：acc 93.2–94.8、写完acc 95.9–96.5、每个 λ 配对 p≥0.31。
   §5.1 的 help/harm 刻画统计上建立在一组**本身不显著**的 flip 上（每 λ 仅 9–19 个事件，
   pooled 才 68/59）。方向性很一致，但不能当作已验证的机制。
3. **n 小**：AIME 60 题、单次 greedy。λ=0.2 的 McNemar p=0.021，在 5 次对照的 Bonferroni
   (0.05/5=0.01) 下**不存活**；λ=0.6 的 p=0.064 也不存活（但以 λ=0.2 为参照时 p=0.001 存活，
   见 `DREAMREASONER.md:40-49`）。
4. **donor 跨协议**：§4 的 donor 标签来自 Qwen3-8B avg@4、T=0.6、32768 tok，与 diffusion 侧
   greedy/8192 不可直接配对；n=20/11。只能定性。
5. **未核硬件一致性**：`aime/` 组 6 个 run 的 `results.json` 不记录节点类型，
   `s/prob` 80→96 s 的变化**可能**部分来自硬件而非模型（`DREAMREASONER.md:53` 同样标注为"当观察"）。
6. **MATH-500 的 `-m500-8k` 组本身也没有硬件标注**（无 `-b300`/`-h200` 后缀），
   而同名模型的 `-b300`/`-h200` 变体差 2.6 分 —— 组内一致性靠的是"同一批提交"，不是记录。
7. **dup-shingle 判据是我自己选的**（12-gram、重复率阈 0.3），不是项目既有定义；
   项目里的退化重复定义在 `step2_instruct/scripts/failure_analysis.py` 的 `repeat` 失败类，
   本报告**没有**复用它（那个脚本绑定 GSM8K 的 lm-eval 格式）。

## 9. 明确没能确定的事

1. **AIME 上 over-merge 的伤害是"不终止"还是"算错"** —— 无 `resp`，只能从 pred 形态
   （λ=0.6 出现 4 个 `None`）间接推断，λ≤0.4 段无法判定。
2. **"help 随 λ 持续增长"在 AIME 上为何不成立的机制** —— 只知道现象（λ=0.2 后 help 13→4–8→5）。
3. **论文 appendix 里是否真的写了"两者同时增长"** —— `iclr/overleaf/*.tex` 里**找不到**这句话；
   它在 `README_ZH.md:27` 与 `step2_instruct/results/README.md:123`。我**无法确定**提交版 PDF
   的 appendix 是否以别的措辞表达了它（没有解析 PDF 全文）。
4. **DreamReasoner 的 donor-conditioned 曲线（论文 Fig.1 那种）在 AIME 上长什么样** —— 缺 donor 逐题标签。
5. **λ=0.5 的行为** —— 两个 benchmark 都没有这个点，help 崩塌的确切位置未定位。
6. **逐题最优 λ 是否可预测** —— oracle 80.0% 已测，但**没有任何可用特征**能逼近它；
   我测过的 parent 输出长度（MATH-500）和 λ 间 flip 稳定性（AIME）都失败。
7. **over-merge 是否压低了 diffusion 的 confidence**（conf 阈值 0.95 的达成步数）—— 无轨迹，未测。
