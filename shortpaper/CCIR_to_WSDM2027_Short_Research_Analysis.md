# CCIR Cup → WSDM 2027 Short：研究定位、创新边界与执行路线

> **状态：已被新工作文档取代。** 当前内部分析统一见 `shortpaper/analysis.md`，论文问题、方法与实验设计统一见 `shortpaper/draft.md`。本文件仅保留为 2026-08-18 历史来源记录，不再作为后续编辑入口。

> **版本**：2026-08-18  
> **目标**：将 CCIR 2026 LRAT 赛题冠军工作的数据、代码与实验资产，转化为一篇具有独立研究问题的 **WSDM 2027 Short Paper**。  
> **文档性质**：研究决策与执行底稿。文中凡标记为“现有观察”的内容来自目前比赛实验；凡涉及 false negative、censoring、confidence 等机制解释，现阶段均为**待验证假设**，不能在论文中提前写成已证实事实。

## 0. 结论先行

当前最值得推进的路线不是“把 CCIR 冠军方案整理成论文”，也不是“继续给 LRAT 补更多超参数实验”，而是把比赛过程中出现的反常现象升级为一个独立问题：

> **Agent trajectory 中的 unbrowsed feedback 是有用的 negative signal，但其可靠性很可能不是均匀的；agent 的“不 Browse”只是选择性行为观察，并不天然等于确定的 irrelevance。**

因此，最合适的论文主线不是攻击式的 **“Unbrowsed ≠ Irrelevant / LRAT 的假设是错的”**，而是更准确、更一般、也更适合作为 LRAT follow-up 的：

> **When are unbrowsed results reliable negatives?**  
> **How should we calibrate selective behavioral feedback for trajectory-supervised retrieval?**

可以将核心问题形式化为：

$$
B=0 \not\Rightarrow R=0,
$$

其中 $B$ 表示 agent 是否 Browse，$R$ 表示文档是否真正具有 retrieval relevance / downstream utility。更准确地说，我们真正需要研究的是：

$$
P(R=0\mid B=0,x),
$$

其中 $x$ 可以包含 retrieval rank、trajectory stage、重复检索、后续 Browse、已有 evidence、agent 状态等上下文。论文的关键不是证明“存在少量 false negative”——这太弱，也早已是 dense retrieval 中的常识——而是证明 **unbrowsed feedback 的可靠性具有系统性的异质性（heterogeneous reliability）**，并且这种异质性会影响 trajectory-supervised retriever training。

如果数据支持，最小而完整的贡献链应当是：

$$
\boxed{
\text{Empirical anomaly}
\rightarrow
\text{Reliability diagnosis}
\rightarrow
\text{Mechanism}
\rightarrow
\text{Calibrated supervision}
\rightarrow
\text{Robust validation}
}
$$

而不是：

$$
\boxed{
\text{LRAT}
\rightarrow
\text{negative size / LR / reweight sweep}
\rightarrow
\text{best configuration}
}.
$$

这条路线对 WSDM Short 的尺度是合适的：问题窄，但不是琐碎参数；方法可以简单，但需要机制清楚、证据扎实。WSDM 2027 Short 官方明确允许 narrower-scope、preliminary but empirically validated research；正文上限 4 页，deadline 为 **2026-11-17 AoE**，并明确覆盖 foundation-model retrieval/ranking、Web agents、agentic systems 等主题。

---

## 1. 当前项目资产：比赛结果不是论文贡献，但非常值钱

### 1.1 已经拥有的资产

CCIR Cup 给这篇论文提供的最大优势，不是“冠军”这个标签本身，而是已经完成了一轮相当扎实的 empirical exploration，省掉了从零摸索的阶段。

目前已有的核心资产包括：

- 基于 **LRAT Training Pairs / trajectories** 的完整数据处理与训练流程；
- 主模型 **Qwen3-Embedding-0.6B**，并有 **Multilingual-E5-Large-Instruct** 对照；
- positive / negative trajectory pair 的映射与复现经验；
- negative group size 的系统实验；
- reweight on/off 的 matched comparison；
- 多个 learning rate、epoch 的实验；
- 已完成数十组最终评测；
- 比赛 B 榜冠军，说明当前 pipeline 本身具有较好的工程可靠性和竞争力；
- 对 LRAT 原始实现、数据结构、trajectory 中 Search/Browse/reasoning 事件已经有比普通复现者更深的理解。

这些东西应当在论文中被重新定位为：

> **用于发现研究问题的 preliminary evidence、可靠 baseline 与实验基础设施。**

而不是 contribution 本身。

### 1.2 已有实验真正透露出的研究信号

目前最值得追的不是哪一个配置分数最高，而是几个“为什么”：

1. **为什么减少 negative 数量有时反而更好？**
2. **为什么 reasoning-length reweight 的收益并不稳定？**
3. **为什么强 pretrained retriever 只做较轻 domain adaptation 往往优于多 epoch 训练？**

第三点更像 training recipe / domain adaptation 问题，作为主论文主题偏泛；第二点可以发展成 utility-proxy 论文，但实现和论证更困难；第一点与 trajectory supervision 的标签性质直接相关，因此目前最适合 Short。

现有 negative-size 结果只能支持：

> “存在一个反常现象，值得解释。”

它**不能直接支持**：

> “negative 越多 → false negative 越多 → 性能下降。”

后者仍然只是机制假设，必须额外验证。

---

## 2. 为什么目标是 WSDM 2027 Short，而不是 Full / Findings / Workshop

### 2.1 WSDM Short 的制度位置

WSDM 2027 Short 是正式独立 research track，不是 workshop abstract。官方说明包括：

- 面向 search / data mining / Web / agentic systems 等研究；
- 允许 relevant research findings、methodologies、新任务与应用；
- 相比 Full，贡献可以 narrower in scope；
- 接受 preliminary **but empirically validated** ideas；
- 正文最多 **4 pages**，references 与 ethical consideration 不计入正文页数；
- deadline：**2026-11-17 AoE**；
- accepted papers 进入 ACM Digital Library；
- 2027 要求至少一名作者线下参会展示。

官方页面：  
https://www.wsdm-conference.org/2027/cfsp.html

从题目匹配看，这项工作同时落在：

- Web Search；
- Search user behavior / log analysis 的“agent 化”延伸；
- Retrieval, indexing, and ranking with foundation models；
- Web agents / agentic systems；
- Intelligent assistants / task-driven search。

因此 scope 基本没有问题。

### 2.2 为什么不应为了“更硬”硬冲 Full

当前工作最自然的 contribution 是一个**窄而精确的监督建模问题**。如果冲 Full，reviewer 很可能要求：

- 多 benchmark；
- 多 agent；
- 多 retriever family；
- 更完整的理论或因果建模；
- downstream end-to-end agent evaluation；
- 与更大量 agentic retrieval work 比较。

这会把本来漂亮的 Short 问题扩成一个需要重新搭建大规模实验体系的项目。

Short 的合理目标是：

> **清楚识别一个此前被平均化处理的 trajectory signal；给出可靠证据；提出低复杂度修正；证明修正确实优于简单减少 negatives。**

这已经完整。

### 2.3 Findings 与 Workshop

WSDM 2027 Findings 是 Full 投稿后的分流，不是 11 月独立投稿窗口，因此现在的现实路线就是 Short。

Workshop 更适合早期交流，但需要注意 archival policy：若 workshop paper 进入正式 proceedings，可能影响后续同内容投稿的 originality。既然当前已经有明确的 Short deadline 和足够实验基础，没有必要为了“先挂一个 WSDM”优先选择 workshop。

---

## 3. 导师与研究生态：这个题目并不突兀

### 3.1 毛佳昕相关研究线

毛佳昕老师本人长期研究 IR、用户行为、搜索评价与 dense retrieval。与当前项目最相关的不是“他最近有没有一直投 WSDM”，而是他的研究问题传统：

> **如何从行为或 downstream outcome 中提取更可靠的 supervision。**

他在 WSDM 2022 合作论文 *Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval* 获 Best Paper；2024 年 *Scaling Laws for Dense Retrieval* 又获 SIGIR Best Paper。近两年合作线进一步明显转向 Agentic Search / RAG / multi-agent training，例如：

- **MAO-ARAG**：根据 query 难度动态编排不同 RAG 模块；
- **M-ASK**：将 search behavior 与 knowledge management 解耦，并引入 turn-level supervision；
- **OASES**：讨论 agentic search 中 process reward 是否真正与最终 outcome 对齐，并联合训练 evaluator；
- 以及此前的 CoSearchAgent、TourRank、多 Agent RAG 等工作。

这说明“行为信号 / intermediate signal 是否可靠、如何把它转成训练监督”本身就在一个非常自然的研究脉络里。

你的问题可以理解为把传统 human-search 时代的：

$$
\text{click / examination / behavioral bias}
$$

迁移到 agentic-search 时代的：

$$
\text{browse / exposure / reasoning / stopping bias}.
$$

这不是硬蹭 agentic search，而是传统 IR behavior modeling 的自然延伸。

### 3.2 LRAT 作者线

LRAT（Yuqi Zhou, Sunhao Dai, Changle Qu, Liang Pang, Jun Xu, Ji-Rong Wen）提出从 agent trajectories 中提取：

- browsing actions；
- unbrowsed rejections；
- post-browse reasoning traces；

作为 retriever supervision，并用 reasoning length 估计 relevance intensity。

原论文的关键贡献是：

> **证明 agent trajectories 可以成为可扩展、有效的 retriever supervision source。**

你的论文不应挑战这个总体结论，而应研究其中一个更细的 boundary condition：

> **trajectory feedback 有用，并不代表所有 unbrowsed feedback 的 reliability 相同。**

这是最重要的叙事边界。

### 3.3 PolyU 范文琦—李青相关合作线

范文琦老师近年的工作更偏广义 RAG、WebAgents、LLM recommendation、retrieval-augmented systems 与 trustworthy AI，而不是直接做 LRAT 式 retriever supervision。但这条线很值得读，因为它提供了：

1. **RAG / WebAgent 系统视角**：retrieval 不是独立模块，而是 agent pipeline 的一部分；
2. **query-aware / adaptive retrieval**：不同 query 对知识粒度与 retrieval strategy 的需求不同；
3. **trustworthiness / robustness**：retrieval signal 的噪声、攻击面、系统成本等不能只看静态 relevance；
4. **WSDM 论文写法参考**：范文琦相关团队在 WSDM 2026 有正式论文，可观察其问题定义、实验密度与 concise presentation。

对本项目最有用的不是照搬他们的方法，而是学习一种系统性思路：

> **不要假设一个 retrieval signal 在所有 query / stage / context 下同质有效。**

这与“unbrowsed feedback reliability is heterogeneous”的方向是相容的。

---

## 4. 核心研究问题：从 “Unbrowsed ≠ Irrelevant” 升级为 selective / censored behavioral supervision

### 4.1 不要把论文写成一句过强的反驳

最初的：

> **Unbrowsed ≠ Irrelevant**

直观、好懂，但作为论文核心 claim 太粗。

因为 LRAT 并不需要证明所有 unbrowsed document 都绝对 irrelevant，只需要证明这种行为信号**整体上足以构造有用 negatives**。所以仅仅找到若干：

$$
B=0,\quad R=1
$$

的反例，并不能推翻 LRAT，也不足以构成新论文。

真正有价值的命题是：

$$
P(R=0\mid B=0,x_1)\neq P(R=0\mid B=0,x_2).
$$

也就是说，不同 trajectory context 下，unbrowsed feedback 的可靠性不同。

### 4.2 一个更完整的生成视角

可以把 Browse 行为粗略理解为：

$$
B = g(E, R, \tau, A, C),
$$

其中：

- $E$：exposure，agent 是否真正有机会充分看到 / 判断该结果；
- $R$：文档对当前信息需求的 relevance / utility；
- $\tau$：当前 trajectory state，包括已有证据、当前 reasoning、此前搜索；
- $A$：agent policy / model 特性；
- $C$：cost / budget / stopping constraints。

因此观察到：

$$
B=0
$$

可能来自：

1. 文档确实无关；
2. 排位较低，没有被充分注意；
3. agent 已获得足够信息，不再需要继续 Browse；
4. 文档与已有 evidence 冗余；
5. 搜索预算 / token budget 限制；
6. agent policy 决策错误；
7. 当前 query/trajectory state 使文档暂时看起来不重要，但后续状态改变后会变得有用。

所以 unbrowsed feedback 的本质更接近：

> **selective behavioral observation**

而不是完整人工 relevance judgment。

“censored feedback”可以作为理论化术语，但正式使用前应确认数据生成机制是否确实满足相应 censoring 解释；如果无法严谨建模，论文中用 **selective / heterogeneous behavioral feedback** 会更稳妥。

---

## 5. 建议的研究假设

论文不要一开始就假定机制成立。建议明确拆成几个可证伪假设。

### H1：Unbrowsed negatives 中存在非忽略的潜在正例 / 高 utility 文档

$$
P(R=1\mid B=0)>0.
$$

H1 本身只证明 label noise 存在，贡献不够。

### H2：这种噪声不是均匀随机噪声，而是随 trajectory context 系统变化

例如：

$$
P(R=1\mid B=0,\text{rank}\le k)
\neq
P(R=1\mid B=0,\text{rank}>k),
$$

或者随：

- repeated retrieval；
- later browse；
- trajectory stage；
- evidence saturation；
- agent type；
- query difficulty；

呈现显著变化。

**H2 是整篇论文从“琐碎 false negative”升级成“trajectory supervision 问题”的关键。**

### H3：negative 数量增加时，低置信度 negatives 被引入的概率上升

设 $N$ 为 sampled negatives：

$$
|N|\uparrow
\Rightarrow
P(\exists d\in N:\text{low-confidence negative})\uparrow.
$$

注意这里不要预先把 low-confidence 等同于 true false-negative；先用 proxy / judge / later behavior 验证。

### H4：negative-size 的性能异常与 contamination / confidence 变化相关

真正需要连接的是：

$$
\text{negative-set reliability}
\leftrightarrow
\text{retrieval performance},
$$

而不只是：

$$
|N|\leftrightarrow\text{performance}.
$$

### H5：基于 trajectory evidence 的 selective calibration 优于随机少取 negatives

这是决定 method 是否成立的关键：

$$
\text{Calibrated negatives}
>
\text{Randomly fewer negatives}.
$$

如果做不到这一点，reviewer 很容易认为“你的全部收益只是因为少训练几个 negatives”。

---

## 6. 最小方法：Confidence-Calibrated Trajectory Negatives

### 6.1 Vanilla LRAT-style objective

设 query 为 $q$，positive 为 $d^+$，unbrowsed negatives 为 $\{d_i^-\}$，普通 InfoNCE 可写为：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\log
\frac{\exp s(q,d^+)}
{\exp s(q,d^+)+\sum_i\exp s(q,d_i^-)}.
$$

隐含处理是每个 sampled negative 的负标签置信度相同。

### 6.2 Confidence weighting

定义：

$$
c_i=c(q,d_i,\tau)\in[0,1],
$$

其中 $c_i$ 表示：

> 在当前 trajectory evidence 下，将 $d_i$ 当作 genuine negative 的置信度。

最简单的修改：

$$
\mathcal{L}_{\mathrm{cal}}
=
-\log
\frac{\exp s(q,d^+)}
{\exp s(q,d^+)+\sum_i c_i\exp s(q,d_i)}.
$$

高置信 negative：

$$
c_i\approx1,
$$

疑似被选择性漏掉的 candidate：

$$
c_i\ll1.
$$

Short Paper 不需要把 $c$ 做成复杂神经网络。真正重要的是：

1. $c$ 有清楚的行为含义；
2. $c$ 与真实 / proxy reliability 有实证关系；
3. 使用 $c$ 后优于 vanilla LRAT；
4. 优于 random downsampling；
5. 最好在两个 backbone 或多个 settings 上稳定。

### 6.3 $c$ 的候选构造

建议按“最少引入外部假设”的顺序尝试。

#### A. Later-browse / repeated-retrieval confidence

若文档在早期 Search 中 unbrowsed，但后续：

- 再次被检索；
- 最终被 Browse；
- 或进入后续 reasoning/evidence；

则早期的 $B=0$ 显然应降低 negative confidence。

这是最漂亮的 trajectory-internal signal，因为不依赖额外 judge。

#### B. Exposure-aware confidence

rank 越靠前、snippet exposure 越充分、agent 仍拒绝 Browse，理论上 negative confidence 可以更高；rank 很低、甚至 agent 很可能没有充分处理的结果，confidence 应降低。

但注意：不能直接假设“rank 越高越可靠”，必须先实证，因为 agent 也可能存在 position bias。

#### C. Cross-trajectory / cross-agent disagreement

若同一或高度相似的 $(q,d)$ 在多个 trajectory / agent 中出现：

$$
B^{(1)}=0,\quad B^{(2)}=1,
$$

则说明 browse label 存在 policy dependence。

如果数据足够，这是证明“behavior ≠ ground-truth relevance”的强证据。

#### D. Independent relevance / utility judge

用独立 LLM 或 stronger reranker 判断 unbrowsed candidate 的 relevance / answer utility。

优点：覆盖率高。  
缺点：容易把论文变成“用另一个模型造标签”，且有 judge bias。

因此更适合做验证 proxy，而不是唯一 supervision source。

#### E. Answer-evidence matching

检查 unbrowsed document 是否包含最终答案所需 evidence。

这是 downstream utility proxy，但要小心：

- answer presence ≠ 当前 search step 的必要 relevance；
- lexical overlap 可能导致假阳性；
- final answer 可能是其他文档共同支持。

适合作为辅助证据。

---

## 7. 实验设计：先诊断，再做方法

### 7.1 Stage 0：数据可行性审计

这是最先做、成本最低、但决定论文生死的一步。

需要回答：

1. 每个 Search step 能否准确恢复所有 returned candidates、rank、snippet？
2. Browse event 能否稳定映射回 candidate document？
3. 同一 document 是否会跨 search step 重复出现？
4. “先 unbrowsed，后 browsed”的事件量是否足够？
5. 是否存在同一 / 相似 query-document 在不同 trajectory 中的行为差异？
6. 能否定位 Browse 前后的 reasoning segment？
7. 是否能恢复 trajectory stage、search depth、remaining budget 等变量？
8. Corpus document ID 是否稳定，能否避免 URL / text normalization 造成假重复？

如果 later-browse / repeated-document 数量极少，这条路线仍可做，但必须更多依赖 rank/exposure、judge 或 cross-agent proxy，研究故事会弱一些。

### 7.2 Stage 1：证明 reliability heterogeneity

第一张真正有研究价值的图，不应该是新模型分数，而应是类似：

$$
P(\text{proxy-positive}\mid B=0,x)
$$

随 $x$ 的变化。

可以画：

- rank bucket → suspected false-negative rate；
- trajectory stage → suspected false-negative rate；
- repeated retrieval count → later browse probability；
- negative-set size → low-confidence negative proportion；
- agent / query difficulty → browse disagreement。

只要能稳定看到明显结构，就说明问题不是“随机 label noise”。

### 7.3 Stage 2：解释比赛中的 negative-size anomaly

把已有 group size 4/5/6/7 实验重新分析。

不要只呈现：

| negatives | MRR |
|---|---:|
| 3 | ... |
| 4 | ... |
| 5 | ... |
| 6 | ... |

而要同时加入：

| negatives | MRR | suspected FN rate | avg negative confidence |
|---|---:|---:|---:|
| ... | ... | ... | ... |

目标是展示：

$$
\text{more negatives}
\rightarrow
\text{lower average reliability}
\rightarrow
\text{worse training},
$$

而不是仅仅“4 是最佳超参数”。

### 7.4 Stage 3：方法对照

至少需要：

1. **Base retriever / no trajectory FT**
2. **Vanilla LRAT**
3. **LRAT + randomly fewer negatives**
4. **LRAT + simple rank/exposure filtering**
5. **LRAT + proposed confidence calibration**
6. 如果成本允许：**false-negative-aware dense retrieval baseline**（例如 SimANS 思路或 noise-robust loss 的适配）

其中第 3 项必须有，否则 reviewer 可以直接把你的方法解释成“少放点 negative”。

### 7.5 Stage 4：robustness

Short 不要求巨大规模，但至少要避免只在一个随机 seed、一个 backbone、一个配置上成立。

优先级：

1. Qwen3-Embedding-0.6B；
2. E5 作为不同 architecture / pretraining family 对照；
3. 多个 negative size；
4. 至少 2–3 seeds（如果算力允许）；
5. in-domain retrieval metric；
6. 若能运行 agent，补一个 end-to-end agent metric 会非常加分，但不是绝对必要。

### 7.6 Metrics

检索侧可保留比赛已有指标：

- MRR；
- Recall@1 / Recall@k；
- 必要时 NDCG / evidence recall。

论文新增的诊断 metric 可包括：

$$
\mathrm{ProxyFNR}
=
P(\widehat R=1\mid B=0),
$$

以及：

$$
\mathrm{Reliability}(x)
=
P(\widehat R=0\mid B=0,x).
$$

如果使用多个 proxy，不建议强行合成一个“ground truth FNR”；应该分别报告，强调它们是不同角度的证据。

---

## 8. Reasoning-length reweight：保留，但降级为 secondary analysis

LRAT 的另一个重要设计是：

$$
\text{reasoning length}\approx\text{relevance intensity / utility}.
$$

你们已有 matched experiments 显示 reweight on/off 的收益并不稳定。这非常值得保留，但不适合与 negative reliability 同时做两个并列 main contribution，否则 4 页会散。

更好的用法有两种。

### 8.1 作为 supporting evidence

论文主张：

> trajectory behavior 是有价值但有噪声 / heterogeneous 的 supervision。

那么：

- unbrowsed negative reliability 不均匀；
- reasoning length 的 utility proxy 也不总稳定；

可以共同作为：

> **trajectory-derived behavioral signals should be calibrated rather than treated as noiseless labels**

的辅助证据。

但正文最多用一个小表 / 一两句，不展开新方法。

### 8.2 作为失败后的 B 计划

如果 negative reliability 的数据审计显示：

- later-browse 太少；
- cross-agent evidence 不足；
- confidence method 不能稳定优于 random downsampling；

则可以重新评估：

> **Reasoning length is a confounded proxy for document utility.**

再研究：

$$
L_{\mathrm{reasoning}}
=
U_{\mathrm{doc}}
+
D_{\mathrm{question}}
+
V_{\mathrm{agent}}
+
N_{\mathrm{irrelevant}}
+\epsilon.
$$

可能的方法是从 length-based utility 转向 information-gain / outcome-aligned utility。但这条路线已经与 OASES、Agentic-R 等“outcome-aligned supervision”工作更接近，novelty boundary 更拥挤，实现成本也更高，所以仍然作为 B 计划。

---

## 9. 与 LRAT 的关系：如何避免“拿 WSDM Short 砸自家 SIGIR Full 的脚”

这个问题既是组内叙事问题，也是论文科学表述问题。

### 9.1 不应这样写

避免：

> LRAT assumes unbrowsed documents are irrelevant, which is incorrect.

避免：

> LRAT introduces severe label noise.

避免：

> We challenge the key assumption of LRAT.

除非你最终真的有跨 agent / benchmark 的极强证据，否则这些 claim 都太大。

### 9.2 推荐这样写

更好的逻辑是：

> LRAT demonstrates that unbrowsed results provide effective trajectory-aware negative supervision. We further investigate whether the reliability of such feedback is homogeneous across trajectory contexts.

然后：

> We find that the reliability of unbrowsed feedback varies systematically with trajectory evidence, motivating confidence-calibrated negative learning.

逻辑关系是：

$$
\text{LRAT establishes usefulness}
\rightarrow
\text{we characterize reliability}
\rightarrow
\text{we improve calibration}.
$$

这是一篇很标准的 follow-up，而不是 rebuttal。

### 9.3 学术上真正精确的 claim

最值得争取证明的不是：

$$
B=0\not\Rightarrow R=0,
$$

而是：

$$
B=0\text{ is informative but heterogeneously reliable}.
$$

这既承认 LRAT 的价值，又给你留下独立 contribution。

---

## 10. 这个 focus 会不会“太琐碎”？

### 10.1 会显得琐碎的版本

如果最终只有：

1. 找到几个 unbrowsed-but-useful case；
2. group size=4 比 6 好；
3. 给疑似 false negative 降权；
4. MRR 提升一点；

那么 reviewer 很可能总结成：

> LRAT negative sampling contains label noise; authors add a heuristic.

这个版本确实偏弱。

### 10.2 足够成为 WSDM Short 的版本

至少应完成下面三个层次：

#### 层次 A：现象不是个例

证明 unbrowsed reliability 随 trajectory context 有**系统性结构**。

#### 层次 B：现象能解释已有训练行为

证明 negative-set reliability 与 negative-size anomaly 有对应关系。

#### 层次 C：针对性修正优于 trivial baseline

证明：

$$
\text{confidence calibration}
>
\text{random downsampling / simple smaller group size}.
$$

做到 A+B+C，就不再是“LRAT 一个小 bug”，而是：

> **一个新的 trajectory supervision reliability 问题 + 实证 characterization + 简洁解决方案。**

这正好符合 Short 的尺度。

### 10.3 不要再扩大一级

也不建议直接把论文扩成：

> “统一建模所有 agent trajectory feedback 的 causal utility framework”。

那已经进入 Full-paper 体量，而且会撞上 Agentic-R、OASES、各种 process reward / outcome alignment 工作。

当前最佳尺度就是：

> **只研究 unbrowsed negative supervision，但把这个窄问题讲得一般、准确、可验证。**

---

## 11. Reviewer 最可能攻击的点与预防方式

### Attack 1：“False negatives 在 dense retrieval 里早就有人研究了。”

正确。

你的 novelty 不能是：

> “negative 里可能有 false negative。”

而应是：

> **Agent trajectory 中的 negative labels 来自 agent behavior；这种行为监督具有 trajectory-dependent selection mechanism。**

需要把经典 false-negative retrieval work（SimANS、noise-robust dense retrieval）作为基础，而不是竞争对手假装不存在。

### Attack 2：“这不就是 click / position bias？”

思想上确实有亲缘关系，而且应该主动引用 Unbiased Learning-to-Rank。

区别在于 agent Browse decision 受到的变量更复杂：

- reasoning state；
- already-acquired evidence；
- search budget；
- stopping；
- redundancy；
- tool policy；
- agent-specific behavior。

因此可以写：

> We view agent browsing as a new form of implicit behavioral feedback, analogous in spirit to clicks but generated under a multi-step reasoning policy.

不要声称“以前从未有人研究 selection bias”。

### Attack 3：“你的 proxy 不是真 relevance。”

这是最强攻击之一。

解决方式：**多 proxy triangulation**。

例如：

- later browse；
- repeated retrieval；
- cross-agent disagreement；
- independent judge；
- answer evidence。

如果多个独立 proxy 指向同一趋势，就比单一 LLM judge 强很多。

### Attack 4：“你只是少用了 negatives。”

必须有：

$$
\text{ours vs random fewer negatives}.
$$

最好在相同 expected negative mass / 相同 negative count 下比较。

### Attack 5：“只有一个 dataset / competition setting，泛化不足。”

Short 可以接受较窄 scope，但至少做：

- 两个 retriever backbone；
- 多组 negative sizes；
- 若可能，不同 agent subset / query subset；
- 清楚承认 scope limitation。

如果能找到 LRAT 公共数据或直接使用其公开 benchmark 做额外小规模验证，会显著增强论文。

### Attack 6：“为什么不直接用 stronger reranker / LLM judge 清洗？”

回答应该是：

> 我们关注的是 trajectory-internal supervision calibration，希望不依赖昂贵 external annotation / judge。

如果最后方法确实依赖 LLM judge，那需要对成本与 judge bias 做说明。

### Attack 7：“与 LRAT 的 novelty 太近。”

论文要把 LRAT 定位为 foundation：

- LRAT：agent trajectory **can provide supervision**；
- ours：trajectory-derived unbrowsed supervision **should be reliability-calibrated**。

一句话差异必须在 Introduction 第一页就说清楚。

---

## 12. 4 页 Short Paper 的推荐结构

4 页极其有限，不应该写成缩小版 Full。

### Page 1：Problem + Motivation

约 0.8–1.0 页：

- agentic retrieval 背景，极短；
- LRAT 建立 trajectory-supervised retrieval；
- competition / reproduction 中发现 negative-size anomaly（匿名叙述，不写比赛身份）；
- 核心问题：unbrowsed feedback reliability 是否 homogeneous？
- 贡献 2–3 条。

建议 Figure 1：

> 一个 trajectory：某 document 在 $t$ 时刻被检索但未 Browse，后续再次出现并被 Browse；展示 “non-browse ≠ certain negative” 的直觉。

### Page 2 上半：Diagnosis

约 0.6–0.8 页：

- 定义 $B,R,c$；
- reliability proxies；
- 一张关键统计图证明 heterogeneity。

### Page 2 下半：Method

约 0.6 页：

- vanilla InfoNCE；
- confidence-calibrated loss；
- $c$ 的构造；
- 不要铺复杂 architecture。

### Page 3–4：Experiments

实验应占正文最多空间。

核心表：

| Method | Backbone A | Backbone B | ... |
|---|---:|---:|---:|
| Base | | | |
| LRAT | | | |
| LRAT fewer random neg | | | |
| heuristic filter | | | |
| Ours | | | |

再加一个小型 diagnostic table / plot：

- negative count；
- estimated contamination；
- performance。

Related Work 不必单独占大节，可以压缩到 Introduction / Method 周边两段。

Conclusion 3–5 句即可。

---

## 13. 执行时间线：2026-08-18 → 2026-11-17

### 8 月下旬：只做数据审计，不急着训练

目标：

- 写 trajectory analysis script；
- 统计 repeated retrieval / later browse；
- 建立 candidate-level dataframe；
- 定义最少 2 个 reliability proxy；
- 看 H2 是否成立。

**这一阶段决定是否继续主路线。**

### 9 月：机制验证 + 最小方法

完成：

- rank/stage/repeated-retrieval 等 reliability curves；
- group-size anomaly 与 contamination 的对应；
- confidence weighting v0；
- random fewer negatives baseline。

9 月底应该能回答：

> 这是不是一篇论文，而不是一个直觉？

### 10 月：稳定性与补实验

完成：

- Qwen3 + E5；
- seeds；
- 关键 ablation；
- judge / answer evidence 辅助验证；
- 若有条件，做一次 end-to-end agent evaluation。

同时开始写初稿，不要等实验全部结束。

### 11 月 1–10 日：压缩成 4 页

重点不是继续加实验，而是：

- 删除比赛工程细节；
- 保留唯一主线；
- 图表压缩；
- related work 去重；
- claim 与 evidence 一一对应。

### 11 月 11–17 日：投稿检查

检查：

- anonymization；
- ACM 格式；
- 4-page limit；
- originality；
- citations；
- ethical consideration；
- supplementary anonymity；
- 所有数字能从脚本复现。

---

## 14. Go / Pivot / Stop 判断

### Green：继续主线

如果满足大部分：

- 至少两个独立 proxy 显示 reliability heterogeneity；
- heterogeneity 与 negative size / performance 有对应关系；
- proposed calibration 稳定优于 random fewer negatives；
- 至少跨两个 backbone 或多个 settings 成立；

则主线很适合 WSDM Short。

### Yellow：继续但弱化 claim

如果：

- 能证明 heterogeneity；
- method gain 较小或只在部分配置成立；

则可以把论文定位成：

> **empirical characterization + lightweight calibration**

Short 仍可能成立，不要硬吹 universal gain。

### Red：及时 pivot

如果：

- later-browse / disagreement 几乎不存在；
- proxy 之间互相冲突；
- contamination 与 group size 无明显关系；
- confidence method 不优于 random downsampling；

则不要为了维护最初故事继续堆 heuristic。

此时优先回到 B 计划：

> reasoning-length utility calibration / outcome-aligned utility；

或者把工作转成更纯粹的 empirical analysis，再重新评估 venue。

---

## 15. 推荐标题与一句话定位

### 最推荐

**Calibrating Unbrowsed Feedback for Trajectory-Supervised Retrieval**

优点：中性、方法导向、不会显得在攻击 LRAT。

一句话：

> Agent trajectories provide scalable retrieval supervision, but unbrowsed feedback is not equally reliable across trajectory contexts; we characterize this heterogeneity and calibrate negative learning accordingly.

### 更强调研究问题

**When Are Unbrowsed Results Reliable Negatives in Agentic Search?**

优点：问题清楚，非常适合 Short。  
缺点：如果 method contribution 强，标题略偏 analysis paper。

### 更一般但风险更大

**Learning from Selective Behavioral Feedback in Agentic Retrieval**

优点：理论感更强。  
缺点：标题覆盖范围比实际方法大，若只研究 unbrowsed negatives 容易 overclaim。

目前建议优先使用第一或第二种思路。

---

## 16. 论文 / Reference 阅读清单

下面不是“相关领域所有论文”，而是对当前 WSDM Short 最有价值的阅读顺序。优先级按与本项目的直接关系排序，并对人大毛佳昕相关合作线、LRAT 作者线、PolyU 范文琦相关合作线适度加权。

### S 级：必须精读，直接决定 novelty boundary

#### 1. Learning to Retrieve from Agent Trajectories — LRAT
**Yuqi Zhou, Sunhao Dai, Changle Qu, Liang Pang, Jun Xu, Ji-Rong Wen. 2026.**

https://arxiv.org/abs/2604.04949

**为什么必须读**：你的数据、baseline、问题都从这里长出来。需要逐段确认作者如何定义 browsed positives、unbrowsed rejections、reasoning-based relevance intensity，以及他们对“reliable negatives”的实际证据到底有多强。论文不能攻击一个作者实际上没有声称的命题。

**重点读**：trajectory statistics、negative construction、reweight、position bias / reliability analysis、跨 agent 泛化实验。

---

#### 2. Agentic-R: Learning to Retrieve for Agentic Search
**Wenhan Liu, Xinyu Ma, Yutao Zhu, Yuchen Li, Daiting Shi, Dawei Yin, Zhicheng Dou. 2026.**

https://arxiv.org/abs/2601.11888

**为什么必须读**：它直接重新定义 agentic search 中 passage utility，用 local relevance + global answer correctness，而不是只看静态相似度。你的“behavior signal ≠ true utility”必须与它区分。

**与你的差异**：

- Agentic-R：如何构造更 outcome-aware 的 positive / utility signal；
- 你的方向：trajectory-derived **negative behavioral feedback 的 reliability calibration**。

---

#### 3. AgentIR: Reasoning-Aware Retrieval for Deep Research Agents
**Zijian Chen, Xueguang Ma, Shengyao Zhuang, Jimmy Lin, Akari Asai, Victor Zhong. 2026.**

https://arxiv.org/abs/2603.04384

**为什么必须读**：它说明 agent reasoning trace 本身包含传统 query 丢失的信息，并将 reasoning + query 联合编码。它限制了你把“加入 reasoning context”包装成创新的空间。

**与你的差异**：

- AgentIR：改 query representation；
- 你：改 trajectory-derived negative supervision。

---

#### 4. OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search
**Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao. 2026.**

https://arxiv.org/abs/2604.03675

**毛佳昕相关合作线，强烈建议精读。**

**为什么重要**：OASES 的核心问题也是“intermediate proxy supervision 是否真正与 final outcome 对齐”。它能帮助你学习怎样把一个看似局部的 supervision-quality 问题提升成清楚的 research question。

**与你的关系**：

- OASES：search policy 的 process rewards；
- 你：retriever 的 behavioral negative labels。

不要复制 outcome-evaluator 体系，但可以借鉴其“proxy may be misaligned”的论证方式。

---

#### 5. SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval
**Kun Zhou et al. 2022. EMNLP Industry.**

https://arxiv.org/abs/2210.11773  
https://aclanthology.org/2022.emnlp-industry.56/

**为什么必须读**：false negative / hard negative 并不是你的新发现。SimANS 非常直接地讨论“太 hard 的 negative 可能是假负例”。你的论文必须明确说明：新意不是 false negative 本身，而是 **agent behavioral selection mechanism**。

---

#### 6. Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization
**2024.**

https://arxiv.org/abs/2401.00165

**为什么必须读**：与你 proposed confidence weighting 的表面形式很接近。必须确认你的方法是否会被 reviewer 认为只是把已有 false-negative robust loss 搬到 LRAT。

真正应强调的贡献顺序应该是：

> trajectory-specific diagnosis → reliability estimation → simple calibration，

而不是“我们发明了 weighted InfoNCE”。

---

#### 7. Unbiased Learning-to-Rank with Biased Feedback
**Thorsten Joachims, Adith Swaminathan, Tobias Schnabel. WSDM 2017.**

https://dl.acm.org/doi/10.1145/3018661.3018699

**为什么必须读**：这是“行为观察不是 relevance ground truth”的经典理论背景。你的 browse / non-browse 可以与 click / non-click 做概念对照。

不要把 agent behavioral bias 写成凭空出现的新问题；更好的 framing 是：

> agentic search creates a new behavioral-feedback regime with different selection mechanisms.

---

### A 级：人大毛佳昕相关合作线

#### 8. Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search — M-ASK
**Yiqun Chen et al., Jiaxin Mao. 2026.**

https://arxiv.org/abs/2601.04703

**价值**：理解毛佳昕相关合作线当前如何定义 agentic search 的结构性问题、turn-level supervision、search noise 与 credit assignment。对写 Introduction 很有帮助。

---

#### 9. MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation
**Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, Jiaxin Mao. 2025.**

https://arxiv.org/abs/2508.01005

**价值**：query complexity 不同 → RAG workflow 不应一刀切。与你“feedback reliability is heterogeneous”不是同一问题，但方法论上都反对 homogeneous treatment。

---

#### 10. Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning
**Yiqun Chen et al., Jiaxin Mao. 2025.**

https://arxiv.org/abs/2501.15228

**价值**：补齐毛佳昕相关合作线从 RAG 到 multi-agent / agentic search 的演化。重点关注 reward 如何从 downstream answer 反传到 retrieval/reformulation 行为。

---

#### 11. CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models
**Peiyuan Gong, Jiamian Li, Jiaxin Mao. SIGIR 2024 Short.**

建议从毛佳昕个人主页 / ACM DL 获取正式版本：  
https://sites.google.com/site/maojiaxin/

**价值**：它本身不是 retriever-supervision 论文，但非常适合作为“毛佳昕组的 Short Paper 写法”参考：问题聚焦、方法不庞大、实验围绕一个明确机制展开。对你控制 WSDM Short 体量有现实参考价值。

---

#### 12. Scaling Laws for Dense Retrieval
**Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, Yiqun Liu. SIGIR 2024 Best Paper.**

毛佳昕主页：  
https://sites.google.com/site/maojiaxin/

SIGIR Best Paper list：  
https://sigir.org/awards/best-paper-awards/

**价值**：与当前 agent trajectory 主题不直接，但非常值得学习“empirical phenomenon 如何被提升成 generalizable research question”，而不是只做参数表。

---

### A 级：PolyU 范文琦—李青相关合作线

#### 13. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models
**Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, Qing Li. KDD 2024.**

https://arxiv.org/abs/2405.06211

**价值**：建立 RAG 训练策略与系统设计的全景背景。你后续写 related work 时，需要知道 trajectory-supervised retrieval 与传统 RAG retriever training 的边界。

---

#### 14. A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models
**Liangbo Ning et al., Wenqi Fan, Qing Li. KDD 2025.**

https://arxiv.org/abs/2503.23350

**价值**：补 WebAgent 的 tool / action / observation 视角。你的 Browse 本质上是一个 agent action，不能只用静态 IR 的语言描述。

---

#### 15. Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation
**Shijie Wang, Wenqi Fan, Yue Feng, Xinyu Ma, Shuaiqiang Wang, Dawei Yin. ACL 2025.**

https://arxiv.org/abs/2501.02226

**价值**：观察范文琦相关团队如何处理“retrieved knowledge 质量 / 结构”和 downstream LLM utility 的关系。与你的 agentic retrieval 不完全相同，但对 retrieval utility 的叙事有参考价值。

---

#### 16. Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs
**Jiani Huang, Shijie Wang, Liangbo Ning, Wenqi Fan, Shuaiqiang Wang, Dawei Yin, Qing Li. WSDM 2026.**

https://arxiv.org/abs/2503.09382

范文琦 publication list：  
https://wenqifan03.github.io/publications.html

**价值**：一方面是最新 WSDM 正式论文的写法参考；另一方面体现“LLM assistant 的行为 / personalization / evaluation”怎样被包装成 Web/IR 社区问题。

---

#### 17. Inference Cost Attacks for Retrieval-Augmented Large Language Models
**Chengliang Liu, Liangbo Ning, Yujuan Ding, Wenqi Fan. WWW 2026.**

可从范文琦主页进入：  
https://wenqifan03.github.io/publications.html

**价值**：并非核心 related work，但对你理解 agent/RAG 系统中“retrieval decision 有下游成本，不能只看 relevance”有帮助。

---

### A-/B+：当前 Agentic Retrieval 前沿，建议快速扫

#### 18. Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems
**Yilun Zhao et al. 2026.**

https://arxiv.org/abs/2605.04018

**价值**：强调 static retrieval quality 与 agentic utility 并不总一致，并提出 agent-in-the-loop evaluation。非常接近你需要建立的“不要把静态 label 当最终 utility”的认识。

---

#### 19. A Picture of Agentic Search
**2026.**

https://arxiv.org/abs/2602.17518

**价值**：提供 reasoning-induced queries、retrieved documents、thoughts、不同 agents / retrievers 的 trajectory 数据视角。特别值得看它如何组织 agentic search logs，以及是否能为你的 cross-agent disagreement 实验提供思路。

---

#### 20. Towards Retrieving Interaction Spaces for Agentic Search
**Shengyao Zhuang et al. 2026.**

https://arxiv.org/abs/2606.06880

**价值**：将 retrieval 从“返回 top-k 文档”重新理解为给 agent 构造 interaction space。它能帮助你避免把 agentic retrieval 完全按传统一次性 ranking 来理解。

---

#### 21. Rethinking Agentic Search with PI-SERINI: Is Lexical Retrieval Sufficient?
**Tz-Huan Hsu, Jheng-Hong Yang, Jimmy Lin. 2026.**

https://arxiv.org/abs/2605.10848

**价值**：提醒你：agentic loop 中 stronger reasoning 可能改变 retriever 的相对价值。若未来扩展实验，BM25 / lexical baseline 可能比想象中更值得保留。

---

### B 级：Negative Sampling / Label Noise 方法储备

#### 22. Negative Sampling Techniques in Information Retrieval: A Survey
**Laurin Wischounig, Abdelrahman Abdallah, Adam Jatowt. 2026.**

https://arxiv.org/abs/2603.18005

**价值**：快速梳理 random、hard negative、dynamic mining、false-negative mitigation、LLM-generated negatives。写 related work 前先用它建立 taxonomy，再回到原论文。

---

#### 23. TriSampler: A Better Negative Sampling Principle for Dense Retrieval
**Zhen Yang, Zhou Shao, Yuxiao Dong, Jie Tang. 2024.**

https://arxiv.org/abs/2402.11855

**价值**：理解“什么样的 negative 才真正 informative”这一更一般的问题。可作为方法 baseline / discussion 参考。

---

#### 24. Noisy Pair Corrector for Dense Retrieval
**EMNLP Findings 2023.**

https://aclanthology.org/2023.findings-emnlp.765/

**价值**：如果最终发现 trajectory pair 本身存在较明显 label noise，这篇可以帮助你定位“pair correction”类相关工作。

---

## 17. 最终判断

**建议继续做，而且目前 focus 不算太琐碎；真正的风险不是 focus 太窄，而是把它写成一个普通 false-negative heuristic。**

最合理的 intellectual contribution 是：

> **LRAT 证明了 agent trajectory 是有价值的 retrieval supervision source；我们进一步发现 trajectory-derived unbrowsed feedback 的可靠性依赖其行为上下文，因此提出 lightweight confidence calibration。**

如果最终能够证明：

$$
\text{reliability heterogeneity}
+
\text{negative-size mechanism}
+
\text{targeted calibration}>\text{random downsampling},
$$

这已经是一篇结构非常完整的 WSDM Short。

反过来，如果只能证明：

$$
\text{“有少量 unbrowsed document 其实相关”},
$$

则价值不够，应尽早 pivot，而不是继续为一个弱故事追加实验。

当前最优策略不是扩大题目，而是把这个窄问题做到**定义精确、证据多源、baseline 强、claim 克制**。这也最符合你现有 CCIR 资产、4 页篇幅、11 月时间窗口以及人大 IR / Agentic Search 的研究生态。
