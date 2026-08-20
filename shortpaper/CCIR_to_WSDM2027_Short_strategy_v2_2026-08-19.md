# CCIR Cup → WSDM 2027 Short：方向判断、竞争风险与投稿策略

> **状态：已被新工作文档取代。** 当前内部分析统一见 `shortpaper/analysis.md`，论文问题、方法与实验设计统一见 `shortpaper/draft.md`。本文件仅保留为 2026-08-19 历史来源记录，不再作为后续编辑入口。

> 更新时间：2026-08-19  
> 目的：整理本次讨论中与 WSDM Short、CCIR Cup 转论文、Rank2/DND 竞争风险及最终研究尺度有关的结论。本文不是 research proposal，而是当前阶段的判断底稿；后续应以导师/师兄反馈、数据可得性和首轮实验结果为准更新。

## 1. WSDM Short 到底是什么，值不值得作为目标

WSDM 2026 第一次设置独立 Short Paper Track。它不是 Full 拒稿后的安慰轨，也不是 Workshop，而是单独投稿、单独审稿的正式 research track：正文 4 页，至少由 3 名 PC 和 1 名 Senior PC 审稿，录用后进入 ACM Digital Library，并需要在主会展示。官方允许的工作形态包括 scope 较窄但完整的研究，以及 preliminary but empirically validated ideas。

因此更准确的层级是：

$$
\text{WSDM Main Full} > \text{WSDM Short} \gg \text{普通 Workshop paper}.
$$

这里的“$>$”主要指 contribution scope、完整工作预期和圈内认可度。专业人士看到 `WSDM Short` 会明确知道它不是 Main Full，但也会把它视作 WSDM 正式同行评审成果。对已经有多篇 SIGIR/WSDM Full 的成熟博士或 PI，Short 通常只是 portfolio 里的小项目；对本科高年级或早期硕士，一篇自己深度参与完成的 WSDM Short 已经是相当正常且可展示的研究成果。

WSDM 2026 共录用 47 篇 Short，但官方没有公开 Short 投稿总数，因此无法可靠计算录用率。不能因为 reviewer 会看到不少“课程作业式、AI 拼稿式”提交，就推断 Short 很水。更合理的投稿池结构是：大量明显弱稿首先被筛掉，真正决定录用的是“问题小但完整、证据充分”的那一批。

首届 accepted papers 很能说明 Short 的审美。IR × LLM/Agent 方向已经出现 LLM relevance judgment bias、reproducible relevance evaluation、agent-mediated query expansion、multi-hop retrieval 等工作。这些论文共同特点并不是方法复杂，而是 claim 很聚焦，验证足够硬。Short 的“小”主要体现在 scope，而不是 rigor。

2027 又新增 Findings Track。Main Full 仍是 9 页；部分 Main 投稿如果方法扎实但 novelty/selectivity 不够，可以被转入 Findings；Short 仍是独立 4 页投稿。因此现在最稳妥的理解是：

$$
\text{Main Full} \gg \{\text{Findings},\text{Short}\},
$$

而 Findings 与 Short 之间尚没有足够历史形成稳定 prestige 排序。

CCF 口径也要区分。WSDM 本身是 CCF-B 推荐会议，但国内高校内部经常把 Full、Short、Demo、Workshop 分开认定。因此最稳妥的表述是“WSDM Short Paper（WSDM 为 CCF-B 推荐会议）”，不能直接把它等同于 WSDM Main Full。

## 2. CCIR Cup 工作为什么不能直接整理成论文

比赛本身已经给了很好的研究基础：我们有 LRAT training pairs、trajectory/corpus 对齐经验、Qwen3-Embedding-0.6B 训练 pipeline，以及 negative size、learning rate、epoch、reweight、多 seed、paired bootstrap 等一批现成实验。尤其是 reweight 并没有显示稳定独立收益，这已经暴露出一个值得研究的异常：Agent 行为确实包含监督信号，但不同形式的行为监督未必同样可靠。

如果论文仍然写成“我们调了 negative size / reweight / lr，因此比 baseline 好”，它更像 competition report 或 engineering study。真正需要完成的升级是：

$$
\text{competition anomaly}
\rightarrow
\text{scientific question}.
$$

当前最合适的上位问题是：

> **When can agent behavior be trusted as retrieval supervision?**

Short 中不应该试图全面研究所有 trajectory supervision，而应切到一个足够窄、可验证的对象：

> **How reliable are unvisited / unbrowsed candidates as negatives in search-agent trajectories?**

设真实相关性为 $R$，Agent 是否访问文档为 $B$。LRAT 类构造隐含使用了近似：

$$
B=0 \Rightarrow R\approx0.
$$

论文真正值得回答的不是“这个近似会不会偶尔错”，而是：

$$
P(R=1\mid B=0,X)
$$

是否随检索排名、trajectory stage、已有 evidence state、Agent 类型等条件产生稳定变化。若存在这种规律，核心结论就可以从“有 false negatives”提升为：

> **Agent-derived negative supervision has systematic reliability heterogeneity.**

这比再提出一种 hard-negative weighting 更像独立研究问题，也更容易被后续 Agentic IR、trajectory supervision、retrieval training 工作引用。

理想实验闭环可以非常简单。首先用 human document-level qrels 做 ground-truth audit，把 trajectory-derived negatives 区分为 true negatives 和 false negatives；随后分析 false-negative rate 是否随 rank、search stage、evidence state、Agent 等变量呈现稳定异质性；最后如果规律足够清晰，再用一个很轻的 filtering / weighting / sampling calibration 检验这些发现是否真的能改善 retriever training。

因此最理想的论文结构不是“复杂方法”，而是：

$$
\boxed{
\text{identify failure}
\rightarrow
\text{characterize failure}
\rightarrow
\text{simple intervention}
}
$$

只证明“unvisited 中存在 relevant document”会偏薄；找到稳定、非显然的 heterogeneity，就已经达到比较合理的 Short 形态；若再跨 Agent / 数据源成立并能支持简单 calibration，整体会明显更强。

## 3. Rank2 DND 与我们到底有多大重合

DND 的比赛方案已经非常接近我们问题的前一层。他们将全部 training pairs 回连 trajectory，发现官方负样本池大小从 5 到 407 不等，56.4% 的 negatives 来自其他 sub-query 的搜索轮次；他们特别强调，普通 retrieved-but-unvisited candidate 并没有直接“不相关”证据，而 opened-but-rejected 文档具有更明确的行为否决证据。

他们据此构造的是一个 hard-negative mining 方案：使用完整因果历史作为候选池，去除答案相关分块，用 embedding similarity 排序，并优先保留“被 Agent 打开但未成为 positive”的文档，再确定性选满 9 个 negatives。换句话说，DND 已经隐含形成：

$$
\text{opened but rejected}
>
\text{retrieved but unvisited}
$$

这样的 negative-evidence hierarchy。

但当前公开 DND 的核心问题仍然是：

$$
\text{behavioral heuristic}
\rightarrow
\text{better negative mining}
\rightarrow
\text{retrieval improvement}.
$$

我们应该主动与它分叉为：

$$
\text{human qrels}
\rightarrow
\text{measure label reliability}
\rightarrow
\text{conditional heterogeneity}
\rightarrow
\text{calibration}.
$$

一句话说，DND 问的是“怎样从 trajectory 里选更好的 negatives”；我们要问的是“trajectory behavior 什么时候真的足以支持一个 negative label”。

真正会构成严重撞题的情况，是他们后续也做出 human-qrels audit，并以 rank / stage / evidence state 等条件解释 false-negative heterogeneity，再做 reliability-aware calibration。只要他们停留在 hard-negative mining 或行为启发式，这就是强相关 related work，而不是直接 novelty collision。

同时，比赛生态中已经出现其他参赛方案尝试用强 LLM judge 识别疑似 false negatives、删除或降权错误 negative。这进一步说明：**“LRAT negatives 里存在 false negatives，所以我们过滤/降权”本身已经不够新。** 我们必须把贡献提升到“行为监督的可靠性具有系统、可测、可泛化的结构”。

## 4. DND 两位作者近期 pipeline：为什么目前不用过度担心 WSDM Short 正面撞车

陈浩东是 UQ ielab 的 PhD student，研究方向包括 RAG / Agentic Search，并已有 SIGIR 2026 Full 一作工作；王率已经博士毕业并任 UQ Research Fellow，手里有持续的 SIGIR/WSDM/ECIR/EMNLP 等研究 pipeline。若他们认真把 CCIR 方向论文级推进，研究设计和写作成熟度大概率高于本科阶段的我们，这是客观风险。

但“他们能快速做 Short”与“他们愿意优先做 Short”是两回事。对已有丰富 Full-sized pipeline 的成熟 researcher，4-page Short 的边际价值明显小于对本科生。如果 DND follow-up 最终能形成更一般的研究，他们更自然的选择可能是继续补 cross-agent、qrels audit、training intervention，去投未来的 SIGIR / WWW / WSDM Full，而不是主动把它压成 Short。

他们近期公开 arXiv 也支持这一判断。最显眼的是 2026-08-03 发布的：

> **Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents（SIEVE）**

它研究的是 Deep Research Agent 如何利用网页结构做 Search–Inspect–Fetch：title、heading、section、metadata 被用于结构化 Boolean retrieval，再 selective fetch，从而减少 context token 并提升 QA accuracy。这是明显的 Full-sized Agentic Search 项目，但与我们的 supervision reliability 几乎不撞：

$$
\text{SIEVE: web structure}
\rightarrow
\text{better agent retrieval interface}
$$

而我们是：

$$
\text{trajectory behavior}
\rightarrow
\text{training-label reliability}.
$$

他们另外公开的 memory/RAG diagnostic、diffusion retrieval、LLM-ranker vulnerability 等近期工作也没有直接撞题。因此截至现在，**没有发现一篇已公开 arXiv 与我们的核心问题直接重合**。

真正不可观测的风险是：他们是否另有尚未公开的 CCIR follow-up。

WSDM 2027 Full abstract deadline 是 2026-08-17，而 DND 决赛材料日期是 8 月 12 日，比赛流程当时尚未完全结束。若他们已经把 CCIR follow-up 投向本届 WSDM Full，就意味着他们必须在比赛尚未结束时已经完成论文立项并提前注册 abstract。这在能力上完全做得到，但时间线上并不自然，目前也没有公开证据支持。

因此当前更合理的风险排序是：

$$
P(\text{已有同题 WSDM Full})=\text{低},
$$

$$
P(\text{赛后直接做同题 WSDM Short})=\text{低到中},
$$

$$
P(\text{未来扩成更成熟 Full})=\text{中等}.
$$

最后一个才是更现实的长期风险。现阶段不建议因为 Rank2 而换题，更不值得把大量研究时间消耗在猜测竞争对手；真正有效的防守是尽快把我们自己的 scientific claim 钉死，并做出第一批 qrels-grounded results。

## 5. 多数据来源在学术上是否成立

成立，而且如果设计得好，可能正是把论文从“CCIR/LRAT implementation artifact”提升为一般 Agentic IR 问题的关键。

但多数据源不能为了“实验看起来多”而堆。最理想的是让不同数据承担互补角色。

LRAT / CCIR training pairs 可以作为真实 trajectory-derived supervision pipeline，用来复现比赛 anomaly、测试 calibration 对 retriever training 的影响，并与官方 baseline、DND 风格方法建立直接联系。

BrowseComp-Plus document-level human qrels，以及《Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents》相关的多 Agent trajectories，则更适合作为 diagnostic ground truth：它们可以把 surfaced、visited、utilized 与 human relevance 解耦，从而直接研究“行为是否等价于 label”。如果数据覆盖多个 Agent，还能回答哪些 reliability pattern 是某个 Agent policy 特有的，哪些规律能跨 Agent 保持。

这样形成的不是“两个数据集重复验证”，而是：

$$
\boxed{
\text{training-supervision data}
+
\text{ground-truth diagnostic data}
}
$$

两者承担不同科学角色。

需要特别注意变量定义。不同数据中的 `browse`、`visit`、`surfaced`、`utilized`、`positive`、`relevant` 不能默认等价。论文必须明确说明哪些状态可以映射、哪些只能分别分析。如果为了强行合并而模糊定义，反而会削弱可信度。

对 4 页 Short 来说，数据源也不宜无限扩张。两个角色清楚、定义严谨的数据源，通常优于五个互相不完全可比的 benchmark。

## 6. 最终应该把 Short 做多“细”

最佳答案不是“特别特别细”，而是：

$$
\boxed{
\text{scope 窄，framing 稍宽，结论可复用}.
}
$$

过宽会导致 4 页里什么都讲不透；过细则会让论文只对某个 LRAT implementation 的某个 corner case 有意义，后续几乎没人引用。

因此建议的尺度是：

> **General question：When can agent behavior be trusted as retrieval supervision?**  
> **Narrow object：unvisited / unbrowsed candidates as negatives.**  
> **Reusable finding：negative reliability varies systematically with the search process.**

如果数据最终支持，可以把核心记忆点压成一句：

> **Unvisited does not imply irrelevant, and the reliability of trajectory-derived negatives is systematically conditioned on rank, search stage, and evidence state.**

这类 framing 对引用很友好：未来任何研究 agent trajectory supervision、behavioral feedback、retrieval negative sampling、search-agent evaluation 的论文，都可能在 motivation 或 related work 中把它作为一个可引用的 prior finding，而不是只有复现 LRAT 的人才能理解。

最理想的 Short 不需要很复杂，但应该至少让 reviewer 清楚看到三件事：现有方法用了一个隐含 assumption；这个 assumption 不是简单随机失效，而具有结构性边界条件；利用这些边界条件能做出一个简单而有效的改进。

当前综合判断是：**这个方向仍然值得以 WSDM 2027 Short 为第一目标。** 竞争风险存在，但公开信息并没有显示 Rank2 已经占住同题；真正决定稿件强弱的，不是方法是否花哨，而是能否找到非显然、稳定、最好跨 Agent 或跨互补数据源成立的 reliability heterogeneity。

下一步优先级也很明确：先与导师/师兄确认内部是否已有高度重合的 follow-up；随后尽快做最小 qrels audit，在不训练新 retriever 的情况下先看 rank / stage / evidence state 是否真的出现强规律。如果第一批图没有清晰 signal，应及时调整问题，而不是为了既定 framing 强行推进；如果规律清晰，再补最简单的 calibration 和 training closure。

---

## 参考入口

WSDM 2026 Short CFP：  
https://wsdm-conference.org/2026/index.php/call-for-short-papers/

WSDM 2027 Full / Findings CFP：  
https://www.wsdm-conference.org/2027/cffp.html

DND GitHub：  
https://github.com/Donovan0243/DND

SIEVE：  
https://arxiv.org/abs/2608.02751

> 当前仍不能确定：DND 是否注册/提交了 WSDM 2027 Full、是否计划做 WSDM Short、是否存在未公开的 qrels-based reliability follow-up，以及 WSDM 2026 Short 的真实投稿总数。后续出现公开 arXiv/GitHub 更新或导师内部信息时，应重新评估。
