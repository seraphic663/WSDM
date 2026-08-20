# WSDM 2027 Short 内部分析

> 更新日期：2026-08-20  
> 文档性质：内部决策、竞争风险与执行边界记录，不作为论文正文。  
> 来源：重构自 `CCIR_to_WSDM2027_Short_Research_Analysis.md` 与 `CCIR_to_WSDM2027_Short_strategy_v2_2026-08-19.md`。原文件保留不动。

## 0. 2026-08-20 实证状态

Stage 0/1 已执行完成。官方六组 BrowseComp-Plus Agent trajectories 共 4,980 条，全部通过发布 manifest 的字节数、SHA-256、行数和 termination 分布核验；按规范化问题文本映射 qid 为 4,980/4,980。共恢复 638,215 个 search-candidate occurrences。所有 127,591 次可恢复的成功 Search 均解析出 ranked list；visit 参数解析率为 9,796/9,807，成功 visit 到已 surfaced candidate 的映射率为 8,587/8,684，即 98.88%。97 个未映射成功 visit 明显集中于 Tongyi-DR，其单 Agent 映射率为 96.25%。

在 633,083 个“本次 Search 后、下一次 Search 前未 visit”的候选 occurrence 中，evidence-qrel positive 比例为 13.77%，query-level cluster bootstrap 95% CI [12.81%, 14.74%]；按 query 内比例再平均为 19.69% [18.49%, 20.89%]。later-visited 候选为 66.69%，其余为 12.14%，差 54.54 个百分点 [51.66, 57.43]；限制到 first exposure 后差异仍为 68.00 个百分点 [65.87, 70.11]。repeated retrieval、rank、search phase、evidence state、cross-Agent exposure 均呈稳定异质性；gold-qrel sensitivity 方向一致。

预注册联合门控结果为 **PIVOT**：signal gate 通过，但 successful-visit mapping 低于预定 99% coverage threshold。由此不启动 retriever 方法训练，不把强诊断曲线包装成 intervention gain。Stage 3 仅完成离线可行性 dry run：本地精确重建 94,113 行 query-disjoint train，93,977 行 stable provenance、136 行 ambiguous、0 mismatch，并构造每行删除量完全相同的 random、exposure-prioritized、later-visit 三臂；三臂均改变 28,880 行、删除 43,575 个 negatives、无行低于 5 个 negatives。它们未训练、不是论文结果。

正式证据入口是 `analysis/negative_reliability/`；当前四页作者可见稿是 `shortpaper/latex/demo.pdf`。下文保留竞争、venue、来源和预注册设计，用于解释为什么做出 PIVOT，而不再代表“尚未获得结果”。

## 1. 结论

当前方向仍适合作为 WSDM 2027 Short 的候选：问题范围足够窄，能够承接 LRAT 与 CCIR Cup 的数据、代码和实验资产，又有机会提出比超参数调优更一般的研究问题。决定是否继续的关键不是再训练一个模型，而是先用 document-level relevance judgments 检查 retrieved-but-unvisited candidates 的负标签可靠性是否呈现稳定、非显然的条件差异。

当前可确认四类事实：LRAT 从 Agent trajectory 中构造检索监督；本地 matched comparison、多 seed 与 paired bootstrap 没有证明 reasoning-length reweight 具有稳定独立收益；qrels-grounded candidate audit 已证明 label reliability 存在显著条件异质性；预注册 coverage gate 没有授权训练干预。calibration 或 filtering 是否带来 retriever 增益仍是未验证假设。

## 2. Venue 与格式边界

WSDM 2027 Short 是独立投稿的正式 research track，不是 Full 投稿后的降档，也不是 workshop abstract。官方允许范围较窄但完整的研究，以及 preliminary but empirically validated ideas；正文最多 4 页，图、表和附录均计入，参考文献与 Ethical Considerations 不计入正文页数。

正式投稿必须使用 ACM `acmart`：

```latex
\documentclass[sigconf,anonymous,review]{acmart}
```

当前暂定的作者可见 arXiv 版本使用同一 `acmart` 体系，但切换为：

```latex
\documentclass[sigconf,nonacm]{acmart}
```

`nonacm` 只关闭正式 ACM 出版元数据和 reference strip，不改变 `sigconf` 双栏排版。当前 demo 显示作者 `Ziming Zhao` 与邮箱 `seraphic221@outlook.com`；未来正式投稿入口不定义 arXiv mode，模板会恢复匿名 review class。

投稿 PDF 必须为英文、自包含、匿名且嵌入非标准字体。外部匿名代码或补充材料可以引用，但审稿人没有义务查看。完整作者名单在投稿系统元数据中填写；PDF 中不得显式暴露作者身份。

当前投稿截止时间为 2026-11-17 23:59 AoE，即北京时间 2026-11-18 19:59。Short 投稿入口尚未公布。会议将在香港线下举行，录用后至少一位作者需要注册并现场展示。

项目中的共享冻结模板位于 `shortpaper/latex/template.tex`，占位示例入口位于 `shortpaper/latex/demo.tex`。模板集中承载 class 选择、会议元数据和强制结构；作者身份、论文宏、实验表格和项目内容留在入口或正文文件中。当前 demo 选择 arXiv mode，未来匿名 review 入口沿用同一模板。

## 3. 为什么不是比赛总结或 Full

CCIR Cup 已提供完整训练与评测基础设施、Qwen3-Embedding-0.6B 与 E5 对照、negative group size 实验、reweight matched comparison、多 seed 与 paired bootstrap。它们的价值是提供异常现象、可靠 baseline 和实验基础设施，而不是直接构成论文贡献。

如果论文只写成 negative size、learning rate、epoch 或 reweight 的配置比较，它仍然是 competition report 或 engineering study。需要完成的升级是从“哪个配置最好”转向“Agent 行为在什么条件下足以充当可靠的 retrieval supervision”。

Full Paper 会自然引出多 benchmark、多 Agent、多 retriever family、端到端 Agent evaluation 以及更完整机制建模等要求，超出当前最自然的贡献尺度。Short 更适合一个定义精确、诊断证据扎实、干预简单且 claim 克制的问题。

Findings 是 Full 投稿经过同一审稿流程后的去向之一，不是 11 月的独立入口。内部评价 WSDM Short 时应明确其正式同行评审和 ACM DL 属性，但不要把 Short 与 Main Full 等同；学校或学院的成果认定需要另行核对。

## 4. 竞争与 novelty 风险

### 4.1 与 LRAT 的关系

LRAT 已经证明 Agent trajectory 可以提供有效的 retrieval supervision，并使用 browse/unbrowsed behavior 与 post-browse reasoning 构造训练信号。当前工作不应声称“LRAT 的关键假设是错误的”或“LRAT 引入严重标签噪声”，除非未来获得跨 Agent、跨数据且极强的证据。

更准确的 follow-up 关系是：LRAT 建立 trajectory supervision 的有效性；本工作检查其中 retrieved-but-unvisited feedback 的可靠性是否在不同搜索上下文中一致；若异质性成立，再检验简单 calibration 是否有用。

### 4.2 与 DND 的重合

DND 已经完成 training pairs 到 trajectory 的全量回连，并公开强调普通 retrieved-but-unvisited candidate 缺乏直接否定证据，而 opened-but-rejected document 具有更强行为证据。他们通过完整因果历史、答案相关分块剔除、embedding similarity 和行为否决优先来确定性选择 hard negatives。

因此，“LRAT negatives 可能含有 false negatives，所以删除或降权一部分”本身不够新。DND 与当前方向在动机层高度相关，在训练干预层存在明显邻近，但公开方案的核心仍是 heuristic hard-negative mining，而不是用 human qrels 测量标签可靠性、刻画条件异质性并解释其与训练结果的关系。

当前最安全的 novelty 分叉是：

1. 用 document-level human relevance judgments 而非行为启发式测量 label reliability；
2. 区分 surfaced、visited、utilized 与 relevant；
3. 检验可靠性是否随 rank、search stage、evidence state、Agent policy 等变量系统变化；
4. 将干预作为诊断后的验证，而不是论文的起点。

真正的高风险情形是竞争方案后续也公开 human-qrels audit、conditional heterogeneity 和 reliability-aware calibration。是否存在未公开 follow-up 当前不可观测，不应把作者个人 pipeline 或时间线猜测当成事实。

### 4.3 其他相关风险

传统 dense retrieval 已有 false-negative learning、ambiguous/hard negative sampling 和 noise-robust loss。当前工作的 novelty 不能落在“发现 false negative”或“给疑似 false negative 降权”，而要落在 Agent policy 产生的选择性行为观察及其条件结构。

《Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents》已经用 human-annotated document-level relevance judgments 区分 evidence 是否 surfaced 以及 Agent 是否有效利用证据。当前工作必须与其分叉：该文诊断的是搜索努力、检索缺口和利用缺口；本工作拟研究行为反馈作为 retriever training label 的可靠性。

## 5. 数据角色与权限边界

LRAT/CCIR training pairs 与 trajectories 适合承担 training-supervision data 的角色：复现 negative-size anomaly、建立 vanilla LRAT baseline，并检验最终干预是否影响 retriever training。

BrowseComp-Plus document-level qrels 与多 Agent trajectories 更适合承担 ground-truth diagnostic data 的角色：把 surfaced、visited、utilized 与 human relevance 分离，并检查规律是否依赖特定 Agent policy。

两类数据不能强行合并成一个统一标签空间。`browse`、`visit`、`surfaced`、`utilized`、`positive` 和 `relevant` 必须分别定义，并记录每个数据源能够观测哪些变量。老师或师兄工作中的未公开轨迹、标注和分析结果必须在获得明确许可后才能进入项目。

## 6. 独立合并审计与裁决

### 6.1 证据状态

旧 Research Analysis 明确把 false negative、confidence 和 censoring 视为假设；旧 strategy v2 的候选总结句已经接近把 reliability heterogeneity 写成发现。裁决：以较严格的证据边界为准。在获得结果前使用 research question、hypothesis、candidate explanation，不使用 `we find`、`we show` 或确定性机制表述。

### 6.2 研究对象

旧文档混用 unbrowsed 与 unvisited。裁决：正式分析至少定义四个变量：是否 surfaced/retrieved、是否 visited/browsed、是否 utilized、是否 human-relevant。只有数据语义被逐项核实后才允许映射；正文避免用斜杠把不同状态写成同义词。

### 6.3 Ground truth 与 proxy

旧 Research Analysis 同时提出 later browse、cross-agent disagreement、LLM judge 和 answer matching；旧 strategy v2 把 human qrels 作为主要审计依据。裁决：human qrels 是主要 relevance judgment；later visit、重复检索和 cross-agent disagreement 是行为一致性证据；LLM judge 与 answer matching 只作为辅助 proxy，不能被称为 ground truth。

### 6.4 方法是否锁定

旧 Research Analysis 已给出 confidence-weighted InfoNCE 公式；旧 strategy v2 要求诊断先行，并保留 filtering、weighting、sampling 多种简单干预。裁决：不锁定方法。现阶段只保留统一的 intervention interface 和必要 baseline；具体采用过滤、采样还是加权，由 Stage 0/1 结果决定。

### 6.5 贡献顺序

旧材料一度将 confidence calibration 作为标题级贡献，但 DND 和已有 false-negative work 使该表述风险较高。裁决：贡献顺序固定为 reliability measurement、conditional heterogeneity、training consequence、simple intervention。若第二项不成立，不以 heuristic 强行维持原 framing。

### 6.6 Censoring 术语

`censored feedback` 有理论吸引力，但当前没有证明数据生成机制满足严格 censoring 假设。裁决：默认使用 selective behavioral feedback 或 heterogeneous reliability；只有完成生成机制审计后才决定是否采用 censoring 术语。

### 6.7 Negative-size 因果链

现有实验只证明 negative-size performance anomaly，不证明“更多 negatives 导致更多 false negatives，进而降低性能”。裁决：将该链条拆成可检验假设，不在摘要、标题或贡献列表中提前写成机制结论。

## 7. Go / Pivot / Stop

### Go

若至少两个独立 Agent 或设置中，human-qrels reliability 都随可解释的 trajectory variables 呈现稳定结构，并且这种结构能解释部分训练异常，则继续主线。若简单干预还能稳定优于 random downsampling，论文闭环完整。

### Pivot

若 reliability heterogeneity 成立，但方法收益较小或配置依赖，则把论文定位为 empirical characterization 加 lightweight intervention，不声称普遍性能提升。

### Stop or change topic

若变量无法可靠恢复、qrels 覆盖不足、不同 proxy 互相冲突、条件曲线无稳定结构，或所有干预都不优于 random downsampling，则停止为既定故事追加 heuristic。可重新评估 reasoning-length utility calibration，但它与 outcome-aligned supervision 的既有工作更接近，需要重新做 novelty 审计。

## 8. 执行优先级

1. Stage 0：恢复 candidate-level exposure、rank、visit、later visit、stage 和 Agent 字段，报告映射率与缺失率。
2. Stage 1：用 human qrels 测量 retrieved-but-unvisited candidates 的 relevance rate，并绘制条件曲线。
3. Stage 2：检查曲线是否能解释 negative-size anomaly，而不是只做相关性叙述。
4. Stage 3：在 filtering、sampling、weighting 中选择最低复杂度干预，并加入 random-fewer-negatives baseline。
5. Stage 4：根据算力补 backbone、seed 和必要的端到端验证。

## 9. 不进入论文正文的内容

以下内容仅留在内部分析：WSDM/Findings/Workshop 的层级讨论、CCF 或校内认定、竞争团队作者履历、对未公开投稿计划的概率猜测、导师与师兄的内部重合判断、完整阅读清单、投稿费用与参会安排、逐月执行日历。

## 10. 官方与本地入口

- WSDM 2027 Short CFP：https://www.wsdm-conference.org/2027/cfsp.html
- ACM Proceedings Template：https://www.acm.org/publications/proceedings-template
- LRAT：https://arxiv.org/abs/2604.04949
- Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents：https://arxiv.org/abs/2608.01913
- DND：https://github.com/Donovan0243/DND

## 11. 研究生态与 framing 来源

### 11.1 人大 IR 与 Agentic Search 研究线

当前问题与传统 IR 的行为建模有直接连续性：human-search 时代关心 click、examination 和 behavioral bias；agentic-search 时代相应需要处理 browse、exposure、reasoning state、evidence saturation、stopping 与 tool policy。最一般的研究母题不是“Agent 很新”，而是“如何从选择性行为或 downstream outcome 中提取可靠 supervision”。

毛佳昕相关合作线对本项目有三类方法论意义：M-ASK 强调 search behavior、knowledge management 与 turn-level supervision；MAO-ARAG 强调 query difficulty 下 workflow 不应同质处理；OASES 关注 intermediate proxy 与 final outcome 是否对齐。这些工作不直接证明本项目假设，但说明 signal reliability 与 heterogeneous treatment 是自然的 IR/Agentic Search 问题。

### 11.2 LRAT 作者线

LRAT 的核心贡献是从 browsing actions、unbrowsed candidates 和 post-browse reasoning 中构造可扩展的 retriever supervision，并验证 trajectory supervision 整体有效。当前工作的叙事必须建立在这一贡献之上：LRAT 证明 usefulness，本工作测量 candidate-level reliability boundary，只有当条件异质性得到证据支持时才讨论 calibration。

旧材料中的 `unbrowsed rejection` 是对 LRAT 术语的转述，不应直接成为本项目的观测定义。新的变量体系以 surfaced、visited、utilized、human-relevant 分开记录，避免把“未访问”预先解释为“主动拒绝”。

### 11.3 PolyU/RAG/WebAgent 研究线

范文琦—李青相关合作线更偏 RAG、WebAgents、LLM recommendation、adaptive retrieval 与 trustworthy AI。它们的直接价值不是提供当前方法，而是提醒：retrieval 是 Agent pipeline 的组成部分；不同 query/stage/context 下信号可能具有不同 utility；系统成本、攻击面与 downstream outcome 不能被静态 relevance 单独概括。

### 11.4 与《Diagnosing Search Behavior and Failure Modes》的边界

该文使用 human-annotated document-level relevance judgments，区分 evidence 是否被 surfaced 以及是否被 Agent 有效使用，并将失败拆为 retrieval gap 与 utilization gap。本项目借用的是状态分离与 relevance-grounded diagnosis；研究对象则是这些行为状态一旦被转换为 retriever-training labels，其负标签可靠性是否均匀。

## 12. 竞争证据与风险记录

### 12.1 DND 公开方案的已知部分

旧策略文档记录的公开 DND 方案包括：将 training pairs 回连到 trajectory；观察 pair-level negative pool 为 5–407；报告较大比例 negatives 来自其他 sub-query；从完整因果历史构造候选池；移除答案相关分块；按 embedding similarity 排序；优先 opened-but-rejected 文档；确定性选满 9 个 negatives。其逻辑可概括为 behavior evidence + semantic hardness → hard-negative mining。

这些细节来自当时公开方案材料，后续使用前仍需回读原始 PDF/GitHub，不能只引用本分析的二手概括。尤其需要区分：DND 的 9 negatives 是其训练组设计；本项目已有 M10 使用 1 positive + 5 stored negatives，并叠加 in-batch/cross-device negatives，不能直接比较一个“负例数”。

### 12.2 与本项目的真实重合

重合较高的是动机层：普通 retrieved-but-unvisited candidate 缺乏与 opened-but-rejected 同等强度的否定证据。重合中等的是干预层：双方都可能过滤、排序、采样或降权 negatives。当前仍可保持的分叉是 human-qrels measurement → conditional reliability → training consequence；若 DND 后续也公开完成这一链条，novelty 风险将显著上升。

### 12.3 SIEVE 与其他公开工作

旧策略记录的 SIEVE（*Search, Inspect, Fetch: Exploiting Boolean Retrieval for Deep-Research Agents*，arXiv:2608.02751）研究网页结构、Boolean retrieval 与 selective fetch，目标是降低 context token 并提高 Agent QA。它与本项目的 training-label reliability 不直接重合，只能作为竞争团队研究重心的公开旁证，不能据此推断其未公开投稿计划。

### 12.4 风险分级

- **已验证风险**：generic false-negative filtering、hard-negative mining 和 confidence-weighted loss 都已有相关工作；若论文只提出 heuristic，会缺乏 novelty。
- **高可能但需实证**：单一 Agent、单一 dataset 或单一 proxy 会被认为泛化不足。
- **不可观测风险**：DND 或组内是否存在尚未公开的 qrels-based follow-up。
- **不再使用的做法**：根据作者履历或 abstract deadline 给未公开投稿赋精确概率。这类判断只适合作为内部提醒，不是证据。

## 13. 时间线、门控与资源纪律

### 13.1 文档与数据阶段

第一阶段只完成文档迁移、数据源盘点、candidate-level schema 与 coverage audit。此时不训练新模型，也不把 later visit、judge 或 answer matching 写成 ground-truth false negatives。

### 13.2 诊断阶段

第二阶段完成 qrels join、rank/stage/evidence-state curves、query-level uncertainty 和 Agent/query 分层。若 coverage 不足、字段语义不一致或条件曲线不稳定，直接进入 Pivot/Stop，而不是补 heuristic 维持故事。

### 13.3 干预阶段

只有诊断通过才运行 filtering/sampling/weighting pilot。先 smoke test 与单 seed，确认代码、样本数、negative hardness 和训练预算，再扩展到多 seed/backbone。现有 P0 结果用于提供 baseline 和实验设计经验，不能作为新机制的替代证据。

### 13.4 写作阶段

真实结果出现前，arXiv/WSDM 源码保留 hypothesis 与 TBD。四页正文的优先顺序是问题与定义、关键诊断图、最低复杂度干预、核心对照表；比赛工程、竞争推测、研究生态和完整阅读清单只留在本文件。

### 13.5 资源与冻结边界

`/home/seraphic/WSDM` 是论文工作副本，`/home/seraphic/xir` 是完整本地来源；服务器 `/root/data/LRAT` 可能包含当前权威训练状态。任何新实验都必须先核查进程、GPU、分支、数据路径和冻结清单。比赛最终 submission、正式模型、登记表和组织者复现相关材料不因论文实验而覆盖或清理。

## 14. 分级文献地图

以下条目保存旧 Research Analysis 的核心阅读路线。它是内部 novelty map，不表示所有元数据或 2026 论文状态已在本轮重新联网验证；正式引用前需回读原文。

### 14.1 S 级：直接决定 novelty boundary

1. **Learning to Retrieve from Agent Trajectories (LRAT)** — Zhou et al., 2026. https://arxiv.org/abs/2604.04949 。确认 positives/negatives/reweight 的正式定义、trajectory statistics 与作者实际 claim；本项目不能攻击原文未声称的命题。
2. **Agentic-R: Learning to Retrieve for Agentic Search** — Liu et al., 2026. https://arxiv.org/abs/2601.11888 。区分 outcome-aware positive/utility signal 与本项目 negative behavioral feedback reliability。
3. **AgentIR: Reasoning-Aware Retrieval for Deep Research Agents** — Chen et al., 2026. https://arxiv.org/abs/2603.04384 。限制“加入 reasoning context”作为创新的空间；其重点是 query representation，不是 negative supervision。
4. **OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search** — Zhang et al., 2026. https://arxiv.org/abs/2604.03675 。学习如何把 proxy misalignment 提升为清楚问题，但不复制 outcome-evaluator 体系。
5. **SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval** — Zhou et al., 2022. https://arxiv.org/abs/2210.11773 。明确 false/ambiguous negatives 并非新发现。
6. **Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization** — 2024. https://arxiv.org/abs/2401.00165 。与 weighted InfoNCE 表面接近；贡献顺序必须是 diagnosis → estimation → simple calibration。
7. **Unbiased Learning-to-Rank with Biased Feedback** — Joachims et al., WSDM 2017. https://dl.acm.org/doi/10.1145/3018661.3018699 。提供“行为观察不等于 relevance”经典背景；Agent browse 是新的 multi-step policy regime，而非 selection bias 首次出现。

### 14.2 A 级：人大毛佳昕相关合作线

8. **M-ASK** — Chen et al., 2026. https://arxiv.org/abs/2601.04703 。关注 Agentic Search 结构、turn-level supervision、search noise 与 credit assignment。
9. **MAO-ARAG** — Chen et al., 2025. https://arxiv.org/abs/2508.01005 。query complexity 不同，workflow 不应同质处理。
10. **Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning** — Chen et al., 2025. https://arxiv.org/abs/2501.15228 。关注 downstream reward 如何反传到 retrieval/reformulation。
11. **CoSearchAgent** — Gong, Li, Mao, SIGIR 2024 Short. https://sites.google.com/site/maojiaxin/ 。作为聚焦问题、轻量方法和紧凑实验的 Short 写法参考。
12. **Scaling Laws for Dense Retrieval** — Fang et al., SIGIR 2024 Best Paper. https://sigir.org/awards/best-paper-awards/ 。学习如何把 empirical phenomenon 提升为 generalizable question。

### 14.3 A 级：PolyU 范文琦—李青相关合作线

13. **A Survey on RAG Meeting LLMs** — Fan et al., KDD 2024. https://arxiv.org/abs/2405.06211 。建立 RAG training/system 背景。
14. **A Survey of WebAgents** — Ning et al., KDD 2025. https://arxiv.org/abs/2503.23350 。补足 tool/action/observation 视角。
15. **Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation** — Wang et al., ACL 2025. https://arxiv.org/abs/2501.02226 。观察 retrieved knowledge quality 与 downstream utility 的叙事。
16. **Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs** — Huang et al., WSDM 2026. https://arxiv.org/abs/2503.09382 。参考 WSDM 对 LLM assistant behavior/evaluation 的包装方式。
17. **Inference Cost Attacks for Retrieval-Augmented Large Language Models** — Liu et al., WWW 2026. https://wenqifan03.github.io/publications.html 。提醒 retrieval decision 有 downstream cost，不能只看静态 relevance。

### 14.4 A-/B+：Agentic Retrieval 前沿

18. **Rethinking Reasoning-Intensive Retrieval** — Zhao et al., 2026. https://arxiv.org/abs/2605.04018 。static retrieval quality 与 agentic utility 未必一致；关注 agent-in-the-loop evaluation。
19. **A Picture of Agentic Search** — 2026. https://arxiv.org/abs/2602.17518 。参考 reasoning-induced queries、documents、thoughts 与多 Agent trajectory 组织。
20. **Towards Retrieving Interaction Spaces for Agentic Search** — Zhuang et al., 2026. https://arxiv.org/abs/2606.06880 。避免把 Agentic Retrieval 简化为一次 top-k ranking。
21. **Rethinking Agentic Search with PI-SERINI** — Hsu, Yang, Lin, 2026. https://arxiv.org/abs/2605.10848 。提醒保留 lexical/BM25 baseline 的潜在价值。

### 14.5 B 级：Negative Sampling 与 Label Noise

22. **Negative Sampling Techniques in Information Retrieval: A Survey** — Wischounig et al., 2026. https://arxiv.org/abs/2603.18005 。用于建立 random/hard/dynamic/false-negative/LLM-generated taxonomy。
23. **TriSampler** — Yang et al., 2024. https://arxiv.org/abs/2402.11855 。理解 informative negative 的一般原则。
24. **Noisy Pair Corrector for Dense Retrieval** — EMNLP Findings 2023. https://aclanthology.org/2023.findings-emnlp.765/ 。若 pair-level label noise 显著，可作为 correction 类 related work。

## 15. 旧文档标题级迁移审计

本表覆盖两份旧文档的 118 个 Markdown 标题。`完整迁移` 表示核心论点与必要细节已进入新文件；`合并重写` 表示重复内容已在更严格的证据边界下统一；`有意降级` 表示旧内容仍作为内部风险提醒，但不再保留缺乏直接证据的精确概率或作者意图推断。完成本表后，新文档的功能不依赖旧文档；旧文件仅保留历史 provenance，是否删除另行确认。

| 来源 | 旧标题 | 处理 | 新位置 |
|---|---|---|---|
| Research Analysis | CCIR Cup → WSDM 2027 Short：研究定位、创新边界与执行路线 | 合并重写 | draft §§1–3, 12; analysis §1 |
| Research Analysis | 0. 结论先行 | 合并重写 | draft §§1–3, 12; analysis §1 |
| Research Analysis | 1. 当前项目资产：比赛结果不是论文贡献，但非常值钱 | 合并重写 | analysis §3; draft §§1, 10 |
| Research Analysis | 1.1 已经拥有的资产 | 合并重写 | analysis §3; draft §§1, 10 |
| Research Analysis | 1.2 已有实验真正透露出的研究信号 | 合并重写 | analysis §3; draft §§1, 10 |
| Research Analysis | 2. 为什么目标是 WSDM 2027 Short，而不是 Full / Findings / Workshop | 完整迁移 | analysis §§2–3 |
| Research Analysis | 2.1 WSDM Short 的制度位置 | 完整迁移 | analysis §§2–3 |
| Research Analysis | 2.2 为什么不应为了“更硬”硬冲 Full | 完整迁移 | analysis §§2–3 |
| Research Analysis | 2.3 Findings 与 Workshop | 完整迁移 | analysis §§2–3 |
| Research Analysis | 3. 导师与研究生态：这个题目并不突兀 | 完整迁移 | analysis §§11, 14 |
| Research Analysis | 3.1 毛佳昕相关研究线 | 完整迁移 | analysis §§11, 14 |
| Research Analysis | 3.2 LRAT 作者线 | 完整迁移 | analysis §§11, 14 |
| Research Analysis | 3.3 PolyU 范文琦—李青相关合作线 | 完整迁移 | analysis §§11, 14 |
| Research Analysis | 4. 核心研究问题：从 “Unbrowsed ≠ Irrelevant” 升级为 selective / censored behavioral supervision | 完整迁移 | draft §2 |
| Research Analysis | 4.1 不要把论文写成一句过强的反驳 | 完整迁移 | draft §2 |
| Research Analysis | 4.2 一个更完整的生成视角 | 完整迁移 | draft §2 |
| Research Analysis | 5. 建议的研究假设 | 完整迁移 | draft §12 |
| Research Analysis | H1：Unbrowsed negatives 中存在非忽略的潜在正例 / 高 utility 文档 | 完整迁移 | draft §12 |
| Research Analysis | H2：这种噪声不是均匀随机噪声，而是随 trajectory context 系统变化 | 完整迁移 | draft §12 |
| Research Analysis | H3：negative 数量增加时，低置信度 negatives 被引入的概率上升 | 完整迁移 | draft §12 |
| Research Analysis | H4：negative-size 的性能异常与 contamination / confidence 变化相关 | 完整迁移 | draft §12 |
| Research Analysis | H5：基于 trajectory evidence 的 selective calibration 优于随机少取 negatives | 完整迁移 | draft §12 |
| Research Analysis | 6. 最小方法：Confidence-Calibrated Trajectory Negatives | 完整迁移 | draft §§6, 13 |
| Research Analysis | 6.1 Vanilla LRAT-style objective | 完整迁移 | draft §§6, 13 |
| Research Analysis | 6.2 Confidence weighting | 完整迁移 | draft §§6, 13 |
| Research Analysis | 6.3 $c$ 的候选构造 | 完整迁移 | draft §§6, 13 |
| Research Analysis | A. Later-browse / repeated-retrieval confidence | 完整迁移 | draft §§6, 13 |
| Research Analysis | B. Exposure-aware confidence | 完整迁移 | draft §§6, 13 |
| Research Analysis | C. Cross-trajectory / cross-agent disagreement | 完整迁移 | draft §§6, 13 |
| Research Analysis | D. Independent relevance / utility judge | 完整迁移 | draft §§6, 13 |
| Research Analysis | E. Answer-evidence matching | 完整迁移 | draft §§6, 13 |
| Research Analysis | 7. 实验设计：先诊断，再做方法 | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.1 Stage 0：数据可行性审计 | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.2 Stage 1：证明 reliability heterogeneity | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.3 Stage 2：解释比赛中的 negative-size anomaly | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.4 Stage 3：方法对照 | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.5 Stage 4：robustness | 完整迁移 | draft §§5, 15 |
| Research Analysis | 7.6 Metrics | 完整迁移 | draft §§5, 15 |
| Research Analysis | 8. Reasoning-length reweight：保留，但降级为 secondary analysis | 合并重写 | analysis §§1, 7; draft §§10, 17 |
| Research Analysis | 8.1 作为 supporting evidence | 合并重写 | analysis §§1, 7; draft §§10, 17 |
| Research Analysis | 8.2 作为失败后的 B 计划 | 合并重写 | analysis §§1, 7; draft §§10, 17 |
| Research Analysis | 9. 与 LRAT 的关系：如何避免“拿 WSDM Short 砸自家 SIGIR Full 的脚” | 完整迁移 | analysis §4.1; draft §§8, 14 |
| Research Analysis | 9.1 不应这样写 | 完整迁移 | analysis §4.1; draft §§8, 14 |
| Research Analysis | 9.2 推荐这样写 | 完整迁移 | analysis §4.1; draft §§8, 14 |
| Research Analysis | 9.3 学术上真正精确的 claim | 完整迁移 | analysis §4.1; draft §§8, 14 |
| Research Analysis | 10. 这个 focus 会不会“太琐碎”？ | 完整迁移 | draft §§11, 16 |
| Research Analysis | 10.1 会显得琐碎的版本 | 完整迁移 | draft §§11, 16 |
| Research Analysis | 10.2 足够成为 WSDM Short 的版本 | 完整迁移 | draft §§11, 16 |
| Research Analysis | 层次 A：现象不是个例 | 完整迁移 | draft §§11, 16 |
| Research Analysis | 层次 B：现象能解释已有训练行为 | 完整迁移 | draft §§11, 16 |
| Research Analysis | 层次 C：针对性修正优于 trivial baseline | 完整迁移 | draft §§11, 16 |
| Research Analysis | 10.3 不要再扩大一级 | 完整迁移 | draft §§11, 16 |
| Research Analysis | 11. Reviewer 最可能攻击的点与预防方式 | 完整迁移 | draft §14 |
| Research Analysis | Attack 1：“False negatives 在 dense retrieval 里早就有人研究了。” | 完整迁移 | draft §14 |
| Research Analysis | Attack 2：“这不就是 click / position bias？” | 完整迁移 | draft §14 |
| Research Analysis | Attack 3：“你的 proxy 不是真 relevance。” | 完整迁移 | draft §14 |
| Research Analysis | Attack 4：“你只是少用了 negatives。” | 完整迁移 | draft §14 |
| Research Analysis | Attack 5：“只有一个 dataset / competition setting，泛化不足。” | 完整迁移 | draft §14 |
| Research Analysis | Attack 6：“为什么不直接用 stronger reranker / LLM judge 清洗？” | 完整迁移 | draft §14 |
| Research Analysis | Attack 7：“与 LRAT 的 novelty 太近。” | 完整迁移 | draft §14 |
| Research Analysis | 12. 4 页 Short Paper 的推荐结构 | 完整迁移 | draft §16 |
| Research Analysis | Page 1：Problem + Motivation | 完整迁移 | draft §16 |
| Research Analysis | Page 2 上半：Diagnosis | 完整迁移 | draft §16 |
| Research Analysis | Page 2 下半：Method | 完整迁移 | draft §16 |
| Research Analysis | Page 3–4：Experiments | 完整迁移 | draft §16 |
| Research Analysis | 13. 执行时间线：2026-08-18 → 2026-11-17 | 完整迁移 | analysis §13 |
| Research Analysis | 8 月下旬：只做数据审计，不急着训练 | 完整迁移 | analysis §13 |
| Research Analysis | 9 月：机制验证 + 最小方法 | 完整迁移 | analysis §13 |
| Research Analysis | 10 月：稳定性与补实验 | 完整迁移 | analysis §13 |
| Research Analysis | 11 月 1–10 日：压缩成 4 页 | 完整迁移 | analysis §13 |
| Research Analysis | 11 月 11–17 日：投稿检查 | 完整迁移 | analysis §13 |
| Research Analysis | 14. Go / Pivot / Stop 判断 | 完整迁移 | analysis §7; draft §11 |
| Research Analysis | Green：继续主线 | 完整迁移 | analysis §7; draft §11 |
| Research Analysis | Yellow：继续但弱化 claim | 完整迁移 | analysis §7; draft §11 |
| Research Analysis | Red：及时 pivot | 完整迁移 | analysis §7; draft §11 |
| Research Analysis | 15. 推荐标题与一句话定位 | 完整迁移 | draft §17 |
| Research Analysis | 最推荐 | 完整迁移 | draft §17 |
| Research Analysis | 更强调研究问题 | 完整迁移 | draft §17 |
| Research Analysis | 更一般但风险更大 | 完整迁移 | draft §17 |
| Research Analysis | 16. 论文 / Reference 阅读清单 | 完整迁移 | analysis §14 |
| Research Analysis | S 级：必须精读，直接决定 novelty boundary | 完整迁移 | analysis §14 |
| Research Analysis | 1. Learning to Retrieve from Agent Trajectories — LRAT | 完整迁移 | analysis §14 |
| Research Analysis | 2. Agentic-R: Learning to Retrieve for Agentic Search | 完整迁移 | analysis §14 |
| Research Analysis | 3. AgentIR: Reasoning-Aware Retrieval for Deep Research Agents | 完整迁移 | analysis §14 |
| Research Analysis | 4. OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search | 完整迁移 | analysis §14 |
| Research Analysis | 5. SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval | 完整迁移 | analysis §14 |
| Research Analysis | 6. Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization | 完整迁移 | analysis §14 |
| Research Analysis | 7. Unbiased Learning-to-Rank with Biased Feedback | 完整迁移 | analysis §14 |
| Research Analysis | A 级：人大毛佳昕相关合作线 | 完整迁移 | analysis §14 |
| Research Analysis | 8. Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search — M-ASK | 完整迁移 | analysis §14 |
| Research Analysis | 9. MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation | 完整迁移 | analysis §14 |
| Research Analysis | 10. Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning | 完整迁移 | analysis §14 |
| Research Analysis | 11. CoSearchAgent: A Lightweight Collaborative Search Agent with Large Language Models | 完整迁移 | analysis §14 |
| Research Analysis | 12. Scaling Laws for Dense Retrieval | 完整迁移 | analysis §14 |
| Research Analysis | A 级：PolyU 范文琦—李青相关合作线 | 完整迁移 | analysis §14 |
| Research Analysis | 13. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models | 完整迁移 | analysis §14 |
| Research Analysis | 14. A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models | 完整迁移 | analysis §14 |
| Research Analysis | 15. Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation | 完整迁移 | analysis §14 |
| Research Analysis | 16. Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs | 完整迁移 | analysis §14 |
| Research Analysis | 17. Inference Cost Attacks for Retrieval-Augmented Large Language Models | 完整迁移 | analysis §14 |
| Research Analysis | A-/B+：当前 Agentic Retrieval 前沿，建议快速扫 | 完整迁移 | analysis §14 |
| Research Analysis | 18. Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems | 完整迁移 | analysis §14 |
| Research Analysis | 19. A Picture of Agentic Search | 完整迁移 | analysis §14 |
| Research Analysis | 20. Towards Retrieving Interaction Spaces for Agentic Search | 完整迁移 | analysis §14 |
| Research Analysis | 21. Rethinking Agentic Search with PI-SERINI: Is Lexical Retrieval Sufficient? | 完整迁移 | analysis §14 |
| Research Analysis | B 级：Negative Sampling / Label Noise 方法储备 | 完整迁移 | analysis §14 |
| Research Analysis | 22. Negative Sampling Techniques in Information Retrieval: A Survey | 完整迁移 | analysis §14 |
| Research Analysis | 23. TriSampler: A Better Negative Sampling Principle for Dense Retrieval | 完整迁移 | analysis §14 |
| Research Analysis | 24. Noisy Pair Corrector for Dense Retrieval | 完整迁移 | analysis §14 |
| Research Analysis | 17. 最终判断 | 合并重写 | analysis §1; draft §§11, 17 |
| Strategy v2 | CCIR Cup → WSDM 2027 Short：方向判断、竞争风险与投稿策略 | 合并重写 | analysis §§2–3 |
| Strategy v2 | 1. WSDM Short 到底是什么，值不值得作为目标 | 合并重写 | analysis §§2–3 |
| Strategy v2 | 2. CCIR Cup 工作为什么不能直接整理成论文 | 合并重写 | analysis §3; draft §§1, 12, 15 |
| Strategy v2 | 3. Rank2 DND 与我们到底有多大重合 | 完整迁移 | analysis §12; draft §§8, 14 |
| Strategy v2 | 4. DND 两位作者近期 pipeline：为什么目前不用过度担心 WSDM Short 正面撞车 | 有意降级 | analysis §12.3–12.4 |
| Strategy v2 | 5. 多数据来源在学术上是否成立 | 完整迁移 | analysis §5; draft §4 |
| Strategy v2 | 6. 最终应该把 Short 做多“细” | 合并重写 | analysis §§1, 7; draft §§1, 11, 16 |
| Strategy v2 | 参考入口 | 完整迁移 | analysis §§10, 14 |
