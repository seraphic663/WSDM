## 下一阶段工作

> 执行状态（2026-08-20）：知识迁移、Stage 0、Stage 1、Stage 2 对齐、条件式 Stage 3 dry run、真实论文替换与四页编译均已完成。预注册联合门控为 **PIVOT**：信号门通过，但 successful-visit mapping 为 98.88%，低于 99% coverage threshold，因此没有启动 GPU retriever training。以下内容保留为执行协议；正式结果见 `analysis/negative_reliability/README.md`、`shortpaper/analysis.md`、`shortpaper/draft.md` 与 `shortpaper/latex/demo.pdf`。

### 1. 先完成两份新文档的知识迁移

不会继续简单压缩。目标是让：

```
shortpaper\analysis.md
shortpaper\draft.md
```

覆盖两份旧文件的核心内容，同时保持内部分析与论文内容的分离。

具体做法：

1. 建立旧章节到新章节的迁移表，对旧文件全部 118 个标题逐项标记：
   - 已完整迁移；
   - 已合并压缩；
   - 应补回；
   - 有意淘汰及原因。
2. 向 `analysis.md` 补回：
   - DND/SIEVE 竞争证据链；
   - 研究生态与 novelty 边界；
   - 24 篇文献的分级阅读地图；
   - venue、时间线和风险记录；
   - 完整 Go/Pivot/Stop 依据；
   - 来源 URL 和已知/未知边界。
3. 向 `draft.md` 补回：
   - H1-H5 的完整可证伪表述；
   - 五类 reliability signal：later visit、exposure、cross-Agent、judge、answer matching；
   - reviewer 的七类潜在攻击及回应；
   - 公式和不同 intervention 候选；
   - 更细的 baseline、metrics、robustness；
   - 四页论文逐页空间规划；
   - 标题、摘要和 claim 备选。
4. 避免把已经被新版本纠正的内容重新带回来：
   - 不混用 unbrowsed/unvisited；
   - 不把行为 proxy 称为 ground truth；
   - 不提前写成 `we find`；
   - 不锁死 confidence weighting；
   - 不把单次分数变化解释成机制。
5. 更新根 README 与组会材料的链接，使其转向新文件。
6. 完成覆盖审计后，把旧文件标记为 superseded。是否删除仍单独等你确认。

文档迁移完成前，不建议删除旧文件。

### 2. Stage 0：数据可行性审计

这是实验上的第一步，不训练新模型。

需要建立 candidate-level 表，至少包含：

```
query_id
trajectory_id
agent
search_step
doc_id
rank
surfaced
visited
later_visited
utilized
human_relevance
evidence_state
source_dataset
```

先回答：

- 每次 Search 的完整结果列表能恢复多少；
- visit 能否稳定映射回 candidate；
- qrels 与 document ID 的连接率；
- 重复检索和 later visit 的数量；
- stage/evidence-state 是否可恢复；
- 缺失或映射失败是否集中在特定 Agent、rank 或难题上。

同时严格区分四个层级：

1. 单次 Search candidates；
2. 保存到 Training-pairs 的 pair-level negative pool；
3. query-level 聚合；
4. 训练时实际抽取的 negatives。

现有 M10 是从保存的 pool 中取 `1 positive + 5 negatives`，还会加入 in-batch/cross-device negatives，不能把这些层级混成一个“负例数”。

### 3. Stage 1：Negative Reliability Audit

在覆盖率通过后，先做诊断图，不训练 retriever：

- \(P(R=1\mid S=1,V=0)\) 总体值；
- rank bucket；
- early/late search；
- evidence saturation；
- repeated retrieval；
- later visit；
- Agent；
- query difficulty。

统计上以 query 为重采样单位，给出 query-level bootstrap CI，不能把同一 trajectory 的候选当成独立样本。

Human qrels 是主要 relevance judgment；later visit 和 cross-Agent disagreement 单独作为行为稳定性指标。只有完成口径审计后，才决定是否使用 false-negative rate 这个名称。

### 4. Stage 2：连接现有比赛异常

随后把 reliability 诊断与已有 negative-size/reweight 结果对齐。

现有 P0 已经有多 seed、单变量对照和 query-level paired bootstrap 基础，但在正式复用前需要重新检查当前产物与 split。已知边界仍是：

- reweight 没有稳定独立增益；
- 方向依赖 LR、group size 和 seed；
- 关键 CI 跨零；
- Dev1500 只能作为离线检索分析；
- 不能写成新的排行榜收益或机制证明。

这一阶段要检验的是：

```
negative-set reliability
↔
training configuration
↔
retrieval outcome
```

而不是先假定：

```
more negatives
→
more false negatives
→
worse performance
```

### 5. Stage 3：有信号后再做方法实验

只有当 qrels-grounded heterogeneity 稳定出现，才选择 intervention：

- filtering；
- reliability-aware sampling；
- confidence weighting。

必要 baseline：

1. base retriever；
2. vanilla LRAT；
3. randomly fewer negatives；
4. rank/exposure heuristic；
5. reliability-aware intervention；
6. 条件允许时加入 DND-style hard-negative 或 noise-robust baseline。

先小样本、单 seed 验证代码和方向；通过 gate 后再做 Qwen/E5、多 seed 和 paired bootstrap，避免在弱信号上直接消耗完整训练预算。

### 6. 写作与 LaTeX

实验推进时同步把 demo 替换成真实稿件：

- 保留《Diagnosing...》的问题—诊断—发现—干预节奏；
- 数字出来前继续保留 TBD 和 hypothesis wording；
- 图 1 优先做 coverage + conditional reliability；
- 主表优先比较 vanilla、random fewer negatives 和最终 intervention；
- 当前 arXiv 入口保持作者可见；
- 真正投稿时切换到匿名 review 模式，无需复制正文。

### 7. 暂时不会做的事情

- 不删除两份旧文档；
- 不把老师/师兄未授权的数据接入实验；
- 不在 Stage 0 前启动完整 retriever 训练；
- 不把 later visit 直接叫作 false negative；
- 不把现有 P0 单点差异包装成论文机制；
- 不公开上传 arXiv，除非作者、数据和题目边界已确认。

总体顺序是：

```
知识无损迁移
→ 数据覆盖审计
→ qrels-grounded diagnosis
→ Go/Pivot
→ 最小干预
→ 多 seed 验证
→ 正式论文替换 demo
```
