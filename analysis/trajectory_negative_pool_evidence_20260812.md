# LRAT trajectory negative pool、pairs negative 与统计口径核查

日期：2026-08-12（Asia/Shanghai）

范围：只读核查官方 LRAT 论文 HTML、官方 GitHub `src/data_builder.py`、本地官方 clone、发布 pairs、本地 provenance/统计脚本和 query-disjoint manifest；本轮没有读取或输出凭据，没有启动训练，没有修改代码或既有文件。状态标记：**VERIFIED** = 本轮来源直接支持；**INFERENCE** = 基于直接证据的谨慎解释；**UNKNOWN** = 当前来源不足以确认。

## 结论先行

1. **在数据构造链上，pairs 的 `neg`/`neg_id` 就是由 trajectory 检索事件派生出的负例字段，但不能把它简化为“一个 Search 的 top-k 其余项”。** 论文给出的是同一 retrieved candidate set 中未 Browse 文档的 naive negatives；当前官方 builder 实际把本轮/历史的未满足 Browse 文档和搜索历史中的未 Browse 文档合并后去重，再写入 `neg`/`neg_id`。
2. **论文中的 top-k 是每一次 Search 的候选返回规模，不是每个 pair 的负例上限。** 论文实验设置写明每次 Search 返回 top-10；一条 trajectory 可以有多个 Search，builder 还会在一次满意的 Browse 发生前保留搜索历史，所以一个 pair 的负例列表可以跨多个 Search 累积，超过 9，更不等于某次 Search 返回了几百条。
3. **发布 pairs 的 `min/median/max = 5/9/407` 的具体来源是 96,504 条 JSONL 记录中每一行 `len(row["neg"])` 的分布。** 它不是 `search_result_count` 的分布，也不是 query-disjoint manifest 中按 normalized query 聚合后的分布。本轮对 3,883,089,616-byte 原始 JSONL 做了只读流式复核：96,504 行、`neg` 与 `neg_id` 长度 0 处不一致、`satisfied=true` 96,504 行。
4. **`early_stop_v1` 的 `negatives_per_query` 是另一个统计。** 它先把同一 normalized query 的所有 source rows 的正负 ID 做并集，并从负例中删除任何在该 query 组中出现过的正例；因此 dev1500 的 `min/median/max = 5/9/312` 是 query-group 聚合结果，不应替代全量 pairs 每行的 `5/9/407`。

## 1. “trajectory 的负例池是否就是 pairs 的 negative”

### 1.1 论文层面的定义

**VERIFIED：** 论文 §4.1.2 说明 trajectory 生成时每个 Search 返回 top-10 candidate documents；见 [arXiv HTML §4.1.2](https://arxiv.org/html/2604.04949#S4.SS1.SSS2)，本地论文 HTML 当前抓取内容的正文行 132–135 也明确写出 top-10。

**VERIFIED：** 论文 §5.1.1 把一次 Search→Browse 中被 Browse 的文档作为 naive positive，把同一 retrieved candidate set 中除该文档外、未被 Browse 的候选定义为 naive negatives，公式为 `N_t = D_t \ {d_{t+1}}`；见 [arXiv HTML §5.1.1](https://arxiv.org/html/2604.04949#S5.SS1.SSS1)，HTML 正文行 170–176。

**VERIFIED：** 论文 §5.1.2 随后用 Browse 后紧邻的 reasoning 做 LLM relevance filter，标签对象是被 Browse 的 `(q_t, d_{t+1})`；见 [arXiv HTML §5.1.2](https://arxiv.org/html/2604.04949#S5.SS1.SSS2)，HTML 正文行 177–181。论文 §5.2.2 还说明 InfoNCE 的负例集合同时包含同一 retrieved candidate set 的未 Browse 文档和 mini-batch 内其他 query 的文档；见 [arXiv HTML §5.2.2](https://arxiv.org/html/2604.04949#S5.SS2.SSS2)，HTML 正文行 204–212。

**边界：** 因此“trajectory negative pool”在论文里首先是 Search→Browse 事件的候选级概念；论文并没有说存在一个对整条 trajectory、整条原始用户 query 永远固定的单一负例列表。

### 1.2 官方 builder 的实际写出关系

**VERIFIED：** 官方 `src/data_builder.py` 的输入是 corpus、trajectory JSON 目录和 judge endpoint，输出字段包含 `query`, `pos`, `neg`, `pos_id`, `neg_id`, `reasoning_len`, `satisfied`, `reweight_rate`；见本地官方 clone [`LRAT/src/data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:4)–14 和官方 [raw source](https://raw.githubusercontent.com/Yuqi-Zhou/LRAT/main/src/data_builder.py#L3-L13)。

**VERIFIED：** builder 解析每次 Search 的 `DocID:` 列表，并将每个 Search 的文档列表追加到 `history_search_results`；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:225)–246、278–289。它对连续 Browse 逐个读取 docid，取紧邻的下一步 reasoning，并调用 judge 得到 `satisfied`；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:291)–325。

**VERIFIED：** 负例构造不是单一的 `current_search_docs[1:]`：`make_neg_for_unsat_doc` 合并本轮 unsatisfied Browse、此前 unsatisfied Browse 和 `unbrowsed_docs`；`make_neg_for_sat_sample` 也合并本轮/此前 unsatisfied Browse 与 `unbrowsed_docs`，然后排除本轮 `sat_set` 中的文档并按出现顺序去重；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:328)–346。`unbrowsed_docs` 从所有当前保留的历史 Search 结果中反向收集；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:238)–246。

**VERIFIED：** builder 最终将 `neg_texts` 写入 `neg`，将对应的文档 ID 写入 `neg_id`；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:352)–375。一个 Browse 事件对应一个训练 sample，而不是一条 trajectory 只写一个全局 negative list。

**VERIFIED：** 当本轮出现至少一个 `run_sat_docs` 时，builder 才清空 `history_search_results`、`history_browsed_unsat` 和当前 query；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:377)–385。因此在满意 Browse 之前发生的多个 Search 可以共同贡献该 sample 的 `neg`。

**VERIFIED：** 官方 GitHub 当前 main 的相同实现可由 [GitHub `src/data_builder.py`](https://github.com/Yuqi-Zhou/LRAT/blob/main/src/data_builder.py#L215-L353) 以及 [raw source](https://raw.githubusercontent.com/Yuqi-Zhou/LRAT/main/src/data_builder.py#L215-L353) 回读；本地 `LRAT` clone 的 `src/data_builder.py` 未被本地工作树修改，文件内容 SHA-256 为 `61871d894f68d67ba1198a0777548846b73b5e043f3df7827001fe45608fd00e`，本地文件最近一次相关提交为 `68259350ace254245b17c09480df77878acd906e`。

**结论：** 对“是否就是 pairs 的 negative”的最准确回答是：**是，发布 pairs 的 `neg`/`neg_id` 是从 trajectory 的搜索/Browse 事件池派生、筛选后写出的负例字段；但它是按 pair sample 生成的、可能跨历史 Search 的负例池，不是论文公式的单次 top-10 候选集的机械复制。** 另外，发布 pairs 已经没有 trajectory ID；provenance 脚本的模块说明也明确记录了这一点，见 [`build_trajectory_provenance.py`](/home/seraphic/xir/solution/src/build_trajectory_provenance.py:2)–7，因此不能仅凭 pairs 行反推出完整事件边界。

## 2. 为什么单次 Search top-k 不是几百，但一个 query 可能有 400+ negative

**VERIFIED：** 论文正文把 Search 返回抽象为 ranked top-K candidate set，并在 trajectory generation 设置中固定为 top-10；见 [§3.1](https://arxiv.org/html/2604.04949#S3.SS1) HTML 行 105–109 和 [§4.1.2](https://arxiv.org/html/2604.04949#S4.SS1.SSS2) HTML 行 132–135。

**VERIFIED：** 当前 builder 的 `history_search_results` 是“每个 Search 一个 doc list”的列表；在有满意 Browse 之前，多个 Search 的 doc lists 都会保留。`_collect_unbrowsed_docs` 遍历这些历史列表，再由 `_unique_preserve_order` 去重；见 [`data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py:238)–246、278–289、328–346。

**INFERENCE：** 如果多个 Search 各自返回约 10 个候选，且跨 Search 的文档重复较少，那么把历史未 Browse 文档合并后，负例数量自然可以达到几十、几百；此外 builder 还会把此前被 Browse 但被 judge 判为 unsatisfied 的文档作为负例来源。要达到 400+，需要较长的搜索历史和较少的文档重复；但仅凭发布 pair 行不能反推出该行究竟经历了多少次 Search。

**VERIFIED：** 全量发布 pairs 的每行 `neg` 数量实际范围是 5–407，具体统计见 [`solution/outputs/data_report.json`](/home/seraphic/xir/solution/outputs/data_report.json:1)–32 和 [`solution/outputs/data_report.md`](/home/seraphic/xir/solution/outputs/data_report.md:18)–29。该报告还记录 96,504 行全部 `satisfied=true`，所以当前发布版本的 `neg` 不是包含 `satisfied=false` 正样本行的混合统计；但这不等于负例全部来自“未 Browse”，因为 builder 对满意 sample 的负例构造仍可包含此前 unsatisfied Browse 文档。

**VERIFIED：** 本轮对原始 pairs 做了只读流式交叉检查，结果为：`rows=96504`；`len(row["neg"])` 的 `mean=19.610700074608`、`median=9`、`p95=58`、`min=5`、`max=407`；`len(row["neg"])` 与 `len(row["neg_id"])` 不一致数为 `0`；`satisfied=true` 为 `96,504` 行。该检查没有写入输出文件。

**边界修正：** 本地实际保存有完整的 [`trajectories.tar.gz`](/home/seraphic/xir/data/raw/LRAT-Train/trajectories.tar.gz)，但本轮没有为整个 archive 重建完整 event-level `search_result_count` 分布。我们额外对最大 pair 行做了定点回溯：pairs 第 94,164 行的 `neg_id` 为 407，唯一匹配到 `trajectories/bm25_true/1633_bm25.json` 的 `search_idx=74`（零基），截至该 Browse 已发生 75 次 Search，每次 10 个结果，累计 408 个不重复候选；该行的 `neg_id` 集合恰好等于累计候选集合去掉 1 个正例。因此，至少这一条 407 的直接来源已经由 trajectory 事件核对；但全量事件级 `search_result_count` 的 min/median/max 仍未在本轮重算。

## 3. min/max/median 的具体数据来源

### 3.1 全量 raw pairs：每行 negative 数量

| 统计对象 | count | min | median | mean | p95 | max | 具体来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原始 pairs 每行 `len(neg)` | 96,504 | 5 | 9 | 19.610700074608307 | 58 | 407 | [`data_report.json`](/home/seraphic/xir/solution/outputs/data_report.json:26)–32；原始文件 [`LRAT-training-pairs.jsonl`](/home/seraphic/xir/data/raw/LRAT-Train/LRAT-training-pairs.jsonl) |

**VERIFIED：** 可复用的本地审计函数在 [`solution/src/audit_early_stop_split.py`](/home/seraphic/xir/solution/src/audit_early_stop_split.py:25)–36 中定义排序、`statistics.median`、最小值、最大值和 p95；其 `row_stats` 在 39–45 行明确使用 `len(row["neg"])`。官方 quick-stat helper [`LRAT/src/training_data_stats.py`](/home/seraphic/xir/LRAT/src/training_data_stats.py:36)–37 也使用 `len(row.get("neg", []))`，但它只输出平均 negative count，不是当前 min/median/max 报告的完整来源；官方文档 [training_data_construction.md §4](https://github.com/Yuqi-Zhou/LRAT/blob/main/docs/training_data_construction.md#4-quick-statistics) 也只列 average negative count。

**VERIFIED：** raw pairs 的身份记录在 [`data/processed/early_stop_v1/manifest.json`](/home/seraphic/xir/data/processed/early_stop_v1/manifest.json:12)–25：96,504 行、3,883,089,616 bytes、SHA-256 `dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9`，非空 normalized query 组 80,889 个。该 manifest 的 `rows_per_nonempty_query` 统计是每组 source row 数量，**不是** negative 数量。

### 3.2 `early_stop_v1` dev1500：按 normalized query 并集后的 negative 数量

| 统计对象 | count | min | median | mean | p95 | max | 具体来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| dev1500 `negatives_per_query` | 1,500 | 5 | 9 | 20.504666666666665 | 61 | 312 | [`early_stop_v1/manifest.json`](/home/seraphic/xir/data/processed/early_stop_v1/manifest.json:54)–60 |
| dev1500 `candidates_per_query` = positives + negatives | 1,500 | 9 | 10 | 21.640666666666668 | 62 | 313 | [`early_stop_v1/manifest.json`](/home/seraphic/xir/data/processed/early_stop_v1/manifest.json:62)–68 |

**VERIFIED：** 这两个统计的生成逻辑不是逐行复制：`aggregate_eval_row` 先读取 query 组的 `positives` 和 `negatives`，从负例中排除所有正例 ID，再写出并集；见 [`solution/src/prepare_early_stop_split.py`](/home/seraphic/xir/solution/src/prepare_early_stop_split.py:77)–88。读取每个 source row 时，代码把 `pos_id`/`pos` 和 `neg_id`/`neg` 合并到同一 normalized-query 组；见 157–164 行；随后用 `len(eval_row["neg_id"])` 和正负之和生成 summary；见 174–205 行。

**结论：** raw pairs 的 `407` 是“一个发布 pair 行”的最大 negative 列表；dev manifest 的 `312` 是“一个 normalized query 聚合 eval 行”的最大 negative 并集。二者都可以出现几百，但统计单位不同。

### 3.3 其他容易混淆的统计

**VERIFIED：** [`archive/evidence/early_stopping_research_20260720/data_audit.json`](/home/seraphic/xir/archive/evidence/early_stopping_research_20260720/data_audit.json:10)–54 中的 `legacy_dev500.negative_count` 是旧 dev500 子集，`min=5, median=9, max=229`；同文件 56–98 的 literal last-500 source rows 是尾部子集，`min=6, median=9, max=168`。它们不能替代全量 96,504 行统计。

**VERIFIED：** [`data_report.md`](/home/seraphic/xir/solution/outputs/data_report.md:20)–27 中还列有 query 字符长度、正样本文本字符数、负样本文本字符总字符数、reasoning_len 和 reweight_rate；如果问题中的 min/max/median 指的是这些字段，必须按字段名区分，不能把“负样本数”的 407 与“负样本文本字符总数”的 1,028,372 混用。

## 4. provenance 脚本能证明什么，不能证明什么

**VERIFIED：** [`solution/src/build_trajectory_provenance.py`](/home/seraphic/xir/solution/src/build_trajectory_provenance.py:183)–231 从 trajectory archive 的每个 Browse event 提取 `search_idx`、`browse_idx_in_search`、`retrieved_rank`、`search_result_count`、`searched_doc_ids_so_far` 和 `trajectory_source_doc_ids`。这里的 `search_result_count` 是当前 Search 返回列表长度；它与 pairs 的 `negative_count` 是不同字段。

**VERIFIED：** 同脚本 254–271 的 `describe` 对事件字段计算 count/min/median/mean/p90/p99/max；274–316 的 `load_pair_minimal` 对 pairs 读取 `neg_id` 并把 `negative_count` 定义为 `len(negative_ids)`。因此如果要回答“单次 Search top-k 的 min/max/median”，应读取 provenance event 的 `search_result_count`；如果要回答“pairs negative 的 min/max/median”，应读取 pair row 的 `negative_count`。

**VERIFIED：** provenance 映射在 344–421 行以 `query + pos_id + reasoning_len` 找候选事件，并要求 pair 的 negative ID 集合是 event 的 `trajectory_source_doc_ids` 子集；映射摘要在 423–431 行区分 stable/ambiguous/mismatch 和 negative ID traceability。这支持“pairs negative 来自 trajectory 可追溯来源”的核验，但 stable provenance 表示来源映射稳定，不表示每个 negative 已被人工判定为语义 irrelevance。

**VERIFIED（项目记录）：** M01 全量 provenance 记录为 `96,366 stable / 138 ambiguous / 0 mismatch / 96,504 negative IDs traceable`；见 [`control/CCIR/REGISTRY.md`](/home/seraphic/xir/control/CCIR/REGISTRY.md:261)–276。该记录是已完成的项目验收证据，不应被解释成 `search_result_count` 的统计表。

**UNKNOWN：** 本地保存了 provenance 构建脚本、pairs 统计和 manifest，但没有保存该全量 trajectory archive 或全量 provenance JSONL 的可读副本；所以本轮没有声称、也没有重新计算事件级 `search_result_count` 的 min/median/max。若后续从服务器取证，应保持只读，并把 `search_result_count`、`negative_count`、query-group `negatives_per_query` 分成三列报告。

## 5. 已确认、推断与建议表述

| 问题 | 已确认 | 推断/未知边界 |
|---|---|---|
| trajectory negative pool 是否就是 pairs negative | `data_builder.py` 将构造出的 `neg_texts`/`neg_ids` 写入 pairs 的 `neg`/`neg_id`；本地全量复核两者长度 0 mismatch；provenance 记录 negative IDs 可追溯 | 发布 pairs 无 trajectory ID，不能仅凭一行 pair 恢复完整 Search/Browse 历史或证明每个负例的语义标签 |
| 单次 Search 是否返回几百条 | 论文和方法设置为每次 Search top-10 | 400+ 来自多个 Search 的历史池合并和/或此前 unsatisfied Browse；max=407 不能反推出精确 Search 次数 |
| `min/median/max=5/9/407` 是什么 | 96,504 条 raw pairs 每行 `len(neg)` 的统计 | 不是 event `search_result_count`，也不是 query-group 聚合后的 `312` |
| dev1500 的 `5/9/312` 是什么 | normalized-query 组内 union 后的 `negatives_per_query` | 不是 raw pair row 的分布 |

建议对外或答辩使用的准确表述：**“论文的每次 Search 返回 top-10 候选；官方 builder 把 Search→Browse 轨迹中未 Browse/未满足的候选按当前实现组合到每个 Browse-derived pair 的 `neg`/`neg_id`。由于 builder 会保留满意 Browse 前的搜索历史，一个 pair 的负例数可以跨多个 Search 累积；发布 pairs 的全量每行负例数为 5–407，中位数 9，407 不是单次 Search 的 top-k。”**

## 6. 来源清单

- 官方论文 HTML：[arXiv:2604.04949](https://arxiv.org/html/2604.04949)，重点为 §4.1.2、§5.1.1、§5.1.2、§5.2.2。
- 官方源码：[Yuqi-Zhou/LRAT `src/data_builder.py`](https://github.com/Yuqi-Zhou/LRAT/blob/main/src/data_builder.py)；本地对应文件为 [`LRAT/src/data_builder.py`](/home/seraphic/xir/LRAT/src/data_builder.py)。
- 官方数据构造说明：[training_data_construction.md](https://github.com/Yuqi-Zhou/LRAT/blob/main/docs/training_data_construction.md)。
- 本地原始 pairs：[`data/raw/LRAT-Train/LRAT-training-pairs.jsonl`](/home/seraphic/xir/data/raw/LRAT-Train/LRAT-training-pairs.jsonl)。只做流式聚合，未输出原始 query、文档文本或凭据。
- 本地统计产物：[`solution/outputs/data_report.json`](/home/seraphic/xir/solution/outputs/data_report.json)、[`solution/outputs/data_report.md`](/home/seraphic/xir/solution/outputs/data_report.md)。
- 本地可复核统计/分组脚本：[`solution/src/audit_early_stop_split.py`](/home/seraphic/xir/solution/src/audit_early_stop_split.py)、[`solution/src/prepare_early_stop_split.py`](/home/seraphic/xir/solution/src/prepare_early_stop_split.py)、[`LRAT/src/training_data_stats.py`](/home/seraphic/xir/LRAT/src/training_data_stats.py)。
- 本地 manifest/provenance：[`data/processed/early_stop_v1/manifest.json`](/home/seraphic/xir/data/processed/early_stop_v1/manifest.json)、[`solution/src/build_trajectory_provenance.py`](/home/seraphic/xir/solution/src/build_trajectory_provenance.py)、[`control/CCIR/REGISTRY.md`](/home/seraphic/xir/control/CCIR/REGISTRY.md)。
