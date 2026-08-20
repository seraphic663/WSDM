# CCIR / LRAT 答辩 v2.0 一手来源核查

日期：2026-08-09（Asia/Shanghai）

范围：只核查 P4、P6、P9、P10 所涉及的 LRAT 监督信号、训练 loss、比赛 Recall、B 榜 Agent/分数及复现证据；不改 PPT，不把内部记录冒充主办方公开记录。

状态标记：**VERIFIED** = 可由本轮读取的一手来源直接支持；**INFERENCE** = 基于已核事实的谨慎推断；**UNKNOWN** = 现有一手材料不足或彼此未完全对齐。

## 结论先行

1. P6 的 `[1,0,0,0,0,0]` 应删除。代码级准确说法是：**“候选组内正例偏移 = 0（CrossEntropy 使用整数 target；非 one-hot）”**。M10 开启跨卡 negatives，两个 query 的全局 target 实际为 `[0,6]`，所以不宜无条件写“全局 target index = 0”。
2. P4 应补充 trajectory→pair 的监督链，但要区分**论文方法**、**当前公开 builder**与**实际发布 pair**。最稳妥的答辩表述是：**“官方 LRAT：Browse 文档先成为候选正例；紧随其后的 reasoning 用于相关性筛选，未 Browse 的检索候选形成轨迹负例池；reasoning length 映射为 `reweight_rate`。本队不重算发布 pair 的标签或权重，只做来源核验并使用官方 pair。”**
3. P9 的 Recall 不是静态 Recall@K。更准确的标签是：**“Evidence Recall：Agent 全程检索结果对标注证据文档的覆盖率（按 query 宏平均）”**。比赛页简称“检索器召回率”不算事实错误，但会让 IR 评委误解为固定 K 的离线检索指标。
4. P9 的“实验归因”应改为“结果分解”或“对比关系”。`34.0→40.6` 是外部 baseline 对比，不能把 `+6.6` 因果归给单一方法改动；`39.5→40.6` 才更接近同一 query-disjoint 条件下的 LR 对比。
5. B 榜本地平台截图直接显示记录 `998912` 的三个结果为 `67.42 / 60.96 / 40.16`、平均 `56.18`，且中间 Agent 显示为 `Qwen3.5-35B-A3B`；但当前主办方公开的 B 榜 inference README 列出的 Qwen 后端是 `Qwen3.5-122B-A10B-FP8`。二者对应对象可能不同，**不能用公开默认配置覆盖平台具体计分记录，也不能在没有原始平台导出/API 的情况下宣称冲突已经解决**。
6. P10 可以保留“复现约束 + B 榜跨 Agent”双栏，但左栏措辞需按证据强度收紧：GitHub 固定 commit、代码内 manifest gate、canonical dry-run、CPU load/validator、HF commit/LFS SHA read-back、ZIP 逐文件 manifest/CRC 是可分开的验收项；“HF 重新下载并重算哈希”和“GitHub/ZIP 按同一清单回读”比现有公开证据更强，不宜笼统合并。

## 1. 官方 pair 如何从 Agent trajectory 形成

### 1.1 论文方法

**VERIFIED**：LRAT 论文把 `Search→Browse` 后被 Browse 的文档称为 *naive positive*；同一 retrieved candidate set 中未被 Browse 的其余文档是 *naive negatives*。随后取 immediate post-Browse reasoning，由 Qwen3-30B-A3B-Thinking-2507 判断 Relevant/Irrelevant，以过滤 browsed-but-unhelpful 文档。[论文 §5.1.1](https://arxiv.org/html/2604.04949#S5.SS1.SSS1)，[论文 §5.1.2](https://arxiv.org/html/2604.04949#S5.SS1.SSS2)

**VERIFIED**：reasoning token length `l` 经饱和函数映射并按全数据 raw mean 归一化：`w=(1/μ_raw)·(1-exp(-ln2·l/β))`，其中 `β` 为 reasoning length 中位数；weighted InfoNCE 的 negatives 还包含 in-batch negatives。[论文 §5.2.1](https://arxiv.org/html/2604.04949#S5.SS2.SSS1)，[论文 §5.2.2](https://arxiv.org/html/2604.04949#S5.SS2.SSS2)

### 1.2 公开 builder 与发布 pair 的必要区别

**VERIFIED**：当前官方 GitHub `src/data_builder.py` 比论文简述更复杂：它对 reasoning 产生 `satisfied` 判断，但代码会为每个被 Browse 的文档写出 `pos`；negative pool 可组合本轮/历史 `satisfied=false` 的 Browse 文档与搜索历史中的 unbrowsed 文档，而不只是单次 Search 的其余候选；同一文件实现 Eq.(3) 权重。[官方 builder@01234b5](https://github.com/Yuqi-Zhou/LRAT/blob/01234b50f14b2582581723e5e4c815cf020ebc2a/src/data_builder.py#L325-L424)

**VERIFIED（项目发布数据审计）**：锁定的 96,504 条官方 pair 全部为 `satisfied=true`，每行 negatives 为 5–407 条；这证明发布数据已筛选，但也说明不能把实际 M10 的每个 `neg` 列表简单说成“某一次 Search 的固定 top-K 其余项”。[data_report.md](/home/seraphic/xir/solution/outputs/data_report.md:20)

**建议 P4 一行文案**：`官方 LRAT：Browse 文档先成为候选正例；post-Browse reasoning 用于相关性筛选，未 Browse 的检索候选形成轨迹负例池；reasoning length 映射为 reweight_rate。本队不重算发布 pair 的标签或权重，只做来源核验并使用官方 pair。`

## 2. P6：loss 到底是 target index、one-hot 还是其他

**VERIFIED**：M10 的 vendored dataset 先 append 一个 positive，再随机采 `train_group_size-1` 个 negatives；group size=6 时就是 1+5，positive 位于组内偏移 0。[AbsDataset.py@31949f6](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/vendor/FlagEmbedding/FlagEmbedding/abc/finetune/embedder/AbsDataset.py#L105-L136)

**VERIFIED**：训练使用 PyTorch `CrossEntropyLoss`；in-batch target 为 `idxs * group_size`，跨设备 gather 后仍是 `cross_idxs * group_size`，并把逐样本 CE 按 `reweight_rate` 加权。没有显式 one-hot target，也不是 BCE/multi-label loss。[modeling.py：CE 与加权](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/vendor/FlagEmbedding/FlagEmbedding/finetune/embedder/decoder_only/base/modeling.py#L188-L210)，[modeling.py：in-batch target](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/vendor/FlagEmbedding/FlagEmbedding/finetune/embedder/decoder_only/base/modeling.py#L224-L252)，[modeling.py：cross-device target](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/vendor/FlagEmbedding/FlagEmbedding/finetune/embedder/decoder_only/base/modeling.py#L316-L339)

**代码级推荐文案**：`候选组内正例偏移 = 0（CrossEntropy 使用整数 target；非 one-hot）`。P6 下方只保留候选位置 `0 1 2 3 4 5` 即可；若要画标签，不应再画 `[1,0,0,0,0,0]`。

## 3. P9：A 榜 Recall 的准确含义

**VERIFIED**：主办方比赛 API 的 `cmptDataDescription` 定义 Recall 为“检索器返回结果对 BC-Plus 标注证据文档的覆盖比例”。[XIR competition 1170 API](https://www.xir.cn/competitionApi/api/competitions/1170?part=daily)

**VERIFIED**：BrowseComp-Plus evaluator 对每个 query 取 Agent 运行过程中累计的 `retrieved_docids` 集合，计算 `|retrieved ∩ evidence_qrels| / |evidence_qrels|`，再对有证据标注的 queries 取平均；LRAT 论文也称其为 execution-time Evidence Recall。[evaluate_run.py@0469490：per-query recall](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L509-L515)，[macro average](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/scripts_evaluation/evaluate_run.py#L680-L699)，[论文指标定义](https://arxiv.org/html/2604.04949#S6.SS1.SSS3)

**VERIFIED**：这不是静态 Recall@K；benchmark README 把 retrieval-only 的固定 K 指标另列为 Recall@5/100/1000。[BrowseComp-Plus README@0469490](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/README.md#L93-L149)

**UNKNOWN**：主办方 A runner 会拉取 BrowseComp-Plus `main`，公开脚本没有固定历史 commit；因此不能仅凭公开仓库证明 2026 年 7 月线上评测所用 evaluator 的精确 commit。但“执行轨迹级 evidence coverage，而非静态 Recall@K”在比赛页、论文和当前 evaluator 三者间一致。

**建议 P9 指标注释**：`Recall：Agent 执行期间标注证据文档覆盖率`；空间允许时写全：`Evidence Recall：逐 query 证据覆盖后宏平均`。

## 4. B 榜三个 Agent 与 56.18

**VERIFIED（平台截图内容）**：本地保存的“最新 B 榜榜单”截图显示 DefaultGroup、记录 `998912`、平均 Total `56.18`，分项为 `DeepSeek-V4-Flash-0731 67.42 / Qwen3.5-35B-A3B 60.96 / gpt-oss-120b 40.16`。[平台截图](/home/seraphic/xir/report/CCIR答辩/temp/codex-clipboard-74e9d757-d752-4403-bd19-b5a631bcde5b.png)

**VERIFIED**：算术平均 `(67.42+60.96+40.16)/3 = 56.18`；最高与最低绝对差为 `27.26`。

**VERIFIED（公开默认 inference 配置）**：主办方当前公开 B 榜 inference README 列出 `DeepSeek-V4-Flash-0731 / GPT-OSS-120B / Qwen3.5-122B-A10B-FP8`，并明确该目录记录 inference implementation、比赛页才是规则/排名事实源。[官方 B inference README@01234b5](https://github.com/Yuqi-Zhou/LRAT/blob/01234b50f14b2582581723e5e4c815cf020ebc2a/xir_b_leaderboard_inference/README.md#L1-L13)

**UNKNOWN**：公开 API 当前 `isShowRank=0` 且日榜为空；项目也未归档原始平台导出、逐 Agent 评测明细或可重放的计分 API。因此 35B 截图与 122B 公开 inference 配置的关系尚未由一手材料解释。对“记录 998912 实际计分 Agent”的表述，结果截图比通用 runner README 更直接；但答辩前最好补存平台原始记录/官方通知，不要静默把 35B 改成 122B，也不要反向声称公开 runner 就是该记录唯一计分配置。[项目证据边界](/home/seraphic/xir/control/CCIR/B_LEADERBOARD.md:5)

**INFERENCE（可用的谨慎分析）**：三种 Agent 均产出端到端 Total，说明同一 retriever 至少能在三个 Agent 系统中完成评测；27.26 分的绝对差说明端到端表现对 Agent 的 query/tool/Browse 行为与推理能力敏感。不能写“各 Agent 均有提升”或“收益不均衡”，因为没有每个 Agent 的 matched baseline。

**建议 P10 分析句**：`三种 Agent 上均完成端到端评测，但绝对得分差异明显；这支持跨 Agent 可运行性，也表明端到端表现受 Agent 的 query / Browse 行为与推理能力影响。`其中后一因果机制应在口头表达中明确为合理推断，而不是已做受控归因。

## 5. P10：复现链路可由哪些一手证据支持

### 5.1 当前可公开核验

**VERIFIED**：canonical GitHub commit `31949f6f...37bba` 提供唯一 M10 入口 `code/reproduce_m10.py`，固定 pairs、trajectory archive、corpus、base model、tokenizer 与 train split 身份；依次构建 provenance、完整 corpus 核验、query-disjoint split 和双卡训练命令，并把状态写入 `CANONICAL_REPRODUCTION.json`。[canonical README@31949f6](https://github.com/seraphic663/ccir-lrat-retriever/tree/31949f6f53722f06e91bd2ded6ec7f3a48037bba)，[reproduce_m10.py](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/reproduce_m10.py#L17-L23)，[reproduction chain](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/reproduce_m10.py#L56-L138)，[identity checks and final status](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/reproduce_m10.py#L176-L238)

**VERIFIED**：`preprocess.py` 逐行验 query/positive/negative/reasoning lineage、验证 `reweight_rate` 公式误差不超过 `1e-12`，复制官方 pair 时再次核对相同 SHA，并生成 `full_raw_traceability_verified` manifest；`train.py` 在训练前强制检查该 manifest、query-disjoint split、corpus coverage 与训练数据身份。[preprocess.py](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/preprocess.py#L255-L353)，[manifest output](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/preprocess.py#L398-L442)，[train.py gates](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/code/train.py#L134-L184)

**VERIFIED（本轮现场）**：在临时 checkout 的精确 commit `31949f6...37bba` 上运行 `python3 -m unittest discover -s tests -v`，4/4 通过；`python3 -m compileall -q code tests` 通过。测试覆盖训练命令合同、全部输入 identity pins、deterministic/query-disjoint split 及“公开 repo 不跟踪训练数据/权重/ZIP”。[test_reproduction.py](https://github.com/seraphic663/ccir-lrat-retriever/blob/31949f6f53722f06e91bd2ded6ec7f3a48037bba/tests/test_reproduction.py#L24-L114)

**VERIFIED（本轮 live read-back）**：HF revision API 返回 public repo、commit `9cdfdba5...016bb`、`model.safetensors` 2,383,139,480 bytes、LFS SHA-256 `cea87cca...a9984`。[HF revision API](https://huggingface.co/api/models/Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729/revision/9cdfdba51359ec65069a748cf8c00c55477016bb?blobs=true)

### 5.2 项目记录已验收，但本轮未重新执行完整重任务

**VERIFIED（项目原始记录）**：2026-07-31 canonical real-input dry-run 记录为 `dry_run_verified`；独立完整训练的 dev1500 四项与提交模型绝对差均不超过 `0.002`；匿名 HF/GitHub identity read-back、45-file ZIP 的两套 validator、逐文件 manifest 与 `unzip -t` 均记录通过。[LOG.md](/home/seraphic/xir/control/CCIR/LOG.md:409)

**UNKNOWN / 措辞边界**：本轮没有重新下载 2.38GB HF 权重后在本地重算文件 SHA，也没有重新取得服务器 ZIP 做全量 CRC；本轮确认的是 HF API commit/root file/LFS SHA 与公开 GitHub 代码。P10 若要严格描述当前可即时复核的事实，建议把“HF 重新下载并重算哈希”改为 `HF fresh read-back：commit / 根目录 / LFS SHA`，把“GitHub / ZIP 按同一清单回读”拆成 `GitHub canonical commit 固定；ZIP 逐文件 manifest + validator + CRC 回读`。

## 6. 对 v2.0 的最终修改优先级

1. **必须改 P6**：one-hot → `候选组内正例偏移 = 0（CrossEntropy 整数 target；非 one-hot）`。
2. **必须补 P4**：补 trajectory supervision 一行，但使用“候选正例 / 轨迹负例池 / reasoning 筛选和加权”，不要把全部实际 negatives 绝对化为“同一次 Search 的其余项”。
3. **必须改 P9**：`实验归因` → `结果分解`；Recall 注释改为 execution-time Evidence Recall。
4. **必须核 P10 Agent 名称证据**：保留记录 998912 截图中的 35B 只能以该具体截图为来源；同时在底稿中记录当前公开 runner 为 122B，不得互相替代。
5. **建议改 P10 复现措辞**：将 GitHub、manifest/dry-run、CPU load、HF API/LFS read-back、ZIP manifest/CRC 分成可独立验收的事实，避免“同一清单”和“重新下载”这种超出当前公开证据的概括。

