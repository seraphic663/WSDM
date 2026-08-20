# CCIR 项目日志

> 由 LOG.md 与 LOG.md 合并（2026-08-04）。

## 目录
- [LOG.md](#history)
- [LOG.md](#project_log)

## HISTORY.md

# XIR 历史摘要

这是实验历史的导航页。详细报告在 [`../../archive/experiments.md`](../../archive/experiments.md)，旧实验说明已移至 [XIR archive/history.md](/home/seraphic/xir/archive/history.md)，完整逐日日志在 [`LOG.md`](LOG.md)。

## 模型路线

| 阶段 | 版本 | 结论 |
|---|---|---|
| 基础/全量 pairs | M00–M02 | M01 的 1 epoch 优于累计 2 epoch 的 M02 |
| 继续累计 | M06 | 第 3 epoch 有回升，但仍低于 M01 |
| 插值诊断 | M03–M05 | 没有形成提交候选 |
| query-disjoint early stop | M07 | 召回提高，但综合 A 榜低于 M01 |
| weighting/multi-positive | M08–M09 | gate 或方向性门槛未通过 |
| 学习率路线 | M10 | `2e-6` 完整训练成为当前 A 榜最佳 |
| search_idx weighting | M12 | Recall 提高，但综合分没有超过 M01 |
| 其他研究设计 | M11、M13 | 尚未形成正式权重 |

## 详细报告索引

- M09/M10 后续实验：[`M09_M10_FOLLOWUP_20260728.md`](../../archive/experiments.md)
- trajectory 质量：[`TRAJECTORY_QUALITY_V1_EXPERIMENT_20260724.md`](../../archive/experiments.md)
- offline flywheel：[`OFFLINE_DATA_FLYWHEEL_V1_EXPERIMENT_20260728.md`](../../archive/experiments.md)
- paper flywheel：历史研究已从 WSDM 副本移出，见 [XIR archive/research.md](/home/seraphic/xir/archive/research.md)
- 存储治理：[`SERVER_STORAGE_CLEANUP_20260728.md`](../../archive/experiments.md)
- M01/M10 B 榜材料：已从 WSDM 副本移出，完整历史见 [XIR control/CCIR/B_LEADERBOARD.md](/home/seraphic/xir/control/CCIR/B_LEADERBOARD.md)


---

## PROJECT_LOG.md

# CCIR / LRAT 项目更新日志

本日志从 Zhao Ziming 在服务器 `/root/data/LRAT` 的第一个项目提交开始记录。服务器开发分支为 `ccir/dev`；以下提交均为本地提交，未 push 到官方 `origin`。

## 2026-07-15：环境与项目工作区

### `4b81265 chore: add uv project metadata`

- 根据仓库已有 `uv.lock` 补充缺失的 `pyproject.toml`。
- 使用 `/root/.local/bin/uv` 建立 `/root/data/LRAT/.venv`，不修改 Conda 环境。
- 核验 Python 3.10.16、torch 2.7.0+cu126、transformers 4.53.2、vLLM 0.9.0.1、DeepSpeed 0.17.2。
- 两张 NVIDIA A40 均通过 CUDA 可用性与计算检查。

### `8db9c64 chore(ccir): initialize workspace and data inspection tools`

- 建立 `ccir/` 文档、报告、数据、模型和输出目录约定。
- 增加比赛说明、论文解读、服务器环境说明和数据报告。
- 增加 `solution/src/inspect_data.py` 与 `solution/src/prepare_data.py`。
- 完整 LRAT training-pairs 共 96,504 条；生成 query-disjoint 的 5,000 条 train 与 500 条 dev。
- 数据、模型、checkpoint、输出、缓存和虚拟环境加入忽略规则，不进入 Git。

## 2026-07-16：Qwen3 原始模型 baseline

### `e2f8d26 feat(ccir): add adaptive Qwen3 base baseline runner`

- 增加 `solution/src/run_qwen3_base_baseline.py` 和服务器适配说明。
- 明确 retriever 必须是原始 `Qwen3-Embedding-0.6B`，不使用 Qwen3.5 retriever。
- 包装器根据两张 A40 的实时状态做资源预检，默认不下载、不安装、不执行。
- 该端到端 A 榜包装器仍依赖完整 BrowseComp-Plus corpus 和比赛规定的 Agent/Judge；当前没有把它当作已完成的正式 A 榜结果。

### `83ace4c eval(ccir): record Qwen3 dev baseline`

- 新增 `solution/src/evaluate_qwen3_pairs.py`，补齐当前 `query / pos / neg` 数据格式的候选集评测入口。
- 原始 `Qwen3-Embedding-0.6B` 已部署到 `ccir/models/Qwen3-Embedding-0.6B/`，未训练、未修改权重。
- 在服务器 GPU 1 上评完完整 500 条 dev，共编码 9,942 个候选段落。
- 结果：Recall@1 `0.494`、Recall@5 `0.804`、Recall@10 `0.944`、MRR `0.6343869118`。
- 总耗时 `165.4904` 秒，峰值 CUDA 显存约 `3.12 GiB`。
- 结果写入 `ccir/outputs/eval/qwen3_base_dev500.json`，实验报告写入 `ccir/reports/QWEN3_BASELINE_DEV500_20260716.md`。
- 上述指标是每条样本自带候选集上的训练前对照，不是 BrowseComp-Plus 全库或 Agent 端到端指标。

### `8f4bc45 docs(ccir): log baseline and draft smoke training`

- 新增本日志，把从首个项目提交至完整 dev baseline 的工作串联起来。
- 新增 `ccir/docs/QWEN3_TRAINING_SMOKE_DRAFT.md`，记录下一步训练模拟的已具备条件、缺口、参数、命令、验收和全量耗时外推方法。
- 该提交只更新文档，没有上传训练数据、修改依赖或启动训练。

## 2026-07-16：Qwen3 单卡训练 smoke

- 将本地 5,000 条、202,698,916 bytes 的 train 上传到服务器，行数、大小和 SHA-256 一致。
- 生成并检查固定 64 条子集；5,000/64 条数据均无非法 JSON、空字段、重复 query 或正负 ID 重叠。
- 新增 `solution/scripts/run_qwen3_smoke_train.sh`，复用仓库内 FlagEmbedding decoder-only 训练入口。
- 单卡完成 10-step 链路验证：runtime 39.83 秒，无 OOM/NaN，checkpoint 可在新进程加载。
- 单卡完成 5,000 条数据上的 100-step 计时短跑：runtime 291.57 秒，稳态约 2.59 秒/step。
- 100-step 模型在固定 500 条 dev 上达到 Recall@1/5/10 `0.522 / 0.826 / 0.960`、MRR `0.6539454`，均高于原始模型。
- 估算当前单卡配置下 5,000 条约 1.03 小时/epoch，96,504 条约 19.95 小时/epoch。
- 详细记录见 `ccir/reports/QWEN3_TRAINING_SMOKE_RESULTS_20260716.md`。
- 本轮未上传 3.88 GB 完整训练数据，未启动完整 epoch 或端到端 A 榜。

## 当前已知工作树例外

服务器工作树另有未提交修改：

```text
M xir_a_leaderboard_eval/run_browsecomp_plus_eval.sh
```

它来自此前停止的 A 榜环境适配尝试，与候选集 baseline 和本次训练草案无关；本次不修改、不回退、不暂存。

## 2026-07-16：双卡训练与可恢复执行器

### `48cd45c fix(ccir): support resumable weighted dual-GPU training`

- 修复 decoder-only 跨卡负例训练没有同步 `reweight_rate` 的问题，使全局 query、passage 与权重保持相同的 rank 顺序。
- 对权重数量、有限性、非负性和正权重和增加显式校验。
- 新增双卡训练入口、完整 checkpoint 选择器及对应单元/两进程分布式测试。
- 双卡入口固定保存运行配置和数据/模型 SHA-256；配置变化时拒绝误续训。
- 增加 `RUNNING / FAILED / COMPLETED` 标记、GPU 空闲预检、运行重叠保护和自动选择最新完整 checkpoint。

### `a9e0382 fix(ccir): make dual-GPU checkpoints resumable`

- 实际中断恢复演练发现 DDP 与 reentrant gradient checkpoint 不兼容，改为 `use_reentrant=false`。
- 实际恢复又发现 FlagEmbedding 包装层不能直接加载其自身保存的无前缀模型权重，增加兼容路由和回归测试。
- 在 `qwen3_dual_resume10_v2` 中主动于完整 `checkpoint-4` 后中断；修复后自动从 step 5 继续，最终完成 step 10，并保留 checkpoint 6/8/10。

### 双卡 100-step 验证

- run：`qwen3_dual_smoke100`；两张 A40；global effective query batch 8；跨卡加权负例。
- 训练 runtime `343.3643` 秒，平均 `3.43` 秒/step，train loss `1.3920889`，无 OOM/NaN。
- dev500：Recall@1/5/10 `0.514 / 0.838 / 0.972`，MRR `0.6560629`。
- 完整数据固定为 96,504 行、3,883,089,616 bytes、SHA-256 `dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9`。
- 两卡完整一轮恰为 12,063 optimizer steps；按实测与保存开销预留约 13–14 小时运行窗口。

## 2026-07-16：双卡完整一轮与最终评测

- 上传并逐项复核完整 96,504 条 training-pairs：3,883,089,616 bytes，SHA-256 `dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9`。
- 全量数据检查为 0 条非法 JSON、0 条正负 ID 重叠；原始数据的 1 条空 query 经 instruction 格式化后仍为非空模型输入，按“全内容”要求原样保留。
- `qwen3_dual_full_epoch1_20260716` 在两张 A40 上一次完成 12,063 steps / 1 epoch，未触发自动重试。
- runtime `42138.7996` 秒（11:42:18.8），train loss `0.8477314882`，无 OOM/NaN/Traceback。
- 最终保留 checkpoint 11500/12000/12063；最新有效恢复点为 checkpoint-12063，最终权重 SHA-256 `b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501`。
- 最终模型在相同 dev500 上达到 Recall@1/5/10 `0.672 / 0.934 / 0.984`、MRR `0.7803662`；原始模型为 `0.494 / 0.804 / 0.944`、MRR `0.6343869`。
- 结果写入 `ccir/outputs/eval/qwen3_dual_full_epoch1_dev500.json`，SHA-256 `1201b5a8238abf32cb16b56f36ab82902065cd85af47768bc8b2982cf9621ab0`；详细报告见 `ccir/reports/QWEN3_DUAL_FULL_TRAINING_20260716.md`。
- 这些指标是固定候选集检索指标，不是 BrowseComp-Plus 全库或 Agent 端到端 A 榜成绩；后续审计进一步确认该500条与完整训练数据重叠。

## 2026-07-18：多 Session 共享上下文体系

- 根 `AGENTS.md` 重构为长期协作规则、事实优先级和五个 `sol_*` session 的职责路由；动态实验状态不再持续堆入总纲。
- 新增 `STATUS.md`、`REGISTRY.md`、`REGISTRY.md` 和 `SESSION_STARTERS.md`，分别记录实时快照、模型SHA、每日提交和新session启动提示。
- 新增 coord、train、eval、submit、audit 五份工作流，明确Git/GPU/HF所有权、开始检查、操作边界和交接内容。
- 记录1epoch正式HF提交、2epoch与插值候选、查询冲突清洗结果，以及group10/batch2 smoke加载失败。
- 修正评测口径：500条候选集的所有query均在完整训练数据中，只能作为训练重叠诊断；这不是比赛隐藏测试集泄漏，正式泛化判断以A榜为准。
- 用户已反馈1epoch A榜成绩相当不错，但具体数值尚未提供，注册表明确标为待录入，未做推测。

### 跨环境架构与操作手册细化

- 新增 `ccir/README.md`，统一说明 Codex 客户端、本地 WSL、训练服务器、Hugging Face 和比赛平台五个层面的拓扑、路径映射、认证边界、Git边界及数据—模型—提交生命周期。
- 把五份 workflow 从职责摘要扩展为可执行手册，补充开始检查、命令入口、资源所有权、验收条件、异常恢复、证据要求和交接模板。
- 扩展 `SESSION_STARTERS.md`：每个无聊天上下文的新 session 都会读取架构文件、做职责范围内的实时只读核查，并明确本地/服务器/HF/比赛平台的权限边界。
- 核查时本地 HF CLI 为 `1.23.0`，登录账号为 `Seraphic663`；认证只留在本地用户缓存，不向服务器或Git同步。服务器为 Python 3.10.16、torch 2.7.0+cu126、transformers 4.53.2、vLLM 0.9.0.1、DeepSpeed 0.17.2 和 2×A40。
- 所有共享文档不写SSH密码、HF token或比赛平台凭据；根 `AGENTS.md` 保持本地私有，不同步到服务器。
- 本轮实测当前WSL尚未实际配置 `lrat-ebcloud` SSH别名；文档已区分推荐别名与 `ssh -p 34988 root@ssh-cn-huabei1.ebcloud.com` 直连回退，未擅自修改用户SSH配置。
- 本轮同步又确认服务器没有 `rsync`；操作手册改用精确 `scp`/空 staging 回退并保留文件清单与SHA验收，不为此擅自安装服务器依赖。

## 2026-07-18：登记 M01 正式 A榜成绩

- 用户提供平台榜单记录：`DefaultGroup` 的 M01 1epoch 为 `Total 40.1 / Recall 44.8 / Success 21.0 / AvgSteps 15.4 / StepScore 69.1`，提交时间 `2026-07-17 10:40:36 UTC`。
- 同表 `BaselineLRAT` 为 `34.0 / 36.1 / 15.5 / 16.8 / 66.5`；M01 总分绝对提升 `6.1`、相对提升约 `17.9%`，且Recall、Success和步骤效率均改善。
- 平台 Total 与比赛公式计算的 `40.14` 四舍五入一致。M01 的 `40.1` 正式取代“成绩较好、数值待录入”的占位描述，成为后续候选的A榜基准。

## 2026-07-18：累计 3epoch、M06 冻结与隔离策略套件

- `qwen3_dual_epoch3_from_epoch2_pathfix_20260718` 从 M02 精确权重继续训练完整 1 epoch，一次完成 12,063 steps；runtime `39,578.2804` 秒、loss `0.4103184`，最终权重 SHA-256 为 `996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5`。
- 首个独立 RUN_ID 的失败根因为 FlagEmbedding 用大小写敏感路径识别 Qwen，导致向 `Qwen3Model` 传入不兼容的 `use_flash_attention_2`。证据保留；恢复只使用指向同一 M02 源目录的含大写 `Qwen` 符号链接和新 RUN_ID，没有修改核心训练代码。
- 自动 dev500 训练重叠诊断为 R@1/5/10 `0.772/0.952/0.992`、MRR `0.8529487`。500 个 query 全部参与过训练，该结果不是 held-out 或泛化成绩。
- M06 冻结到 `ccir/models/Qwen3-Embedding-0.6B-LRAT-3epoch-20260718/`。共享文件系统最初拒绝 `renameat2(RENAME_NOREPLACE)`；安全恢复用原子空目录预留和本任务 staging 改名完成，不覆盖目标。源/目标逐文件 SHA 一致，CPU 新进程加载成功，目录 `0555`、文件 `0444`。
- `strategy_suite_20260718` 完成 A–I 的隔离数据/CPU/GPU 模拟：7 个双卡 10-step 训练和 4 个小规模评测全部返回 0，无 OOM/NCCL/Traceback。A/B/C/E/F/G/H/I 为 PASS；D 因官方离线 corpus 不存在而为 PARTIAL。
- cleaned 对照 10-step loss 较 raw 低约 `19.5%`，值得更长受控对照；group10 较 group6 慢约 `31.3%` 且每卡峰值多约 `610 MiB`；学习率短测不足选优；两点/三点尾部平均可加载，但小样本诊断不能替代独立评测或 A榜。
- 新增隔离数据准备、平均、冻结、GPU 套件、CPU/最终验证脚本、固定配置、单元测试、复现手册和完整报告。没有自动启动新的完整 epoch，没有 HF 上传、比赛 JSONL、平台提交或 `push origin`。

## 2026-07-18：M06 3epoch Hugging Face 与比赛提交准备

- 从服务器只读冻结目录精确同步 8 个推理文件到本地 `versions/M06/`，未复制 manifest、checkpoint、optimizer、训练状态、数据或凭据；全部文件逐项大小和 SHA-256 与服务器一致。
- 使用本地 WSL 的持久 HF CLI 登录创建公开 Model 仓库 `Seraphic663/lrat-qwen3-0.6b-3epoch-20260718`，没有将 token 复制到训练服务器。
- HF commit 为 `b51ef2b1fb8a1bca671b551b030aa1b7f47c59f3`。匿名 API 回读确认仓库公开，远端 `model.safetensors` 为 2,383,139,480 bytes，LFS SHA-256 为 `996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5`。
- 生成并解析验证一行比赛文件 `versions/M06/submission_3epoch_20260718.jsonl`，SHA-256 为 `f70638affe6657935d0fdba647808db6c108877ac6a56fa699728597350154b6`。比赛平台没有可调用 CLI，JSONL 尚未在网页上传，因此没有消耗当日提交额度。

## 2026-07-19：M06 3epoch A榜结果与路线调整

- 用户提供 M06 平台记录 `998655`：`Total 38.7 / Recall 43.4 / Success 19.0 / AvgSteps 15.8`，提交时间 `2026-07-18 08:31:42 UTC`。
- M06 相对 BaselineLRAT 仍为正向：`Total +4.7 / Recall +7.3 / Success +3.5 / AvgSteps -1.0`；相对 M01 则退化：`Total -1.4 / Recall -1.4 / Success -2.0 / AvgSteps +0.4`。Success 的加权下降约 `-0.8` 分，是展示值分解中的最大项。
- 内部训练重叠 dev500 从 M00 到 M01/M02/M06 单调上升，但 A榜从 M01 的 `40.1` 降至 M06 的 `38.7`，再次证明该诊断不能选择泛化最好的 epoch。M01 保持正式最佳参考；M02 仍未上传、未提交。
- 路线调整为：先用 M02 的一次 A榜提交确定 epoch 曲线；随后优先做假负例清洗的更长等步数对照和零训练成本的平均/插值候选；若继续优化训练，只做从 M01 出发的低学习率部分 epoch，不直接启动完整第4轮。

## 2026-07-19：M02 2epoch Hugging Face 与比赛提交准备

- 从服务器冻结目录精确同步 9 个推理文件到本地 `versions/M02/`；服务器和本地逐文件大小及 SHA-256 一致，权重为 2,383,139,480 bytes、`1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050`。
- 创建公开 HF Model 仓库 `Seraphic663/lrat-qwen3-0.6b-2epoch-20260717`，HF commit 为 `28ab688b5e91d4e6bc61642950ec4bb354a73c23`。匿名 API 回读确认仓库公开，远端权重大小和 LFS SHA-256 与服务器冻结副本一致。
- 生成并解析验证一行比赛文件 `versions/M02/submission_2epoch_20260719.jsonl`，SHA-256 为 `3107b79f45197a1ac9ddac5257943817416d94757964aa9d14ac6cac3e4ded7b`。
- 比赛平台没有项目内可调用的提交 CLI；JSONL 尚未在网页上传，因而当前只完成 HF 发布和本地提交文件准备，未登记虚构的平台成绩。

## 2026-07-19：M02 2epoch A榜结果与epoch路线终止

- 用户提供 M02 正式记录：`Total 38.2 / Recall 42.7 / Success 18.3 / AvgSteps 15.5 / StepScore≈69.0`，提交时间 `2026-07-19 08:59:22 UTC`。StepScore 是按展示 AvgSteps 和比赛公式近似推算，不冒充平台正式显示值。
- M02 相对 M01 为 `Total -1.9 / Recall -2.1 / Success -2.7 / AvgSteps +0.1`；M06 相对 M01 为 `-1.4 / -1.4 / -2.0 / +0.4`。退化从第2轮已经发生，第3轮虽比第2轮回升0.5分，仍未恢复第1轮。
- 平台同时给出 `BaselineLRAT 34.0` 和 `Baseline Qwen3 Embedding 28.5`；三次提交都高于两条基线。正式评测的 Agent model 为 `Qwen3.5-4B`，但提交 retriever 的基础模型始终是 `Qwen3-Embedding-0.6B`。
- 决策调整：停止继续累计epoch。下一步从原始 M00 做 raw/cleaned 的500–1000 step query-disjoint等步数对照；只有cleaned在未参与该短训的验证query上稳定改善，才启动一个从M00训练的cleaned 1epoch候选。

## 2026-07-19：query-disjoint raw/cleaned 五阶段实验

- 在 `05859ea` 实现 gated cleaned-data 流水线，随后由 `d7ce25a` 修正为空 query 保留、normalized-query 两遍隔离、对称排除不足负例 query 组，并把流水线改为 `set -euo pipefail`。纯标准库测试、`py_compile` 和 `bash -n` 均通过。
- 首次实验 `cleaned_ab_from_m00_20260719` 在数据准备阶段发现官方第90,327行为空 query；旧流水线未 fail-fast，误进入缺少训练文件的 retry。该次没有发生 GPU 训练；精确停止相关进程并保留 `FAILED` 证据，未删除失败记录。
- 修复后的 `cleaned_ab_queryholdout_20260719` 生成每臂95,893行的 raw/cleaned 训练文件和507-query诊断集。held-out query来自源数据603行，共10,875个passage；训练/诊断 normalized-query 重叠为0。两臂分别只改变假负例清洗，M00、seed、batch、LR、步数和入口完全一致。
- raw 与 cleaned 均用2×A40一次完成1,000 step，runtime分别为`3,262.5957/3,342.7314`秒，train loss为`1.0910384/1.0895236`。两臂checkpoint-500/1000均含完整恢复状态，final权重SHA分别为`5fc827d6e30314c10fd8733f20d7d74c4301e1512e1d3984ac974ac11d830146`和`ec054876c683145db4410709e1b07afc3709a0c06401fb960d06082b9dfe4374`。
- query-heldout 500步 cleaned 相对 raw：`R@1 +0.000000 / MRR -0.000640 / R@5 +0.003945 / R@10 +0.005917`；1000步为`R@1 +0.007890 / MRR +0.005113 / R@5 -0.001972 / R@10 +0.001972`。
- 10,000次逐query配对bootstrap显示1000步R@1/MRR的95%区间仍跨0，且R@5退化；三组预声明门槛全部失败。闸门报告SHA为`03d59b6959a3cdb77e3d90f9723c86b29c9ba61089534c97a6ec8e9a465217a0`。
- 流水线于`2026-07-19T22:10:39+08:00`写入`STOPPED_BY_GATE`并正常退出；没有启动完整cleaned 1epoch，没有产生M07、HF上传或比赛提交。GPU、HF和比赛所有权均已释放。完整报告见`ccir/reports/QWEN3_CLEANED_AB_EXPERIMENT_20260719.md`。

## 2026-07-20：旧dev审计与无泄漏早停研究

- 重新扫描96,504行、SHA-256 `dd75a3...be1b9` 的完整训练源。旧dev500恰有500条精确源记录匹配；其500个normalized query在完整训练中实际对应605条记录，78个query为多记录组。纠正此前“501条精确匹配”的错误计数。
- 审计字面最后500行：仅449个唯一normalized query，尾部内部重复51行；38个query组、41行在更早源位置已出现，因此不能作为无泄漏holdout。
- 新增 `solution/src/prepare_early_stop_split.py`：用salted SHA-256稳定排序完整normalized-query组，生成dev1500和锁定test500；聚合同query全部正负候选，并屏蔽任何曾为正例的负例ID。共从训练排除2,391条源记录，剩余94,113行，train/dev/test query交集为0。
- 新增 `solution/src/audit_early_stop_split.py` 和3项纯标准库行为测试。切分脚本完整扫描耗时约25秒、峰值RSS约195 MiB；dev/test分别为63,076,167/21,355,780 bytes，SHA-256 `6b9cac...f9e` / `24e477...0d9`。
- 形成便携技术报告 `archive/evidence/early_stopping_research_20260720/report.html`，报告构建和结构验证通过；本机没有Chromium，浏览器交互验证未执行。
- 论文启发调整为：优先扩大真实同时对比batch，而不是继续堆epoch或重复已有reweight。当前双卡每次前向只有2个query，gradient accumulation 4不会扩大batch内负例池；第一对照拟将per-device batch改为2、grad-acc改为2，保持每次optimizer更新仍为8个query，再加入normalized-query防碰撞sampler。
- 新早停契约：所有run从M00和同一94,113行train独立开始，dev1500每1000步评测，MRR `min_delta=0.001`、`patience=2`；锁定test500最终只测一次。M01/M02/M06及旧raw/cleaned短训已见过新holdout中的query，不具备无泄漏选模资格。
- 本轮SSH复用连接已失效，因此上述新增代码、报告和切分尚未同步服务器、未执行服务器pytest、未启动GPU或Git commit；不得把本地研究误记为服务器已部署状态。
- 随后新增 `query_collision_sampler.py` 与隔离入口 `run_query_collision_train.py`，通过现有双卡脚本的 `COLLISION_FREE_QUERIES=1` 显式启用；默认入口和旧run的 `run_config.env` 保持兼容。按Transformers 4.53.2与Accelerate默认batch sharding语义设计为“各rank相同全局顺序，由Accelerate分发连续per-device batch”，避免二次分片。
- sampler在94,113行新train逻辑视图上完成全量CPU审计：global microbatch 4时为23,528个完整batch，调度94,112行、仅丢1行、normalized-query碰撞0。新增3项sampler测试。
- 新增 `select_early_stop_checkpoint.py` 与4项测试；输入只允许dev，文件名含test即失败，使用MRR `min_delta=0.001/patience=2`，并要求M00基线及R@1/R@5/R@10守门通过后才标记可进入locked test。
- 用原507-query heldout的SHA阈值与新split provenance精确交叉：raw/cleaned 1000-step训练已经见过新dev的1487/1500个query和新test的497/500个query，不能用它们在新split上的结果选模。
- XIR `archive/workflows.md` 已补充数据生成、batch1/batch2/collision smoke和checkpoint选择命令；这些命令是待服务器验证的执行合同，不是已运行记录。
- SSH复用连接恢复后，19个明确文件（约527 KiB）先上传至服务器独立staging，逐文件SHA与本地一致。服务器 `.venv` 未安装pytest，因此没有擅自改依赖；改用纯Python逐项执行同一批10个测试函数，全部PASS，`py_compile`、`bash -n`和隔离训练入口导入也通过。
- 本轮只部署代码、测试和文档；没有生成约3.8GB的新train，没有启动GPU训练/评测，没有HF或比赛平台操作，也没有触碰既有leaderboard脚本修改和query-conflict未跟踪报告。

## 2026-07-23：单一主会话接管 Prompt 与实时链路复核

- 按 `NEW_SESSION_PROMPT.md` 完成控制面文件地图、必读文档、early-stop/sampler/训练恢复/评测/冻结源码与测试阅读，并实际连接服务器验证流程；未启动 GPU、未打开 locked test、未冻结、未写服务器、未创建 HF repo 或消耗比赛额度。
- 服务器仍为 `ccir/dev` / `04203abe3b01b7f8d2d7bfbc371d53a4edea0755`，工作树仅有两项已知遗留；两张 A40 空闲，无训练/评测进程，`/root/data` 可用约 364G。
- `earlystop_b1_full_epoch_20260722` 的 `COMPLETED`、11,764/11,764 steps、runtime `38,892.1802s`、loss `0.8489937656`、M00/data SHA 和 dev1500 指标均由实时文件复核。最终 2.38GB 权重现场完整重算约 10.3 秒，SHA 为 `c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b`。
- 训练后的独立 dev 评测已经从最终目录加载 model/tokenizer，embedding dimension 为 1024、加载约 23.78 秒；因此旧表述“尚未独立加载”被修正为“尚未完成冻结流程的 CPU/staging 验收”。
- 复核正式早停 `FINAL_SELECTION.json`：step 6000 触发停止，协议选择 step 4000；step 5000 的 MRR 增量 `0.000503` 未达到 `min_delta=0.001`。旧 1000-step 冻结 manifest、test500、HF 404 和本地旧 JSONL 身份也均核对一致。
- 发现 SSH ControlMaster 的真实断点：`ssh -O check` 可显示存活，但被 timeout/中断的慢命令后新复用命令仍会卡住。已验证 `ssh -O exit`、强制 curve25519 重建和探针恢复；长扫描/哈希改用非复用独立连接。
- 本地环境没有 pytest；改用直接执行同一测试函数和 unittest，early-stop split、sampler、selector、segment control、checkpoint 选择与 freeze 共 17 项通过，相关 Python 编译与 shell 语法检查通过。
- 更新 `AGENTS.md`、`STATUS.md`、`REGISTRY.md`、`REGISTRY.md`、架构/README/三阶段 workflow 和 `NEW_SESSION_PROMPT.md`，消除多会话路由、旧生命周期状态和 SSH 假活处理缺口。M07 仍未分配，外部状态未改变。

## 2026-07-23：M07 冻结、公开 HF 发布与正式评测文件准备

- 用户授权完成 M07 验收、上传 HF 并准备官方评测。实时复核服务器仍为 `ccir/dev` / `04203abe3b01b7f8d2d7bfbc371d53a4edea0755`，工作树仅两项既有遗留，两张 A40 空闲；训练目录存在 `COMPLETED` 且无 `RUNNING/FAILED`。重新计算的 M00、early_stop_v1 train 和最终权重 SHA 均与记录一致。
- 冻结脚本把 `earlystop_b1_full_epoch_20260722` 的 8 个白名单推理文件写入 `/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-earlystop-full-20260723/`。新 CPU 进程加载为 `Qwen3Model`，595,776,512 参数、`torch.float32`；目录模式 `0555`、文件模式 `0444`，源/目标权重 SHA 均为 `c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b`。模型正式登记为 M07。
- WSL 到服务器的大文件下行因本地代理网卡 MTU 9000 与实际链路分片阈值不匹配而停滞。使用 `/tmp` 中仅注入本次 `scp` 进程的 `TCP_MAXSEG=1000` 临时兼容层完成精确同步，验收后已删除该临时源码和共享库；没有修改系统路由、项目文件或服务器网络配置。Windows OpenSSH 密码回退失败一次后未重试，也未记录或输出凭据。
- 本地白名单目录为 `versions/M07/`，仅含 8 个推理文件和模型卡；逐文件 SHA 与服务器冻结 manifest 一致，不含 checkpoint、optimizer、日志、数据或凭据。
- 创建并发布公开 HF repo `Seraphic663/lrat-qwen3-0.6b-earlystop-full-20260723`，commit 为 `b5e765a48e30efb42502e2606fc2a932b6cc94b2`。匿名 API 回读确认 repo 公开、权重大小 2,383,139,480 bytes、远端 LFS SHA 为 `c937d61d...a2b`。
- 生成并解析验证单行比赛文件 `archive/diagnostics/submissions/submission_m07_20260723.jsonl`，SHA-256 为 `97238c4007a8feccb17ec3b4d7eafe26a226edaa766262a0a691e46dc406bc6a`。比赛平台尚未上传、未消耗当日额度；唯一下一步是用户确认 2026-07-23 额度未使用并明确授权网页提交。

## 2026-07-24：trajectory provenance、人工校准与 A/B/C 短训

- 用户要求把现有训练 pair 对应回原始 trajectory，利用 trajectory 特征做质量过滤，并以 A 原始、B 高置信硬删除、C 软降权开展 500/1000-step 对照。先在服务器建立空提交 `4614111a6d0b7f092b0f5191207eaad566a31d82`，随后把实现、策略、人工复核与 16 项测试提交为 `ca803f454f4c399fd3a0624855e57a9f2dcd3804`；没有 push origin，两项既有工作树遗留保持原样。
- 官方 trajectory 归档 SHA 为 `fb8ca29a...617a9`。使用官方 M00 tokenizer 后，M07 train 94,113 行中有 93,977 行唯一稳定映射、136 行歧义、0 行不匹配；provenance SHA 为 `924d373c...0b15`。这纠正了使用 M07 冻结 tokenizer 得到的旧临时低映射率，正式报告只采用官方源 tokenizer 结果。
- 两轮人工复核共 92 条：第一轮 62 条为 8 bad / 49 keep / 5 uncertain；严格交叉 30 条为 7 / 17 / 6。answer unmatched、长轨迹、继续搜索、低 rank、显式否定和严格组合的坏例精度均不足以推广为自动硬删除。
- B 因而只删除 15 条人工逐条确认的坏正例，94,098 行、SHA `3e45cdf1...172`；C 保留全部 94,113 行，只改变归一化 `reweight_rate`，SHA `6eecad9a...f6d5e`。独立全量复核确认 B 保留行与 A 字节一致，C 除权重外每个字段一致，A/C 平均权重相等。
- A/B/C 均从 M00、seed `20260716`、相同 1,000 optimizer steps 和同一 dev1500 开始。A@500/1000 MRR 为 `0.689502/0.697399`；B 为 `0.677130/0.692979`；C 为 `0.688759/0.696360`。B/C 在两个点的 MRR、R@1 都没有稳定超过 A。
- 10,000 次逐 query paired bootstrap 对 B/C 的三组预声明条件均为 false。selection SHA 为 `7694fd57...b6bc4`，内容为 `selected_arm=null / full_epoch_authorized=false / locked_test_used=false`。流水线在 `2026-07-24T06:27:58+08:00` 前写入 `STOPPED_BY_GATE` 与 `COMPLETED`，没有启动完整 1 epoch、没有读取 locked test500、没有冻结、HF、JSONL 或平台提交。GPU 最终均为 0 MiB / 0%，无残留训练/评测进程。
- 本实验不分配 M08，不改变 M07。完整证据与所有 SHA、指标、CI 见 `archive/experiments.md`。

## 2026-07-26：M12 纠偏、稀疏备份、trajectory-v2 与 M01 B 榜包

- 初步检查仅依据当时遗留的错误 `prepare.py` 和失真日志，曾错误判定 M12 修改 0 行。随后直接比较训练时锁定 artifact、raw 与 provenance，发现 79,496 行实际按 `event.search_idx` 乘 `1.1/1.2`。新增全量审计确认 94,113 行只有 `reweight_rate` 改动，93,977 stable/136 ambiguous 全部符合规则，数据 SHA `e8f2ef5d...663bb`，审计 SHA `35674e24...1fd7`；总权重放大 `1.1443876568×`，所以 M12 是真实但未归一化、存在尺度混杂的 search_idx 实验。决定继续健康 full epoch，终态自动做 dev1500；评测前不分配正式 ID，不冻结/HF/提交。
- 重写 `solution/experiments/m12_sequence_weight/prepare.py`，固定 raw/provenance/config SHA，按 row-aligned provenance 流式构建，输出存在时只做字节级验证而不覆盖；现场 `verified_existing` 精确重算 94,113 行、3,784,984,856 bytes 和 `e8f2ef5d...663bb`，fresh build/existing verify 单元测试通过，提交为 `cfe0d7c`。
- 停止旧的全量 checkpoint 复制 daemon，删除本任务生成的重复备份 `1500/2500/3000/3500/4500/5000/5500`，保留 `500/1000/2000/4000/6000`，启动 PID `54714` 的稀疏备份，后续只保留 `8000/10000/11500`。训练器自身 checkpoint 轮转和 M12 进程未中断，backup 由约 80G 降至约 34G。
- 服务器先以 `daf7d0225f5e82bbd146d7e3947678d0d5e83ac6` 保存旧 M08–M13 实验恢复点；该提交只是历史 checkpoint，不把其中错误逻辑追认为有效方法。后续修复实现提交为 `a7dbfd2`，生成物 ignore 为 `c5ddedf`，vendor 清理为 `68f6d18`，全量 provenance 脚本为 `bf845ca`，provenance 文档为 `ce9db5b`；未 push origin。
- 新建正确的 `trajectory_search_idx_v2`：读取 `provenance.event.search_idx` 零基字段，对 93,977 stable 行用 `1.0/1.1/1.1/1.2` 乘子并归一化，对 136 ambiguous 行保持原权重。94,113 行输出 SHA 为 `8fd600b9...7fced`；全量独立 audit 和 seed `20260726` 的 1,000 行抽查均通过，neutral changed 为 0，locked test 未读取。
- 部署 PID `56072` 的服务器持久触发器，等待 M12 `COMPLETED` 和 GPU 进程退出后自动运行首 seed raw/search 各 1,000 step、dev1500、10,000 次 bootstrap；首 seed 失败即停，首 seed 通过才扩至 3 seeds，至少 2/3 单 seed gate 及平均方向/top-k 条件通过才允许 candidate full epoch。自动流程不含冻结、HF 或平台提交。
- 在触发链前增加 `audit_and_evaluate_m12.sh`：M12 完成后先复核生命周期、数据 SHA 和全量 artifact 审计，再从最终目录独立加载做 dev1500并写 `m12_final_identity.json`；只有该阶段通过才进入归一化 v2。实现与测试提交为 `a37ca20`。
- 新增通用 full-epoch paired dev gate：以固定 SHA 的 M07 dev1500 为 baseline，要求 candidate 的 R@1/MRR 点估计为正且 95% bootstrap 下界非负、R@5/R@10 不退化。M12 终态和条件运行的 v2 full epoch 都在看到结果前绑定同一门槛；通过只表示 development candidate，不授权 locked test、冻结、HF 或平台提交。实现、两项单元测试和实际 M07 自比较 smoke 提交为 `b2117b6`。
- 联调发现历史 M07 eval 保存相对 dev 路径，而新 eval 将保存绝对路径；数据相同但直接比较字符串会误拒绝。comparator 已改为比较 `Path.resolve()` 后的真实路径，并新增相对/绝对等价测试，提交为 `0dafee9`。
- 在启动前重新计算最坏磁盘峰值：旧 1,000-step run 每个约 16G，6 run 加 M12 备份和 full epoch 可能耗尽余量。新增 `prune_short_training_run.py` 和测试；两点评测完成且 final/checkpoint-1000 SHA 一致后，保留 final 与 checkpoint-500 推理文件，移除 optimizer/scheduler/RNG 并删除冗余 checkpoint-1000，预计每 run 降至约 4.8G。实现提交为 `40609dd`。
- 进一步把三组 raw 短训合并为一份按 raw SHA 锁定的约 3.6G cache，把三组 search 短训及条件 full epoch 合并为另一份按 v2 SHA 锁定的 cache；所有新短训/full epoch 启动前要求 `/root/data` 至少 85G 可用。M12 训练期仍保留 8 个稀疏备份；clean `COMPLETED` 后只有 fresh terminal audit、dev1500、最终 model SHA 和身份报告全部通过，才自动缩减为 `1000/4000/8000/11500`，并在 `POST_COMPLETION_PRUNE_MANIFEST.json` 记录删除 checkpoint、文件、模型/trainer-state SHA 和回收字节。实现和测试提交为 `7de8f37`。
- M12 于 `2026-07-26T09:13:33+08:00` clean 完成 11,764/11,764，runtime `38,816.7924` 秒、train loss `0.8506403666`，最终权重 SHA `c961c32c...253a4e`。fresh 终态审计 SHA `2241b258...4077` 通过；独立 dev1500 为 R@1/R@5/R@10/MRR `0.6300/0.9060/0.975333/0.749001`。相对 M07 虽有 R@1/MRR 正点估计，但 95% CI 下界为负且 R@5 退化，gate=false；M12 不分配正式 ID、不冻结/HF/提交。
- 训练期 checkpoint-8000/10000/11500 的源/备份 15 文件逐项 SHA 均一致；M12 终态身份通过后自动将 8 个备份缩减为 `1000/4000/8000/11500`，prune manifest SHA `efa6d869...acacf`，回收 `28,665,373,459` bytes。一次失败的 SSH 哈希命令遗留本任务 PID group `65089`，从 `/root` 误扫 `eb-public` 并造成尾部吞吐下降；确认 cwd、父进程和启动命令后只终止该进程组并删除其 0-byte `/src.sha`，训练吞吐恢复，未触碰外部数据。
- trajectory-v2 首 seed raw/search 均从 M00 完成 1,000 step；final SHA 分别为 `28cf205d...1de36` 与 `48bd0ea4...34c7`，各自等于 checkpoint-1000。四份 dev1500 均为 1,500 条/1024 维：raw/search@500 R@1/MRR `0.550000/0.688687` 与 `0.546667/0.686951`，@1000 为 `0.560667/0.697297` 与 `0.560000/0.696132`。10,000 次 paired gate SHA `e46b921f...8e18` 的三个 criteria 全 false；流水线于 `11:48:40+08:00` 写 `STOPPED_BY_GATE/COMPLETED/TRIGGER_COMPLETED`，未运行 seed 2/3 或 full epoch。两臂 prune manifest SHA 为 `ce70d091...04c0`、`8ef3132e...7646`；无 locked test、冻结、HF 或平台操作。
- 新增 `solution/b/submission_m01/`：固定 M01 preprocess/train/inference/config/requirements/validator/packager，训练参数和加权跨卡损失代码与 M01 一致，checkpoint SHA 为 `b25b3...02501`。官方 pairs 全量 schema 审计发现并保留第 90,327 行空 query，所有 96,504 行 satisfied 为 true，负例数范围 5–407。
- 第一次 B 打包错误复制整套 FlagEmbedding，夹带 117 个非运行时资产和示例 JSONL；ZIP `6ff2cd...b49d` 被判定不合规并删除。修复后只 vendor 18 个训练 Python 源文件、最小 namespace 与许可证；最终目录 38 文件、Python 23 个，无 `.jsonl/.pt/.pth`，ZIP SHA 为 `ed328bc1...18e16`。
- 最终 B 包通过逐文件 manifest、M01 checkpoint hash、ZIP CRC/内容、vendored module import 和 CPU query/document inference smoke；两种模式均输出 1024 维 L2-normalized 向量。没有训练模型、上传外部服务或提交比赛平台。
- 对完整 M01 96,504 行重新构建 trajectory provenance：96,366 stable、138 ambiguous、0 mismatch，SHA `b7c9608e...018836`。`preprocess.py` 逐行回读通过。当前唯一数据合规阻断是服务器缺少组委会离线 corpus，无法验证每个正负 `doc_id/title/content`；在 corpus 补齐或组委会确认 released pairs 可直接用前，B ZIP 只作为工程基线。
- 新增 pair 文档/trajectory 全文审计，按 doc_id 和仅去尾空白后的文本对完整 1,989,015 个正负引用核验。trajectory 可解析 124,224 个 `get_document` 输出、51,213 个唯一 doc_id；pair 侧有 959,042 个唯一 doc_id。最终仅 184,222 个引用精确匹配，1,804,793 个未解析，其中 1,709,601 个负例引用的 doc_id 不在 trajectory。审计隔离 1 个参数/输出 doc_id 冲突的损坏事件，结果 SHA 为 `af4b0a24...13c07`；实证关闭“trajectory 可替代 corpus”的假设，但未关闭 corpus 合规阻断。
- 继续补齐 M01 B 榜：将 provenance 扩展到每行全部负例 ID，v3 结果为 96,366 stable、138 ambiguous、0 mismatch、96,504 行 negative IDs 全可追溯，SHA `0dd510fc...b690`。`preprocess.py` 同时重算全部 `reweight_rate` 并强制 trajectory/corpus 双门槛；训练命令复刻 M01 的换行模板、master port 29545 和 OMP threads 4，默认比赛推理 contract 对齐公开评测入口。主实现提交 `ab6be0b`，corpus 下载恢复提交 `3bc095e`，最终数据边界说明提交 `f7656b1`；10 项测试通过，未 push origin。
- 从 LRAT 论文固定链接取得 Wiki-25 corpus revision `fc1e312568b14385c04f41bc09157d8fa4c20658`，18,496,937,987 bytes、11,215,099 行、SHA `4d795938...e6393`。服务器直连 CDN 下载后完整 SHA 验收并设为 `0444`；全量流式校验覆盖 1,989,015 个 pair 文档引用和 959,042 个唯一 ID，缺失 0、文本不一致 0，最终 preprocess 状态 `full_raw_traceability_verified`。
- 最终 M01 B 包目录为 `ccir/submissions/submission_B_m01_final_20260726/`，41 文件、0 symlink；ZIP 为 `submission_B_m01_final_20260726.zip`，2,387,140,606 bytes，SHA `8172a899...b4c7fd`，MD5 `62afa7cb7c500402c267af32da85ece7`。checkpoint SHA `b25b3b08...02501`、MD5 `0e1c9c01fe0e49d133614130c2184161`；独立 ZIP CRC、逐文件 manifest、train dry-run、competition query/document CPU smoke 均通过。未上传 B 榜平台；本地约 6.7GB 未完成下载缓存和服务器旧 `submission_B_m01_20260726` 工程基线目录/ZIP 已在 canonical corpus 与最终 ZIP 验收后删除，历史日志、哈希和 Git 源码保留。

## 2026-07-28：M01 B 榜冻结预检与官方发布接入准备

- 核查官方 `Yuqi-Zhou/LRAT main=274d0abb32ce24e4da96594a11ad683f533be9ba`、HF LRAT-Train 与比赛页面。官方仍只有 A 榜评测目录，没有 B 榜代码、测试数据、输出 schema 或 validator；比赛页只明确 7 月 29 日登记模型 MD5、7 月 30–31 日提交测试集推理结果，以及 B Agent/测试代码在 B 榜结束后公布。因此未猜测 B 输出格式、未运行未知官方代码、未写平台。
- 新增 `solution/b/leaderboard_ops/b_preflight.py`：固定核对 M01 checkpoint 与最终 ZIP SHA/MD5，运行 package validator，完整读取 ZIP 做 CRC，检查磁盘安全线，并可在官方 B 文件发布后以非执行方式记录逐文件 SHA、Git 状态、README/validator 候选。新增 5 项单元测试；与原 B 包 6 项测试合计 11 项在本地和服务器均通过，带正确 PYTHONPATH 的 solution 全套 49 项测试也通过。准备实现提交为 `8114f2915b5ad9920fd26d8bbcf592470d75b11d`。
- 服务器正式预检记录 `ccir/submissions/b_m01_preflight_20260728.json` 的 SHA-256 为 `67ab026c74fc88e56159fcbb7fa1dbfca82d627268ebc82b770b8a43bf56fca8`。记录确认模型 MD5 `0e1c9c01fe0e49d133614130c2184161`、ZIP SHA `8172a899...b4c7fd`、41 文件包、CRC 与 30 GiB 空间门槛通过；当时可用 126,947,323,904 bytes，`official_release.status=not_released`，平台写入和模型修改均为 false。
- 服务器固定 `.venv` 的 11 个 requirements 版本逐项一致，`uv pip check` 对 237 个包通过。两个独立 CPU 进程重新编码 query/document，均为 1024 维，norm `0.999999969283/0.999999992432`；临时输入输出验收后删除。训练 dry-run 再次核对 M00、官方 pairs、完整追溯门槛和双卡 argv，记录 SHA 为 `48bda2f919d1221211d24d3287b3496d293121cf0d572a7bd30526719e25e854`。
- 尝试用隔离新 venv 按 requirements 联网安装，安装器在外部软件源阶段长期无网络连接或文件增长；确认只属于本任务后终止并删除 88 KiB 临时环境。该项登记为外部安装验证未完成，不误写为包失败；现有固定环境、独立加载、推理与 dry-run 均通过。
- 新增 XIR `control/CCIR/B_LEADERBOARD.md`，明确 7 月 29 日只登记模型 MD5，7 月 30–31 日先锁定官方文件和合同再运行，8 月 1–6 日只在 TOP3/组委会通知后交最终 ZIP；浏览器控制不可用时由用户接管已登录页面，禁止绕过登录或猜接口。该材料已从 WSDM 副本移出。

## 2026-07-28：M12 探索性 A 榜提交准备

- 用户选择把已经完整训练但内部 paired dev gate=false 的 M12 用于剩余 A 榜额度的探索性验证。准备不推翻原实验结论：M12 相对 M07 的 R@1/MRR 点估计为正，但 95% CI 下界为负且 R@5 下降；M01 仍是正式最佳和 B 榜模型。
- 服务器实时核查为 `ccir/dev@3413a7257910207343049378d304e83bc85b38f6`，工作树仍只有已知 leaderboard 脚本修改和 query-conflict JSON 遗留；两张 A40 均 0 MiB / 0%，无训练或评测进程，`/root/data` 可用约 119G。
- 重新复核 `m12_final_identity.json`：训练目录 clean `COMPLETED`，权重 SHA `c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e`，训练数据 SHA `e8f2ef5d...663bb`，terminal audit SHA `2241b258...4077`，dev1500 SHA `384759dc...659`，locked test 未使用。
- 使用既有冻结工具把 8 个白名单推理文件写入 `/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-searchidx-full-20260728/`。目录为 `0555`、文件为 `0444`，源/目标权重 SHA 一致；全新 CPU 进程加载为 `Qwen3Model`，595,776,512 参数、`torch.float32`。
- 精确同步 8 个推理文件到本地 `versions/M12/`，新增明确披露未归一化权重和 gate=false 的模型卡。公开 HF repo 为 `Seraphic663/lrat-qwen3-0.6b-searchidx-full-20260728`，commit `5461393bba73194ff905977f8713bd5212f20320`；认证与匿名 API 均确认公开可见，远端 `model.safetensors` 为 2,383,139,480 bytes，LFS SHA 与冻结源一致。
- 生成并解析验证单行比赛文件 `versions/M12/submission_m12_20260728.jsonl`，内容日期为 `2026-07-28`，SHA-256 为 `89dc411d93f6e3524a738a4cdcf4ef51bfa293ad5634db8294f4c4da54174ba5`。比赛平台未上传、7 月 28 日额度未由本会话消耗；网页提交仍需用户确认当日额度并执行。
- 随后用户确认已在比赛网页提交 M12；当前只登记为“平台评测中、成绩未知”，尚无平台记录号、评测时间或正式指标。M01 在 M12 出分前仍是当前已知最佳，M01 B 榜包和登记模型不变。
- 用户随后提供 Daily Results 截图：M12 于 `2026-07-27 17:18:15 UTC` 评测，记录 `998769`，正式成绩为 `Total 40.1 / Recall 46.0 / Success 20.5 / AvgSteps 16.1`。按公式推算 StepScore 约 `67.8`；M12 与 M01 展示综合分同为约 `40.16`，但同分规则先比较 Success，因此 M01 仍排在前。M12 相对 M01 的变化为 `Recall +1.2 / Success -0.5 / AvgSteps +0.7`。

## 2026-07-28：M10 补评与 M09 真多正例门控

- 补齐 M10 四个既有 500-step 学习率短训的同一 dev1500 评测。`3e-7/5e-7/1e-6/2e-6` 的 R@1/MRR 依次为 `0.491333/0.641949`、`0.507333/0.657108`、`0.543333/0.683104`、`0.564000/0.699346`；`2e-6` 为短训最佳。它尚无 1,000-step、temperature 或 full-epoch 验证，不分配正式模型 ID。
- 复核旧 M09 失败日志，根因是 torchrun 入口缺失 `FlagEmbedding` 模块路径；旧 `patch_loader.sh` 没有修改训练进程，旧数据脚本还读取错误字段，不能把旧失败解释为多正例方法失败。新增真正的多正例 dataset/collator/cross-device loss、固定门控脚本和测试，提交为 `d34e62c`；锁定训练集唯一空 query 改为按源行独立保留，修复提交为 `6aebba7`。
- 从固定 SHA 的 early_stop_v1 train 94,113 行构造 78,890 个 query-unique 样本，8,720 个样本含多正例，输出 SHA `742988dd...a2b9c`；同 query 下移除 2,195 个正负冲突。独立重建审计逐项确认全部正例、过滤后负例和聚合元数据，审计 SHA `422a9081...ba3`，未读 locked test、未用外部数据。
- M09 真多正例短训 clean 完成 1,000/1,000，runtime `3167.8759` 秒、train loss `0.9577179`，final SHA `3ae86998...3eaf`。M09@500/@1000 的 R@1/MRR 为 `0.550000/0.689276` 与 `0.560000/0.698362`；相对 raw 的 gate SHA `b8fcdd81...8201` 三项 criteria 全 false，写入 `STOPPED_BY_GATE/COMPLETED`，没有 full epoch、locked test、冻结、HF、JSONL 或平台提交。
- 使用受测短训清理工具保留 final 与 checkpoint-500 推理文件，删除冗余 optimizer 状态、完全重复的 checkpoint-1000 和专用 datasets cache；run 从约 16 GiB 降为约 4.5 GiB，prune manifest SHA `ef4d7268...bf8c`。完整报告见 `archive/experiments.md`。

## 2026-07-28：unit-weight 单变量短训

- 用户提出不使用 LRAT `reweight_rate`、其余沿用 M01 思路。确认 M01 本身没有动态早停，而是从 M00 完整训练 12,063 step / 1 epoch；M02/M06 出分后才事后确认 1 epoch 为经验最佳点。
- 新增 unit-weight 流式构建器、持久短训流水线和 2 项单元测试，服务器合计 6 项相关测试通过；实现提交为 `d37532d2b7dfe90eed727c2940ff34a4e16a258f`。数据构建只把 94,113 行 `reweight_rate` 设为 `1.0`，其他字段保持不变；94,111 行发生变化，原权重均值为 `1.000225`，输出 SHA `dc419196...64d32`。
- 从 M00、seed `20260716` 完成 1,000-step 双卡短训，runtime `3,177.1393` 秒、train loss `1.0783080`、final SHA `77d49697...11ec`。unit@500/1000 的 R@1/MRR 为 `0.546667/0.686934` 与 `0.561333/0.696939`；复用逐项配置一致的 weighted raw baseline `0.550000/0.688687` 与 `0.560667/0.697297`。
- 10,000 次 paired bootstrap gate SHA `1356b21b...b4c23` 三项 criteria 全 false。1000-step unit 相对 weighted 的 R@1 为 `+0.000667`、95% CI `[-0.008000, 0.009333]`，MRR 为 `-0.000358`、CI `[-0.005122, 0.004470]`，R@5 为 `-0.000667`。流水线于 `17:18:49+08:00` clean 停止，未做 full、locked test、冻结、HF 或平台操作。
- 清理后保留 final root 和 checkpoint-500 推理文件，prune manifest SHA `24ea2744...c9f8f`；GPU 和相关进程均已释放。完整报告见 `archive/experiments.md`。

## 2026-07-28：可解释的激进 search-stage reweight

- 预声明“后期 trajectory 搜索更接近证据需求”的可解释假设，完全替换官方 reasoning-length 权重：`idx0/idx1_2/idx3plus` 原始权重 `0.75/1.0/1.5`，对 stable 行统一归一化后为 `0.590178/0.786904/1.180356`，136 个 ambiguous 行保持 1。晚期与首轮严格 `2:1`，94,113 行总权重均值精确为 1，ESS ratio `0.946753`。
- 新增流式构建、独立逐行 audit、双基线持久短训流水线和按最早 stable search_idx 的 dev 分组解释器；实现提交 `cc625391`。首次只在数据构建后暴露直接执行导入错误，尚未启动训练；修复提交 `6f81c98`，随后 audit 和训练均一次 clean 完成。6 项针对性测试及 53 项当前环境可运行 unittest 通过。
- candidate 数据 SHA `458a366f...bc69c6`，manifest/audit SHA 为 `422c84e4...483cec` / `bbf89275...6a7a8`。独立 audit 确认只有 `reweight_rate` 改变，未用外部数据、未读 locked test。
- 从 M00、seed `20260716` 完成 1,000-step 双卡短训，runtime `3,165.372` 秒、train loss `1.0873577`。final/checkpoint-500 SHA 为 `e2e1522f...c5dc1` / `c01d2ec7...d1bde`；两份均由独立进程加载完成 dev1500。
- candidate@500/1000 的 R@1/MRR 为 `0.542667/0.684476` 与 `0.558667/0.696319`。1000-step 相对官方权重为 `-0.002000/-0.000978`，相对全 1 权重为 `-0.002667/-0.000620`；两组 10,000 次 paired bootstrap gate SHA `7fd41d42...a319`、`709b47ae...811` 均 false。
- 解释分组覆盖全部 1,500 条 dev query：idx0 220、idx1_2 324、idx3plus 954、ambiguous-only 2。最高权重的 idx3plus 组相对官方仍为 ΔR@1 `-0.002096`、ΔMRR `-0.000810`，表明 search_idx 同时编码困难度，不能直接当质量分数。流程写入 `both_baselines_passed=false / full_epoch_started=false`，未 full、locked test、冻结、HF、JSONL 或平台提交。
- 自动清理保留 final 与 checkpoint-500 推理证据，删除两点训练恢复状态和重复 checkpoint-1000，回收约 11.9GB；prune manifest SHA `8e4aca2d...f58b9`。GPU 与相关进程全部释放。完整报告见 `archive/experiments.md`。
- 报告与实现提交后，确认无进程占用并精确删除专用 datasets cache 3,760,177,113 bytes 和可重建候选 JSONL 3,780,077,654 bytes，合计约 7.54GB；`/root/data` 可用空间由约 93GB 增至约 100GB。两者不能原地恢复，但可由固定源 SHA、config、已提交构建器、manifest/audit 精确重建；模型、checkpoint-500、逐 query eval、双 gate、解释结果和日志均复核保留。

## 2026-07-28：服务器存储治理

- 在双卡空闲、无训练/评测进程时，对 `ccir/data/cache/` 的 32 个直接子目标生成逐路径/字节 manifest 后删除，共回收 `83,571,840,399` bytes；cache 根目录保留，原始 JSONL、代码、模型与日志未删。
- 新增受 allowed-root、`COMPLETED`、无 `RUNNING/FAILED`、checkpoint 模型与 trainer_state 合同约束的恢复状态清理器和 2 项测试。清理器逐 checkpoint 计算模型 SHA，只删除 optimizer/scheduler/RNG/scaler/training_args，保留全部推理文件、trainer_state 和目录。
- 共处理 22 条 completed run，40 个 checkpoint 模型和 trainer_state 保留，删除恢复状态 `190,660,623,320` bytes；M01/M02/M06/M07 最终 checkpoint SHA 与正式注册一致。未完成 smoke 和 early-stop 1000–6000 选点序列未处理。
- 本轮两部分合计回收 `274,232,463,719` bytes，`/root/data` 可用空间由约 100GB 增至约 355GB。完整边界见 `archive/experiments.md`。

## 2026-07-28：服务器高优先级冗余第二阶段

- 实时核查 `ccir/dev@61fb4e038c5128fe12af2aceab258d468d33bcc5`，双卡空闲、无相关进程，工作树只有两项已知遗留。扩展 completed checkpoint 清理器以支持 M12 独立备份根，并只接受 mtime 早于最终 `COMPLETED` 的 superseded `FAILED`；新增 4 项 pruner 测试和 2 项 hash-locked 文件删除测试，本地/服务器均通过。
- M12 full、M12@500 和四个 M10@500 共 12 个 checkpoint 的模型与 trainer_state 全保留，删除恢复状态 `57,198,182,100` bytes。11 个已关闭诊断派生 JSONL 按固定 path/bytes/SHA 全量复核后删除 `40,881,972,945` bytes；plan SHA `6c0b8bd8...5f0e8d`，完成 manifest 为 `ccir/reports/HIGH_PRIORITY_STORAGE_CLEANUP_20260728.json`。
- 第二阶段合计回收 `98,080,155,045` bytes；可用空间从 `371,403,927,552` 增至 `469,484,208,128` bytes。反向核验目标残留 0、checkpoint 恢复状态残留 0；M12 正式数据、M01 B ZIP、M01权重 SHA 分别仍为 `e8f2ef5d...663bb / 8172a899...b4c7fd / b25b3b08...02501`。未触碰 strategy-suite、PAUSED early-stop、旧 smoke、M03–M05、冻结模型和提交物。

## 2026-07-28：合规离线 Data Flywheel 代理

- 复核 LRAT 论文 Data Flywheel：论文以当前 retriever + 冻结 Agent 在 10K InfoSeekQA query 上收集新 trajectory，并循环更新五轮。比赛禁止外部数据与外部生成，因此实现为明确降级的离线代理：当前 allowed retriever 只在官方 pair 原负例中选自己的高分错误，不生成文本、不调用 Agent。
- 新增 shard 构建/独立审计/循环 gate/难度分组解释与持久流水线，提交 `45e1ced`；4 项针对性 unittest、语法和入口检查通过。第一轮 shard 5,834 行、4,886 unique query、97,192 passage，manifest/audit SHA 为 `fde23677...44af8` / `23b16ad4...a836`。
- 首次 control 在加载阶段 0 step 失败：FlagEmbedding 以路径是否含大写 `Qwen` 选择兼容分支，误向 Qwen3Model 传 `use_flash_attention_2`。失败目录/日志保留；用指向同一父 SHA 的可核验 `Qwen` symlink 与新 retry RUN_ID 恢复，修复提交 `6294dd4`。
- control/candidate 均 clean 完成 500/500，runtime `1605.9941/1580.9671s`，权重 SHA `ba84e477...ca22f` / `fd875283...269c99`。独立 dev1500 的 R@1/R@5/R@10/MRR 分别为 `0.558667/0.882000/0.971333/0.697717` 与 `0.558667/0.874667/0.968667/0.695414`。
- candidate 对 control 为 ΔR@1 `0`、ΔR@5 `-0.007333`、ΔR@10 `-0.002667`、ΔMRR `-0.002303`；99 query 改善、1264 持平、137 退化，R@5 95% CI 全负。gate SHA `a6ebca8e...0ea2a` false，写 `STOPPED_BY_GATE/COMPLETED`，未进入 loop2/full/locked test/冻结/HF/提交。
- 终态清理两个终点 checkpoint 的恢复状态 `9,533,030,862` bytes，再删除三个专用 cache 与两份可重建训练 JSONL `666,390,733` bytes，合计回收 `10,199,421,595` bytes；模型、终点证据、trainer_state、逐 query eval、gate/effects、mining/audit、失败日志和 prune manifest 保留。GPU/进程释放，`/root/data` 约 346G 可用。完整报告见 `archive/experiments.md`。

## 2026-07-29：论文 Data Flywheel 研究实现

- 直接核对 LRAT 论文第 9–11、13、15–16 页，确认旧 `offline_flywheel_v1` 只是 retriever hard-negative 代理，不含论文所需的新 query、Agent、Browse/reasoning trajectory 和逐轮 retriever 更新；旧实验完整迁移到 `run_offline_hard_negative_proxy_v1.sh`，原入口改为论文工作流兼容入口。
- 新增五轮 research-only pipeline：每轮从完整 InfoSeekQA pool 稳定抽取 10K query、按 current retriever 重建 Wiki-25 index、用 Tongyi-DeepResearch-30B-A3B 收集新 trajectory、用 Qwen3-30B-A3B-Thinking-2507 过滤 Browse、构造 Search-local negatives、按论文 Eq. (3) 全局归一化权重、训练 2 epoch，再审计 next retriever 后进入下一轮。
- 发现当前官方 `src/data_builder.py` 会保留 `satisfied=false` Browse 为正样本并混入历史搜索负例，因此新增论文专用 builder：只保留 Relevant 正例，每条负例严格来自对应 Search candidate set，并记录 `source_trajectory` 供反向验收。
- 训练合同从比赛缩放参数修正为论文 `group10 / query512 / passage512 / 2 epochs / LR 1e-6 / temperature 0.02 / batch 32`。默认两卡同时前向 32 query，另保留需要额外授权且明确非精确 negative pool 的低显存 profile；论文未公开的 query RNG、batch 口径和 optimizer continuation 被列为 disclosed unknowns，不做伪精确声明。
- 加入 research root 路径隔离、模型 lineage、服务模型 ID、GPU/进程/150 GiB 磁盘门槛、逐阶段 lifecycle marker、query/trajectory 完整覆盖、index shard SHA、pair provenance/公式、训练合同与 next-model SHA 审计。实现提交 `06c9d6c669bcbf5433f0ed95cc4d49badd6b7cf2`，16 项 flywheel/storage 目标测试、Python 编译、shell 语法、diff check 与 plan 入口通过；未 push origin。
- 当前预检为 `ready_for_full_collection=false`：论文 archive、Wiki-25 corpus、M00 和代码存在，但完整 InfoSeekQA research pool、Tongyi 30B Agent 与 Qwen 30B Judge 本地权重缺失。该路线明确比赛不合规，尚未下载外部资产或启动任何 GPU/训练；完整历史报告见 `/home/seraphic/xir/archive/research.md`。
- 对此前高优先级服务器清理做反向验收：11/11 派生 JSONL 不存在，54 个已瘦身 checkpoint 的模型和 trainer_state 保留，恢复状态残留 0，cache 空；两张 A40 空闲，`/root/data` 约 438 GiB 可用，M01 B 与两项既有工作树遗留未触碰。
## 2026-07-29 论文 Data Flywheel execution smoke

提交 `b5f6dec0301f6c8f06624d102e7dece04577d71e` 修正论文训练入口的 `PYTHONPATH`、`.venv/bin/ninja` PATH 与 weighted-loss reduction。真实 M00 tokenizer + builder CLI + fake Judge 端到端通过；论文 `1/N` mean reduction 的精确两卡 `batch32/group10/512` profile 完成 2 step，无 OOM。synthetic smoke 输出在哈希固化后删除，表观清理 `3,655,359,460` bytes；无 M ID、HF 或平台动作。详见 `/home/seraphic/xir/archive/research.md`。

## 2026-07-29：M10 `2e-6` 完整训练、冻结与 A榜准备

- 用户授权推进 M10 `2e-6` 完整训练。首次 `m10_lr2e6_full_epoch_20260729_seed20260716` 在 7,385/11,764 后随服务器重启中断，日志此前正常且没有 OOM/CUDA/NCCL/traceback；完整 `checkpoint-6000` 通过恢复合同。16:30 从该断点恢复模型、optimizer、scheduler 与 RNG，21:46 clean 完成 11,764/11,764，最终权重 SHA `cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984`。
- 最终目录独立 dev1500 于 21:57 完成，结果 SHA `29d07964f99eb774b7f8a36768c4c8b513f2b8394af6d9e2f08fd7cca12eb0c0`，R@1/R@5/R@10/MRR `0.630000/0.905333/0.974667/0.752016`。相对 M07/M12 的 MRR 点估计为 `+0.004507/+0.003015`，但 10,000 次 paired bootstrap 95% CI 均跨零；相对 M12 R@1 持平，R@5/R@10 各退 `0.000667`。locked test 未使用，M01 仍是正式已评测最佳和 B 榜基线。
- 原子冻结到 `ccir/models/Qwen3-Embedding-0.6B-LRAT-lr2e6-full-20260729/`，8 个推理白名单文件源/目标 SHA 一致，目录/文件权限 `0555/0444`；新 CPU 进程加载为 595,776,512 参数 `Qwen3Model`。freeze manifest SHA `d56ceafc...11b2a`，权重 MD5 `7a5b45a2784ba830213d9c50fc7648a9`。
- 服务器到本地单连接 SCP/SFTP 两次断开；随后仅在 `ccir/tmp/m10_hf_transfer_20260729` 生成 9 个本任务分片，四路精确下载并重组为 2,383,139,480 bytes，同 SHA 后删除服务器与本地分片。提交目录只含 README 和 8 个推理文件，无 checkpoint、optimizer、数据、日志或凭据。
- 首次 HF Xet 上传因 CAS connection reset/TLS EOF 卡在仅 `.gitattributes` 状态；确认 16 分钟无 commit 后只终止本任务上传 PID，禁用 Xet 改走标准 LFS。公开 repo `Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729` commit `9cdfdba51359ec65069a748cf8c00c55477016bb` 完成；匿名 API 确认 public、根目录文件完整、远端权重大小和 LFS SHA 与冻结源一致。
- 生成并解析单行 `versions/M10/submission_m10_20260729.jsonl`，日期 `2026-07-29`，repo ID 正确，SHA `193df8cde1decf491e53d2f9e89f5bb0c212ff54894844864eb2e6e30dee9ed2`；Windows 下载目录另有同 SHA 便捷副本。浏览器自动控制在连接前因 WSL/Windows 本地 URI 映射失败，未触达平台；已请用户手动上传并返回成功提示。当前只能登记“HF/JSONL 已准备，平台待回执”，不能写成已提交。
- 用户随后明确回复“已提交”，因此把 M10 更新为“用户确认平台已提交、待评测”；尚无榜单记录号、评测时间或正式指标，不提前补写成绩。

## 2026-07-30：B榜 GitHub 复现仓库

- 根据群公告修正赛程合同：A 榜无需 MD5；B 榜 7 月 30–31 日开放、所有队伍可参加、取最后一次提交，且需额外提供 GitHub 仓库链接。旧的 7 月 29 日 M01 MD5 登记计划取消。
- 创建独立公开仓库 `seraphic663/ccir-lrat-retriever`。默认 M01 profile 使用官方 96,504 pairs 原样完整一轮；M10 profile 只对同一官方 pairs 做固定 normalized-query SHA-256 分组，训练 94,113 行并使用 LR `2e-6`。两条路线均从 M00 开始，不生成新 trajectory、不调用外部 API、不引入外部训练数据或模型参数。
- 本地提交为 `b382fab`。WSL 与 Windows Git 到 `github.com:443` 的 Git 传输均不可达，因此改用已登录 GitHub CLI 的官方 Git Data API，先建立 main，再以完整 Git tree 原子写入源码；未读取、打印或落盘 token。远端 main commit 为 `32ab32915df983a914fe14920ec63ed0e0101c41`，tree 为 `fbd3f7be7f33d96419e6699d715ed0f749db6375`。
- 远端 34 个 blob 与本地 tracked 文件逐项 Git blob SHA 一致，visibility 为 public；GitHub connector 成功回读 README 和 M10 trainer。本地/服务器 4 项单测通过，真实 M01 与 M10 dry-run 分别通过 `full_raw_traceability_verified` 和 `official_pairs_query_disjoint_split_verified`，没有启动训练、生成新模型或写比赛平台。

## 2026-07-30：M10 正式 A榜与 GitHub 默认切换

- 用户提供 M10 正式 A榜记录 `998853`：`Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，提交时间 `2026-07-29 23:36:07`。相对 M01 为 `Total +0.5 / Recall +1.5 / Success +0.1 / AvgSteps +0.6`；正式提升主要来自 Recall，执行轮次略差。
- 赛务信息补充：A榜 Agent 为 Qwen3.5-4B；B榜使用多个、规模最高约 300B 的不同 Agent。该变化扩大 query/行为分布差异，不能保证 A榜排序直接迁移；M10 的 query-disjoint 隔离、正式 Recall 增益和不依赖外部 Agent 生成信号使其成为当前 B榜默认，M01 保留回退。
- 公开仓库根目录 `code/` 从 M01 切换为 M10；M01 移入 `profiles/m01/`。README 补齐官方来源追溯、query 归一化与 salted SHA 分组、94,113 行身份、完整 11,764-step 参数、服务器重启后从 checkpoint-6000 恢复、dev/locked-test边界、正式成绩与多 Agent 泛化风险。默认 trainer 同时要求 `full_raw_traceability_verified` 和固定 split 合同。
- 本地提交 `5389c3d`；服务器使用真实 M00、官方 pairs、全追溯 manifest、M10 train 和 split manifest 的 dry-run 通过，生成 `LR=2e-6 / max_steps=11764 / master_port=29531` 两卡命令，4 项单测通过，没有启动训练。远端 main commit 更新为 `38dbd875341650d4cfe89dd0b18840a1f96f5394`、tree `7fe9f1fe1b2bbea6184214edba1287aa3e7374c3`；34 个 blob 与本地逐项一致，GitHub connector 回读 README 通过。

## 2026-07-30：M10 B榜合同闭环

- 按比赛页新增的 B榜字段逐项审计：平台 JSONL 必须含 `date / hf_repo_id / github_repo_url`；GitHub 要含训练代码、配置与说明；最终复现包需包含 code、checkpoint、requirements 和 README。HF M10 当前 public、commit `9cdfdba5...016bb`，全量权重位于根目录；GitHub public 且默认 M10。
- GitHub README 新增官方三字段 JSONL、HF/GitHub 对应表，以及与主办方示例对齐的 `submission_B_m10_final_20260730.zip` 树。远端冻结 commit 为 `d9671c2a211c4454ef91c268d91a6c9970e2cf07`，34 个 blob 与本地逐项一致。
- 生成旧准备稿 `submission_B_m10_20260730.jsonl`，156 bytes，SHA `77c1599ded5236254cffb8522ecc920089fd473dca305b33a06985dec5d4e37b`；该准备稿随后由 2026-07-31 canonical 文件替代，单行 JSON 解析和三个精确字段通过，尚未上传平台。
- 新增 M10 package builder/validator并在服务器构建 43 文件最终目录与 deterministic ZIP。ZIP 2,387,245,663 bytes，SHA `4e1d0655964a17bb8b5979944b4e919712cf62dbf1ecfe19f7202ee6b0b94047`、MD5 `308fd66c69906dc87e99117b2f8f0b02`；checkpoint SHA/MD5 为 `cea87cca...a9984 / 7a5b45...48a9`。
- package validator、ZIP CRC、逐文件 manifest、HF/GitHub/模型身份和真实训练 dry-run 全通过。package checkpoint 的独立 CPU query inference 输出 1 行、1024 维，norm `1.0000000457`。CPU smoke 因模型加载超过外层 SSH 120 秒而短暂留下本任务 PID `2685`，随后自然完成并退出，无未知进程或 GPU 占用。

## 2026-07-31：停止 A榜复现并冻结 B榜 canonical 交付

- 按用户要求停止 A榜分数复现，不再运行 Qwen3.5 Agent pipeline；停止标记为 `ccir/outputs/reproduction/a_m10_qwen35_smoke_20260731/STOPPED_BY_USER`，两张 A40 均已释放。
- 审计公开仓库的清晰、准确、唯一与速度后，将公开入口收敛为单一 M10：删除 `profiles/m01/` 回退入口，新增 `code/reproduce_m10.py`、基础模型 revision/tokenizer SHA、全部官方输入 SHA、完整依赖锁、资源/耗时及 acceptance 说明。GitHub main 冻结为 `31949f6f53722f06e91bd2ded6ec7f3a48037bba`。
- canonical dry-run 用真实官方 inputs 重建 provenance、完整 corpus 追溯、query-disjoint split 和固定双卡命令，于 14:02 写入 `status=dry_run_verified`，未启动训练。此前独立完整训练耗时约 10.5 小时，dev1500 R@1/R@5/R@10/MRR `0.628000/0.906000/0.974667/0.750895`，相对提交模型各指标绝对差不超过 `0.002`；checkpoint SHA 不同，明确只作功能/统计复现。
- 生成 7 月 31 日三字段 JSONL，156 bytes，SHA `02c271173ad6bf5e438972bf9e617c66e1625bb13f5a7086c1b6bcf0167b3f49`，本地与 Windows Downloads 副本一致。匿名回读确认 HF public commit `9cdfdba5...016bb`、LFS 权重 SHA `cea87cca...a9984` 和 GitHub main 均匹配。
- 重建 45 文件最终目录和 ZIP。ZIP 2,387,249,735 bytes，SHA `c46f15d34a09dfe3108971ce56f837f1a23bbe39e35ee3a541bb609aea159984`、MD5 `cfeeda75d7f7458a147c159f06099ca6`；两套 validator、逐文件 manifest 和 `unzip -t` 全部通过。平台实际应上传 JSONL，尚未由本会话执行网页写入。
