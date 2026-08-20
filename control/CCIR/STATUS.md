# CCIR / LRAT 当前状态

> WSDM Short 范围说明：下文的 A/B 榜、HF 和平台结果是历史状态证据，不是当前论文流程；当前工作只围绕 LRAT 基线与受控研究实验。

## 2026-08-14 高学习率补充实验（运行中）

参数化 runner 已部署并通过编译。用旧的 32 组参数做 dry-run 已确认 `requested=32 / skip_completed=32 / to_run=0 / locked_test_read=false`。补充批次 `grid_lr_g5_rwon_high_20260814` 的第1组 `5e-6 / group5 / reweight on` 已完成，sample1500 为 MRR `0.715105422`、R@1 `58.33%`、R@5 `88.67%`、R@10 `97.20%`；第2组 `6e-6 / group5 / reweight on` 正在双 A40 训练。四组配置为 `learning_rate={5e-6,6e-6,8e-6,1e-5}`、500 optimizer steps、seed `20260716`，只评测 sample1500，不读取 locked test。首次启动因 runner 未创建 `logs/` 目录在 torchrun 前失败，已修复并通过远端编译检查。当前双卡约 14.36GB 显存、GPU 利用率 100%；可用空间约 389GB。远程脚本为 `ccir/reports/GRID_RUNNER_PARAMETERIZED_20260814.py`。SHA-256 校验保留为自动输入身份保护，不要求手工填写。

> 最后模型、GitHub 与 B榜提交物核查：2026-07-31；B榜成绩补充证据：2026-08-08 用户提供的 2026-08-07 榜单截图。M10 `2e-6` 正式 A榜为 `40.6`；B榜截图列 DefaultGroup 第 1，Team ID `362259`、Record ID `998912`、三模型平均 Total `56.18`，分项为 `67.42 / 60.96 / 40.16`。该项目前以截图为证据，尚未完成原始平台导出或 API 独立回读。

## 一句话状态

M10 正式 A榜记录 `998853` 为 `Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，相对 M01 为 `+0.5 / +1.5 / +0.1 / +0.6`；提升主要来自 Recall，执行轮次略差。最新截图显示 B 榜在 DeepSeek-V4-Flash-0731、Qwen3.5-35B-A3B、gpt-oss-120b 三种 Agent 上的 Total 分别为 `67.42 / 60.96 / 40.16`，平均 `56.18`；跨 Agent 结果已出现，但收益不均衡。7 月 29 日群公告取代旧 MD5 计划：A榜无需 MD5，B榜 7 月 30–31 日开放、取最后一次提交，并须额外提供可复现 GitHub 仓库链接。

## 服务器实时状态

```text
host：cs-e5be2-424d6-server
目录：/root/data/LRAT
分支：ccir/dev
HEAD：c3325199d2cf418967c4015864816b67aca79d36（本轮状态文档提交后将前移）
origin：https://github.com/Yuqi-Zhou/LRAT.git
uv：0.11.28
Python：3.10.16
torch：2.7.0+cu126
transformers：4.53.2
GPU：2 × NVIDIA A40，均 0 MiB / 0%
活跃任务：无 M09/M10/M12 训练、评测或触发器进程
/root/data：1.0T，总用约 583G，可用约 442G，57%
```

已知工作树遗留仍只有：

```text
 M xir_a_leaderboard_eval/run_browsecomp_plus_eval.sh
?? ccir/meta/query_conflict_cleaning_20260717.json
```

不得回退、删除、暂存或顺手提交。SSH 密钥与强制 `curve25519-sha256` 可用；本轮发现 ControlMaster 可出现“`-O check` 存活但新命令超时”的假活，恢复步骤见 `AGENTS.md`。

2026-07-24 为 trajectory 质量研究先建立空提交 `4614111a6d0b7f092b0f5191207eaad566a31d82`，随后把 provenance、人工复核、A/B/C 数据构建、训练编排、bootstrap 门槛和测试提交为 `ca803f454f4c399fd3a0624855e57a9f2dcd3804`，最终报告提交为 `eea4107a101e537ad0c1698045345031721bfc7b`。2026-07-26 以 `daf7d02` 保存旧实验恢复点，以 `a7dbfd2/c5ddedf/68f6d18/bf845ca/ce9db5b` 依次提交 trajectory-v2、B 榜包、生成物忽略、vendor 资产清理、全量 provenance 与文档；随后以 `ab6be0b/3bc095e/f7656b1` 提交 M01 B 榜全追溯实现、corpus 下载恢复和最终数据边界说明。没有 push origin。

## 正式模型与 A 榜

| ID | 模型 | 权重 SHA-256 | A榜 | 状态 |
|---|---|---|---:|---|
| M00 | 原始 Qwen3-Embedding-0.6B | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` | 28.5 | 官方基础模型 |
| M01 | 官方 pairs 1epoch | `b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501` | **40.1** | 正式最佳、冻结、HF、已提交 |
| M02 | 累计 2epoch | `1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050` | 38.2 | 冻结、HF、已提交 |
| M03–M05 | M01/M02 插值 | 见 `REGISTRY.md` | 未提交 | 诊断产物 |
| M06 | 累计 3epoch | `996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5` | 38.7 | 冻结、HF、已提交 |
| M07 | early_stop_v1 独立 1epoch | `c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b` | **39.5** | 冻结、HF已发布、已提交、已评测 |
| M10 | early_stop_v1、LR `2e-6`、独立 1epoch | `cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984` | **40.6** | **当前正式最佳、B榜默认候选** |
| M12 | search_idx 未归一化加权独立 1epoch | `c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e` | **40.1** | 已评测，记录 `998769`；Recall 46.0，Success tie-break 低于 M01 |

M01 正式字段：`Total 40.1 / Recall 44.8 / Success 21.0 / AvgSteps 15.4 / StepScore 69.1`。M02/M06 说明继续累计 epoch 不能改善 A 榜，停止第 4 轮路线。模型身份以权重 SHA 为准。

M07 正式字段：`Total 39.5 / Recall 45.0 / Success 19.8 / AvgSteps 15.9 / StepScore 68.2`。Recall 为队内最高，但 Success 和 AvgSteps 均不如 M01，最终 Total 低于 M01 `-0.6`。query-disjoint 路线牺牲了 2,391 条训练数据，检索召回更准但 Agent 端到端效率下降。

## 三条 early-stop 路线

| 路线 | 指标 | 实际状态 |
|---|---|---|
| `earlystop_b2_nocollision_sequence_20260720` | checkpoint-4000：MRR `0.733609`；step 5000：`0.734113` | step 5000 增量仅 `0.000503 < min_delta 0.001`；step 6000 触发停止并回选 checkpoint-4000。仅 checkpoint，未冻结/登记/HF/提交 |
| `earlystop_b1_standard_1k_20260721` → `frozen_earlystop_20260721` | dev1500 MRR `0.697495`；已打开 test500 MRR `0.703472` | 1000-step 诊断模型，SHA `15c937...f05c`；已冻结并通过 CPU 加载，但不是 M07，HF repo 不存在，旧 JSONL 不可提交 |
| `earlystop_b1_full_epoch_20260722` | dev1500 MRR `0.747510` | 已正式登记为 M07，冻结/HF/JSONL完成，平台 Total `39.5` |

三者训练配置、训练长度、选择协议和生命周期不同，目录名或旧 JSONL 不能互相替代。test500 已被历史诊断打开，后续不得回流用于 checkpoint、方法或超参选择。

## 2026-07-24 trajectory provenance 清洗三臂

官方 pair 与原始 trajectory 的映射已经实证可行。M07 train 94,113 行中，93,977 行唯一稳定映射、136 行歧义、0 行不匹配；provenance SHA 为 `924d373cffbc8f229c326c1de02e7e223c87109e57c32c1f2be29ea2e65f0b15`。两轮人工复核共 92 条，显示 answer unmatched、长轨迹、继续搜索、低 rank、显式否定及其严格交叉组合的坏正例精度不足，不能作为大批硬删除规则。

三臂均从 M00、seed `20260716`、相同 1,000 optimizer steps 和同一 dev1500 开始：A 为原始 94,113 行；B 仅删除 15 条人工确认坏正例；C 保留全部行且只改归一化 `reweight_rate`。

| 模型点 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| A@500 | 55.27% | 87.80% | 96.93% | 0.689502 |
| B@500 | 53.47% | 86.53% | 96.60% | 0.677130 |
| C@500 | 55.13% | 87.87% | 96.93% | 0.688759 |
| A@1000 | 56.13% | 88.07% | 96.87% | 0.697399 |
| B@1000 | 55.60% | 87.47% | 96.93% | 0.692979 |
| C@1000 | 56.00% | 88.13% | 96.80% | 0.696360 |

B/C 在 500 和 1000 的 MRR、R@1 都没有稳定超过 A；10,000 次 paired bootstrap 区间也未满足下界非负，且 top-k 守门未全过。`selection.json` 为 `selected_arm=null / full_epoch_authorized=false / locked_test_used=false`，流水线写入 `STOPPED_BY_GATE` 和 `COMPLETED`，没有 full-epoch 输出、冻结、HF 或平台操作。完整报告见 `archive/experiments.md`。

## 2026-07-26 M12 纠偏、trajectory-v2 与 B 榜准备

此前仅依据当时遗留的错误 `prepare.py` 和失真日志，把 M12 判断成修改 0 行的 raw 重序列化，这是错误的。对训练时锁定的 `e8f2ef5d...663bb` artifact、raw 数据和 `924d373c...0b15` provenance 的全量逐行审计证明：79,496 行按稳定 `event.search_idx` 精确乘 `1.1/1.2`，14,617 行乘 `1.0`，136 个 ambiguous 行不变，非权重字段 0 处变化；审计 SHA 为 `35674e24...1fd7`。`prepare.py` 已修复为固定输入 SHA 的流式可复现构建/验证，现场 `verified_existing` 精确重算相同 94,113 行、3,784,984,856 bytes 和输出 SHA，提交为 `cfe0d7c`。M12 是真实但未归一化的 search_idx 实验，总权重放大 `1.1444×`，因此保留健康 full epoch 并在终态自动独立评测；评测前不赋正式模型 ID，不冻结、不 HF、不提交。

M12 在 2026-07-26 09:13:33 +08:00 clean 完成，runtime `38,816.7924` 秒、train loss `0.8506403666`，最终权重 SHA 为 `c961c32c...253a4e`。fresh 数据审计 SHA `2241b258...4077` 再次确认 94,113 行合同；独立 dev1500 SHA `384759dc...659` 为 R@1/R@5/R@10/MRR `0.6300/0.9060/0.975333/0.749001`。相对 M07 的 R@1/MRR delta `+0.003333/+0.001492`，95% CI 下界 `-0.002000/-0.001488`，R@5 delta `-0.001333`，所以 gate SHA `53f5d86e...4c85` 为 false。2026-07-28 用户选择用剩余 A 榜额度做探索性验证后，M12 完成全新只读冻结、独立 CPU 加载、公开 HF 与 JSONL；这不改变 gate=false，也不改变 M01 正式最佳和 B 榜模型身份。

M12 训练期 8 个稀疏备份在 8000/10000/11500 均通过 15 文件源/备份 SHA 验收；完成期在 fresh audit、dev1500、身份和最终权重通过后自动缩减为 `1000/4000/8000/11500`，prune manifest SHA `efa6d869...acacf`，回收 `28,665,373,459` bytes。期间一次失败的 SSH 哈希命令遗留 PID group `65089`，从 `/root` 误扫外部 `eb-public` 并使 M12 尾部吞吐短暂下降；确认其属于本任务后只终止该进程组并删除 0-byte `/src.sha`，未触碰数据，吞吐从 4–8 恢复至约 16 step/分钟。

归一化 trajectory search-index v2 首 seed raw/search 的 final SHA 分别为 `28cf205d...1de36` 和 `48bd0ea4...34c7`，各自与 checkpoint-1000 一致。raw/search@500 的 R@1/MRR 为 `0.550000/0.688687` 与 `0.546667/0.686951`；@1000 为 `0.560667/0.697297` 与 `0.560000/0.696132`。paired gate SHA `e46b921f...8e18`：500 R@1/MRR delta `-0.003333/-0.001737`，1000 为 `-0.000667/-0.001164`，三个 criteria 全 false。两臂 prune manifest SHA 为 `ce70d091...04c0`、`8ef3132e...7646`，均保留 final 与 checkpoint-500 推理证据。流水线于 11:48:40 +08:00 正常停止，没有 seed 2/3、full epoch、locked test、冻结、HF 或平台提交。

M01 B 榜代码位于 `solution/b/submission_m01/`，最终服务器目录为 `ccir/submissions/submission_B_m01_final_20260726/`，ZIP 为 `ccir/submissions/submission_B_m01_final_20260726.zip`，2,387,140,606 bytes，SHA-256 `8172a899636d986e9997c1be1be331bd0543878a25a07f16e0a4ab9cd7b4c7fd`，MD5 `62afa7cb7c500402c267af32da85ece7`。包含 41 个文件、0 symlink，不含训练数据、optimizer、日志或外部模型/API 结果；独立 `unzip -t`、逐文件 manifest、模型 SHA/MD5、train dry-run 和 competition-contract CPU query/document smoke 全部通过。

完整 M01 provenance v3 为 `96,366 stable / 138 ambiguous / 0 mismatch / 96,504 negative_ids_traceable`，SHA-256 `0dd510fcc41444bfcff99381ce077e5048375855add8498f9ab10afabea4b690`。固定 Wiki-25 corpus 为 18,496,937,987 bytes、11,215,099 行，SHA-256 `4d795938...e6393`；全量校验覆盖 1,989,015 个 pair 文档引用和 959,042 个唯一 ID，缺失与文本不一致均为 0，状态 `full_raw_traceability_verified`。官方第 90,327 行空 query 被精确保留。完整历史流程见 `/home/seraphic/xir/control/CCIR/B_LEADERBOARD.md`。

2026-07-28 再次核查官方 `Yuqi-Zhou/LRAT main=274d0abb32ce24e4da96594a11ad683f533be9ba`、HF LRAT-Train 与比赛页面：尚无 B 榜代码、测试数据、输出 schema 或 validator，公开页面仍只说明 7 月 29 日登记模型 MD5、7 月 30–31 日提交测试集推理结果、B Agent 与测试代码在 B 榜结束后公布。新增 `solution/b/leaderboard_ops/b_preflight.py` 与 5 项测试，准备实现提交为 `8114f2915b5ad9920fd26d8bbcf592470d75b11d`；带正确 PYTHONPATH 的 solution 全套 49 项测试通过。服务器正式预检记录为 `ccir/submissions/b_m01_preflight_20260728.json`，SHA-256 `67ab026c74fc88e56159fcbb7fa1dbfca82d627268ebc82b770b8a43bf56fca8`。记录确认模型与 ZIP SHA/MD5、41 文件 package validator、完整 ZIP CRC 和 30 GiB 空间门槛通过，`platform_write_performed=false`，模型登记值为 `0e1c9c01fe0e49d133614130c2184161`。

2026-07-31 已把独立公开复现仓库 `https://github.com/seraphic663/ccir-lrat-retriever` 收敛为唯一 M10 路径，删除公开的 M01 回退入口，新增 `code/reproduce_m10.py` canonical CLI、完整依赖锁、基础模型 revision/tokenizer SHA、全部官方输入 SHA 与资源/耗时说明。远端 main 已冻结为 `31949f6f53722f06e91bd2ded6ec7f3a48037bba`；4 项单测、compile、JSON 和 diff 检查通过。真实 canonical dry-run 于 14:02 完成，重新核验 96,504 条 trajectory provenance、11,215,099 行 corpus、94,113 行 query-disjoint train 及固定双卡训练命令，状态 `dry_run_verified`，未启动训练或 A榜评测。

M10 B榜当日 JSONL 为 `versions/M10/submission_B_m10_20260731.jsonl`，156 bytes，SHA-256 `02c271173ad6bf5e438972bf9e617c66e1625bb13f5a7086c1b6bcf0167b3f49`，字段为 `date / hf_repo_id / github_repo_url`，已解析通过并复制到 Windows Downloads，但尚未上传平台。最终复现目录为 `ccir/submissions/submission_B_m10_final_20260731/`，45 文件；ZIP 为 `ccir/submissions/submission_B_m10_final_20260731.zip`，2,387,249,735 bytes，SHA-256 `c46f15d34a09dfe3108971ce56f837f1a23bbe39e35ee3a541bb609aea159984`、MD5 `cfeeda75d7f7458a147c159f06099ca6`。两套 package validator、逐文件 manifest、ZIP CRC、HF/GitHub/checkpoint 身份和 canonical dry-run 均通过。

服务器固定 `.venv` 的 11 个依赖版本与 package `requirements.txt` 逐项一致，`uv pip check` 对 237 个包通过；两个独立 CPU 进程再次得到 query/document 1024 维、norm `0.999999969283/0.999999992432`。训练 dry-run 记录 SHA-256 为 `48bda2f919d1221211d24d3287b3496d293121cf0d572a7bd30526719e25e854`，M00/pairs/完整追溯和双卡训练参数均通过。隔离新 venv 的联网安装在外部软件源阶段长期无进展，已只终止本任务安装进程并删除 88 KiB 临时环境；因此不把“全新联网安装”登记为通过或包失败。

## 2026-07-28 M10 补评与 M09 真多正例门控

M10 四个既有 500-step 学习率短训已全部完成同一 dev1500 评测。`3e-7/5e-7/1e-6/2e-6` 的 R@1/MRR 依次为 `0.491333/0.641949`、`0.507333/0.657108`、`0.543333/0.683104`、`0.564000/0.699346`；`2e-6` 为短训最佳，但尚无 1,000-step、temperature 或 full-epoch 验证，因此仍是未分配正式模型 ID 的诊断信号。

旧 M09 在任何有效 step 前因 `FlagEmbedding` 模块路径缺失失败，旧 patch 也没有进入 torchrun 进程。M09 v2 已实现真正的 query-unique 联合多正例 InfoNCE；代码提交为 `d34e62c/6aebba7`。锁定 train 94,113 行构造 78,890 个样本，输出 SHA `742988dd...a2b9c`，独立追溯审计 SHA `422a9081...ba3`；8,720 个样本含多正例，移除 2,195 个同 query 正负冲突，唯一空 query 按源行单独保留。

M09 短训 clean 完成 1,000/1,000，runtime `3167.8759` 秒、train loss `0.9577179`，final SHA `3ae86998...3eaf`。M09@500/@1000 的 R@1/MRR 为 `0.550000/0.689276` 与 `0.560000/0.698362`；相对 raw 的 10,000 次 paired gate SHA `b8fcdd81...8201` 三项 criteria 全 false，流程写入 `STOPPED_BY_GATE/COMPLETED`。没有 full epoch、locked test、冻结、正式 ID、HF、JSONL 或平台提交。

短训清理后保留 final root 与 checkpoint-500 推理文件，删除冗余 optimizer 状态、完全重复的 checkpoint-1000 和专用 datasets cache；run 由约 16 GiB 降至约 4.5 GiB，prune manifest SHA `ef4d7268...bf8c`。完整报告见 `archive/experiments.md`。

## 2026-07-29 M10 `2e-6` 完整训练与 A 榜准备

用户授权把 M10 `2e-6` 从短训信号推进到完整训练。`m10_lr2e6_full_epoch_20260729_seed20260716` 从 M00、固定 early_stop_v1 train 94,113 行、官方权重和 seed `20260716` 开始，目标为 11,764 step / 1 epoch。首次进程在 7,385 step 后随服务器重启中断，没有 traceback、OOM、CUDA 或 NCCL 错误；完整 `checkpoint-6000` 保留模型、optimizer、scheduler 和 RNG 状态。16:30 从该断点恢复，21:46 clean 到达 11,764/11,764，最终 `COMPLETED` 存在且无 `RUNNING/FAILED`。

自动独立 dev1500 于 21:57 完成：R@1/R@5/R@10/MRR `0.630000/0.905333/0.974667/0.752016`，评测结果 SHA `29d07964...eb0c0`；最终权重 SHA `cea87cca...a9984`。M10 相对 M07 的 MRR `+0.004507`、相对 M12 `+0.003015`，但逐 query 10,000 次 bootstrap 的 95% CI 均跨零；相对 M12 的 R@1 持平，R@5/R@10 各低 `0.000667`。因此它是本地 MRR 新高点估计，但不是稳健优胜证据；locked test 未使用。

22:04 使用原子冻结工具写入 `/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-lr2e6-full-20260729/`，8 个白名单推理文件源/目标 SHA 一致，目录 `0555`、文件 `0444`；新 CPU 进程成功加载 595,776,512 参数 `Qwen3Model`。本地提交目录权重同 SHA，HF `Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729` commit `9cdfdba5...016bb` 已由匿名 API 确认 public、根目录完整、远端权重 2,383,139,480 bytes 和 LFS SHA 一致。JSONL `versions/M10/submission_m10_20260729.jsonl` 为单行 85 bytes，SHA `193df8cd...e9ed2`。

第一次 HF Xet 上传遭 CAS connection reset/TLS EOF，远端仅有 `.gitattributes` 且 16 分钟无 commit；确认本任务 PID 后只终止该上传进程，禁用 Xet 改走标准 LFS，4 分 44 秒完成正式 commit。服务器到本地的单连接 SCP/SFTP 也两次断开；随后只在明确的本任务临时目录生成 9 个 256MiB 分片，四路下载、重组并命中权重 SHA 后删除本地和服务器临时分片，冻结模型不变。

比赛浏览器自动控制因 WSL 工作目录无法映射为 Windows 浏览器运行时可接受的本地 URI而在连接前失败，没有打开、填写或点击平台。已向用户提供 `C:\Users\31572\Downloads\submission_m10_20260729.jsonl` 的同 SHA 副本；用户随后明确确认“已提交”。当前状态为用户确认平台已提交、待评测，尚无榜单记录号或正式指标。

## 2026-07-28 unit-weight 单变量短训

为检验 LRAT 官方 `reweight_rate` 是否真正有益，从 M00、seed `20260716` 开始，在 query-disjoint train 94,113 行上只把权重设为 `1.0`，其余 M01/M07 参数保持不变。94,111 行权重发生变化，但原权重均值为 `1.000225`，因此本实验主要隔离相对加权而非总训练强度。unit 数据 SHA 为 `dc419196...64d32`。

短训 clean 完成 1,000/1,000，runtime `3,177.1393` 秒、train loss `1.0783080`，final SHA `77d49697...11ec`。unit@500/1000 的 R@1/MRR 为 `0.546667/0.686934` 与 `0.561333/0.696939`；同配置 weighted baseline 为 `0.550000/0.688687` 与 `0.560667/0.697297`。10,000 次 paired gate SHA `1356b21b...b4c23` 三项全 false：1000-step R@1 仅 `+0.000667` 且 CI 跨零，MRR `-0.000358`，R@5 `-0.000667`。流程写入 `COMPLETED / gate_passed=false / full_epoch_started=false`，未读取 locked test、未做 full、冻结、HF 或平台操作；GPU 已释放。完整报告见 `archive/experiments.md`。

## 2026-07-28 aggressive search-stage reweight 短训

为检验“trajectory 后期搜索更接近证据需求”的假设，新增可解释激进权重：stable `idx0/idx1_2/idx3plus` 的归一化权重为 `0.590178/0.786904/1.180356`，晚期与首轮严格 `2:1`，136 个 ambiguous 行保持 `1.0`。94,113 行总权重均值精确为 1，只有 `reweight_rate` 改变；候选数据 SHA `458a366f...bc69c6`，manifest SHA `422c84e4...483cec`，独立 audit SHA `bbf89275...6a7a8` 通过。

从 M00、seed `20260716` 完成 1,000-step 双卡短训，runtime `3,165.372` 秒、train loss `1.0873577`；final SHA `e2e1522f...c5dc1`，checkpoint-500 SHA `c01d2ec7...d1bde`。candidate@500/1000 的 R@1/MRR 为 `0.542667/0.684476` 与 `0.558667/0.696319`，在两个时间点均未同时超过官方权重与全 1 权重。

1000-step 相对官方权重的 R@1/MRR 为 `-0.002000/-0.000978`，相对全 1 权重为 `-0.002667/-0.000620`；两组 10,000 次 paired bootstrap 的核心指标 CI 下界均为负，双 gate SHA 分别为 `7fd41d42...a319`、`709b47ae...811`，均 false。分阶段解释覆盖全部 1,500 条 dev query；被赋最高权重的 `earliest_idx3plus` 组相对官方仍为 ΔR@1 `-0.002096`、ΔMRR `-0.000810`，仅 R@5/R@10 小幅上升。结论是简单单调放大 search_idx 把“查询困难”与“样本质量”混在一起，不做 full。无 locked test、冻结、HF、JSONL 或平台操作；GPU 已释放，prune manifest SHA `8e4aca2d...f58b9`。报告提交后删除专用 cache 与可重建候选 JSONL 共约 7.54GB，空间由约 93GB 增至 100GB；模型、评测、gate、解释、manifest/audit 和日志保留。完整报告见 `archive/experiments.md`。

## 2026-07-28 服务器 cache 与 checkpoint 恢复状态清理

在无活跃训练/评测、双卡空闲时，删除 `ccir/data/cache/` 下 32 个可重建 cache 共 `83,571,840,399` bytes；逐目标 manifest 为 `ccir/reports/SERVER_CACHE_CLEANUP_20260728.json`。随后使用受测白名单清理器处理 22 条 completed run，只删除 optimizer/scheduler/RNG/scaler/training_args 共 `190,660,623,320` bytes，40 个 checkpoint 模型和 40 个 trainer_state 全部保留。两部分合计回收 `274,232,463,719` bytes，`/root/data` 可用空间由约 100GB 增至约 355GB。

M01/M02/M06/M07 最终 checkpoint SHA 与注册表一致；未完成 smoke 和 early-stop 1000–6000 选点序列未处理。删除的 cache 可重建；删除 optimizer 后不能精确恢复旧动量，但所有目标均为 clean completed，最终模型与评测证据不受影响。完整清单和边界见 `archive/experiments.md`。新的 flywheel 500-step loop 使用 `save_steps=1000`，不生成中间 checkpoint；M01 B 包的历史复现配置不修改。

第二阶段高优先级瘦身已完成：M12 full/M12@500/四个 M10@500 的 12 个 checkpoint 保留全部模型与 trainer_state，删除恢复状态 `57,198,182,100` bytes；11 个已关闭诊断派生 JSONL 经路径/bytes/SHA 锁定后删除 `40,881,972,945` bytes。合计回收 `98,080,155,045` bytes，`/root/data` 可用空间从 `371,403,927,552` 增至 `469,484,208,128` bytes，占用从 67% 降至 58%。M12 正式数据 SHA `e8f2ef5d...663bb`、M01 B ZIP SHA `8172a899...b4c7fd`、M01 权重 SHA `b25b3b08...02501` 复核不变；strategy-suite、PAUSED early-stop、旧 smoke、M03–M05 和正式/提交产物未触碰。

## M07 最新完整 epoch

```text
RUN_ID：earlystop_b1_full_epoch_20260722
基础模型：M00，SHA 0437e45c...e23fd
训练数据：ccir/data/experiments/early_stop_v1/train.jsonl
数据：94,113 行，3,784,989,300 bytes
数据 SHA：158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9
配置：2 × A40；per-device batch 1；gradient accumulation 4；LR 1e-6
训练：11,764 / 11,764 steps，1 epoch，attempts 1
runtime：38,892.1802 秒
train loss：0.8489937656
完成：2026-07-22 12:40:17 +08:00
输出：/root/data/LRAT/ccir/outputs/checkpoints/earlystop_b1_full_epoch_20260722/
权重：2,383,139,480 bytes
SHA：c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b
```

2026-07-23 冻结前重新计算 M00、训练数据和最终权重 SHA，全部与 run_config 一致。顶层有 `COMPLETED`，无 `RUNNING/FAILED`，保留 checkpoint-11000/11500/11764。训练结束后的独立 dev 评测进程成功加载最终目录；冻结脚本又在干净 staging 上完成 CPU 加载、8 文件逐项 SHA 和只读权限验收。

dev1500：

| 模型 | MRR | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| 正式早停 checkpoint-4000 | 0.73361 | 61.07% | 89.60% | 97.20% |
| 最新完整 epoch | **0.74751** | **62.67%** | **90.73%** | **97.47%** |

dev1500 是 normalized-query 隔离的选模证据，但不能保证超过 M01 的 A 榜 `40.1`。

## 数据、HF 与本地提交状态

- 官方 raw pairs：96,504 行、3,883,089,616 bytes、SHA `dd75a3f...be1b9`。
- `early_stop_v1`：train 94,113 行、dev 1,500 query、test 500 query，三部分 normalized-query overlap 为 0。
- `REGISTRY.md` 已正式登记 M07。
- 服务器冻结目录：`ccir/models/Qwen3-Embedding-0.6B-LRAT-earlystop-full-20260723/`。
- 本地旧诊断副本和 `submission_earlystop_20260722.jsonl` 均保留作证据；认证 HF 查询 `Seraphic663/lrat-qwen3-0.6b-earlystop-20260721` 返回 404，文件不可提交。
- M07 本地目录、公开 HF repo、commit `b5e765a...94b2`、远端 LFS SHA 和 `submission_m07_20260723.jsonl` 均已核验。
- 平台 2026-07-23 当日额度无法从项目内确认，必须由用户现场确认。

## 当前决策

结论更新：M12 正式成绩为 `Total 40.1 / Recall 46.0 / Success 20.5 / AvgSteps 16.1`，与 M01 展示同分但 Success tie-break 居后。M09、unit-weight 和 aggressive search-stage reweight 的 paired gate 均为 false，不做完整训练。M10 `2e-6` 已从早期短训信号推进为完整一轮模型，并以正式 A榜 `40.6` 成为当前最佳；早期“只做 1,000-step 复核”的判断已过期。

下一步是用户在平台上传 Windows Downloads 中已准备的 `submission_B_m10_20260731.jsonl`。提交后不得修改对应 HF 模型或 GitHub commit `31949f6f...7bba`。最终复现 ZIP 已准备但不是当前 JSONL 上传文件；仅在主办方要求复现包时交付。

## 2026-07-28 合规离线 Data Flywheel 代理

论文原方法用当前 retriever 与冻结 Agent 在新 InfoSeekQA query 上生成新 trajectory，比赛规则下不可照搬。本轮只在官方 pair 的原负例中做 retriever-in-the-loop hard-negative 刷新，不生成文本、不调用 Agent、不使用外部数据/模型/API。

第一轮 query-disjoint shard 为 5,834 行、4,886 个唯一 normalized query。相同父模型、shard、seed 和 500 step 下，control/candidate dev1500 为：

| 模型 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| parent | 55.0000% | 87.7333% | 96.8000% | 0.688687 |
| uniform control | 55.8667% | 88.2000% | 97.1333% | 0.697717 |
| hard-negative candidate | 55.8667% | 87.4667% | 96.8667% | 0.695414 |

candidate 相对 control 的 R@1 为 0、R@5 `-0.7333pp`、R@10 `-0.2667pp`、MRR `-0.002303`；99 query 改善、1,264 持平、137 退化，R@5 95% CI 全为负。gate SHA `a6ebca8e...0ea2a` 为 false，流程只完成一轮即停止；无第二轮/full/locked test/正式模型 ID/冻结/HF/提交。

control/candidate 权重 SHA 分别为 `ba84e477...ca22f`、`fd875283...269c99`。终态清理移除两个 checkpoint 的恢复状态及专用 cache/可重建 JSONL，共回收 `10,199,421,595` bytes；模型、终点 checkpoint、trainer_state、逐 query eval、gate/effects、mining/audit、失败日志和 prune manifest 保留。GPU/相关进程均释放，`/root/data` 约 346G 可用。完整报告见 `archive/experiments.md`。

## 2026-07-29 论文 Data Flywheel 研究实现

提交 `06c9d6c669bcbf5433f0ed95cc4d49badd6b7cf2` 将旧 `run_offline_flywheel_v1.sh` 改为论文复现入口，并把旧 hard-negative 代理完整保存在 `run_offline_hard_negative_proxy_v1.sh`。新流程严格执行每轮 10K InfoSeekQA sample、当前 retriever 新索引、Tongyi Agent 新 trajectory、Qwen 30B judge、Relevant-only Search→Browse pairs、对应 Search 内负例、Eq. (3) 全局归一化权重、group10/query512/passage512/2epoch weighted InfoNCE，再用新 retriever 进入下一轮。

该实现为研究专用，固定 `competition_submission_eligible=false`；research 输入、轨迹、索引、pairs 和 checkpoint 被路径与审计合同隔离，不能进入 M01 B 包、正式模型目录、HF 或比赛平台。当前预检因缺完整 InfoSeekQA pool、Tongyi-DeepResearch-30B-A3B 和 Qwen3-30B-A3B-Thinking-2507 本地权重而 `ready_for_full_collection=false`，尚未下载外部资产、启动 Agent/Judge、构建索引、生成 trajectory 或训练新模型。完整历史证据见 `/home/seraphic/xir/archive/research.md`。

后续提交 `b5f6dec0301f6c8f06624d102e7dece04577d71e` 把该实现推进到真实 execution smoke。真实 M00 tokenizer + builder CLI + localhost fake Judge 成功产出 Relevant-only、search-local-negative、Eq. (3) 全局归一化 pairs；精确两卡 profile `batch32/group10/query512/passage512/cross-device negatives` 在两张 A40 上完成 2 step，runtime `19.6797` 秒、train loss `3.890625`，无 OOM。直接核对论文第 11 页后把 weighted loss 从通用的除以 batch 权重和改为论文专用 `-(1/N)Σw_i log p_i`；只有论文入口设置 `LRAT_WEIGHT_REDUCTION=paper_mean`，默认比赛训练语义不变。13 项相关单元测试和 builder integration 通过。

所有 synthetic smoke 输出已在哈希固化后删除，表观清理 `3,655,359,460` bytes；`/root/data` 可用空间最终为 `469,481,529,344` bytes。没有新模型 ID、HF 或平台动作，GPU 与相关进程均释放。完整历史证据见 `/home/seraphic/xir/archive/research.md`。

高优先级服务器瘦身同时完成反向验收：11/11 hash-locked 派生 JSONL 均不存在，54 个已处理 checkpoint 的模型和 trainer_state 均保留，恢复状态残留 0，`ccir/data/cache` 为 0 文件/0 bytes。实时服务器为 `ccir/dev@06c9d6c`，两张 A40 空闲、无相关进程，`/root/data` 约 438 GiB 可用；两项既有工作树遗留未触碰。
