# CCIR 模型与提交注册表

> 由 REGISTRY.md 与 REGISTRY.md 合并（2026-08-04）；身份以权重 SHA 为准。

> WSDM Short 范围说明：表中的 HF、A/B 榜和外部平台字段只保留作历史 provenance；本副本不执行也不准备新的平台提交。

## 2026-08-08 B榜结果补充证据

用户提供的榜单截图 `/home/seraphic/xir/report/CCIR答辩/temp/codex-clipboard-74e9d757-d752-4403-bd19-b5a631bcde5b.png` 标注更新时间为 2026-08-07，显示 DefaultGroup 排名 1、Team ID `362259`、提交 Record ID `998912`、平均总分 `56.18`。三种 Agent 的 Total 分别为 DeepSeek-V4-Flash-0731 `67.42`、Qwen3.5-35B-A3B `60.96`、gpt-oss-120b `40.16`。该记录是用户提供的截图证据，尚未以原始平台导出或独立 API 回读复核；不要将截图证据扩写为完整逐 query 评测明细。

## 目录
- [REGISTRY.md](#model_registry)
- [REGISTRY.md](#submission_registry)

## MODEL_REGISTRY.md

# 模型注册表

> 模型身份以 `model.safetensors` 的 SHA-256 为准，不以目录名为准。`dev500` 与完整训练数据重叠，表中相关指标仅用于训练内诊断。

| ID | 模型 | 来源/方法 | 权重 SHA-256 | 训练重叠诊断 R@1/R@5/R@10/MRR | HF / A榜 | 状态 |
|---|---|---|---|---|---|---|
| M00 | 原始 Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` | `0.494 / 0.804 / 0.944 / 0.6343869` | 平台 Baseline Qwen3 Embedding：`Total 28.5 / Recall 28.5 / Success 10.6 / AvgSteps 17.8 / StepScore 64.3` | 官方基础模型/平台基线 |
| M01 | LRAT 1epoch | M00，全量官方 pairs，1 epoch | `b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501` | `0.672 / 0.934 / 0.984 / 0.7803662` | `Seraphic663/lrat-qwen3-0.6b-1epoch-20260716`；A榜 `Total 40.1 / Recall 44.8 / Success 21.0 / AvgSteps 15.4 / StepScore 69.1` | **正式参考、冻结、已提交** |
| M02 | LRAT 累计 2epoch | 从 M01 重新初始化 optimizer/scheduler，再跑 1 epoch | `1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050` | `0.738 / 0.942 / 0.990 / 0.8281279` | `Seraphic663/lrat-qwen3-0.6b-2epoch-20260717`；A榜 `Total 38.2 / Recall 42.7 / Success 18.3 / AvgSteps 15.5 / StepScore≈69.0` | **冻结、已提交；低于 M01/M06** |
| M03 | epoch1/2 插值 α=0.25 | `0.75*M01 + 0.25*M02` | `697cd068cb9bddf7e09404bc4bf541d9a389a61c2b5e3b4e1c2dda326dd03e1b` | `0.692 / 0.934 / 0.984 / 0.7940351` | 未上传、未提交 | 诊断产物 |
| M04 | epoch1/2 插值 α=0.50 | `0.50*M01 + 0.50*M02` | `1c1bf0ff00c1ad63c5d019d0acd567677abc3987424770158af3fc24f9a2d917` | `0.710 / 0.928 / 0.986 / 0.8067376` | 未上传、未提交 | 诊断产物 |
| M05 | epoch1/2 插值 α=0.75 | `0.25*M01 + 0.75*M02` | `3f52cf1ed2b8c97acadd46415c1aed484247cab0a9a20bb6c0ad4272659460ad` | `0.714 / 0.940 / 0.988 / 0.8117059` | 未上传、未提交 | 诊断产物 |
| M06 | LRAT 累计 3epoch | 从 M02 重新初始化 optimizer/scheduler，再跑 1 epoch | `996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5` | `0.772 / 0.952 / 0.992 / 0.8529487` | `Seraphic663/lrat-qwen3-0.6b-3epoch-20260718`；A榜 `Total 38.7 / Recall 43.4 / Success 19.0 / AvgSteps 15.8` | **冻结、已提交；低于 M01** |
| M07 | LRAT early_stop_v1 独立 1epoch | 从 M00 在 query-disjoint train 94,113 行上独立训练 1 epoch | `c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b` | dev1500 query-disjoint：`0.626667 / 0.907333 / 0.974667 / 0.747510` | `Seraphic663/lrat-qwen3-0.6b-earlystop-full-20260725`；A榜 `Total 39.5 / Recall 45.0 / Success 19.8 / AvgSteps 15.9 / StepScore 68.2` | **冻结、已提交、已评测** |
| M10 | LRAT early_stop_v1、LR `2e-6`、独立 1epoch | 从 M00 在同一 query-disjoint train 94,113 行上训练 11,764 step；保留官方 `reweight_rate`，仅把 LR 从 `1e-6` 提高为 `2e-6` | `cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984` | dev1500 query-disjoint：`0.630000 / 0.905333 / 0.974667 / 0.752016` | `Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729` @ `9cdfdba51359ec65069a748cf8c00c55477016bb`；A榜 `Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，记录 `998853` | **当前正式 A榜最佳、B榜默认候选** |
| M12 | LRAT search_idx 未归一化加权独立 1epoch | 从 M00 在 query-disjoint train 94,113 行上训练 1 epoch；79,496 行按稳定 trajectory `event.search_idx` 乘 `1.1/1.2` | `c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e` | dev1500 query-disjoint：`0.630000 / 0.906000 / 0.975333 / 0.749001` | `Seraphic663/lrat-qwen3-0.6b-searchidx-full-20260728` @ `5461393bba73194ff905977f8713bd5212f20320`；A榜 `Total 40.1 / Recall 46.0 / Success 20.5 / AvgSteps 16.1` | **已评测，记录 998769；与 M01 同分但 Success tie-break 居后** |
> 2026-07-19 的 query-disjoint raw/cleaned 1000-step 对照未通过预声明 bootstrap 门槛，按实验协议停止在短训阶段；它没有完整 cleaned 1epoch、冻结模型或 HF/比赛提交，因此从未占用 M07。

> 2026-07-24 的 trajectory provenance A/B/C 500/1000-step 对照同样未通过门槛：B@1000 权重 SHA 为 `54842c20cae84a2f1927992f3ea50f39404694c01a3759b923fa3d404cea9aca`，C@1000 为 `5420bf9ad9d8d7dc36c0880663cbfc4e56b4ee62e2727593ca81343393f4fcda`；两者相对 A@1000 的 MRR/R@1 点估计均为负，`full_epoch_authorized=false`。它们只是未冻结、未上传、未提交的短训诊断产物，不分配 M08，不改变 M07。

## 未分配正式模型 ID 的实验别名

| 历史别名 | 实际状态 | 模型注册结论 |
|---|---|---|
| M08 status weighting | 数据中 status 无区分度，未形成有效候选 | 不分配 M08 |
| M09 multi-positive | v2 真多正例目标已完成 500/1000-step 与 paired gate；三项 criteria 全 false，停止完整训练 | 不分配 M09 |
| M10 LR/temp scan | 四个 500-step LR 点中 `2e-6` 最佳；用户随后授权完整训练，正式 full-epoch 权重已冻结并登记为上表 M10 | 已分配 M10 |
| M11 hard negatives | 数据准备设计，未形成正式权重 | 不分配 M11 |
| M13 reasoning weighting | 数据准备设计，且既有粗特征已被 trajectory v1 证明精度不足 | 不分配 M13 |
| unit-weight M01-like | 从 M00 做 500/1000-step 单变量对照；全部 `reweight_rate=1.0`，paired gate 三项全 false | 不分配正式 ID |

2026-07-26 正确实现的 `trajectory_search_idx_v2` 同样只是门控实验名，不预占 M08 或任何正式 ID。首 seed raw/search 1,000-step 已完成；search 相对 raw 在 500/1000 的 R@1、MRR 点估计均为负，10,000 次 paired bootstrap gate=false，因此协议在首 seed 后停止，没有额外 seeds 或 full epoch。两臂只是已清理并保留 SHA/评测证据的短训诊断产物，不冻结、不上传、不提交。

2026-07-28 的 M09 v2 把旧的伪多正例方案改为 query-unique 联合多正例 InfoNCE；数据 SHA `742988dd...a2b9c`、独立追溯审计 SHA `422a9081...ba3`。1,000-step final 权重 SHA `3ae86998...3eaf`，但相对 raw@500/raw@1000 的正式 gate SHA `b8fcdd81...8201` 三项 criteria 全 false，因此没有完整 epoch、冻结、HF 或平台提交。M10 四个既有 500-step LR 点中 `2e-6` dev1500 最佳，为 R@1/MRR `0.564000/0.699346`；这只是后续 1,000-step 复核信号，不登记正式模型。完整证据见 `archive/experiments.md`。

2026-07-28 的 unit-weight 单变量短训把 early_stop_v1 的 94,113 行全部设为 `reweight_rate=1.0`，其余配置与同 seed weighted raw 基线一致。unit@500/1000 的 R@1/MRR 为 `0.546667/0.686934` 与 `0.561333/0.696939`；相对 weighted 的 gate SHA `1356b21b...b4c23` 三项全 false。没有完整 epoch、locked test、冻结、HF 或平台操作；完整证据见 `archive/experiments.md`。

2026-07-28 的 `aggressive_search_weight_v1` 也是未分配正式模型 ID 的短训诊断。它完全替换官方 reasoning-length 权重，将 stable `idx0/idx1_2/idx3plus` 归一为 `0.590178/0.786904/1.180356`，晚期与首轮严格 `2:1`，总权重均值保持 1。final SHA 为 `e2e1522f...c5dc1`；candidate@500/1000 的 R@1/MRR 为 `0.542667/0.684476` 与 `0.558667/0.696319`。相对官方权重和全 1 权重的双 paired gate 均 false，最高权重的 idx3plus 组也没有 MRR 增益；没有 full、locked test、冻结、HF 或平台操作。完整证据见 `archive/experiments.md`。

2026-07-23 23:22 +08，`earlystop_b1_full_epoch_20260722` 通过训练完成标记、M00/data/最终权重 SHA、独立 dev 评测加载和冻结脚本 CPU/staging 验收，正式分配为 M07。`frozen_earlystop_20260721` 仍只是 SHA `15c937...f05c` 的 1000-step 诊断模型，不得追认为 M07。

## M07 冻结位置

```text
原训练输出：/root/data/LRAT/ccir/outputs/checkpoints/earlystop_b1_full_epoch_20260722/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-earlystop-full-20260723/
训练 Git commit：04203abe3b01b7f8d2d7bfbc371d53a4edea0755
基础 M00 SHA-256：0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd
early_stop_v1 train SHA-256：158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9
权重 SHA-256：c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b
```

M07 训练为 11,764/11,764 steps、完整 1 epoch，runtime `38,892.1802` 秒、train loss `0.8489937656`、attempts `1`。dev1500 query-disjoint 指标为 R@1/5/10 `0.626667/0.907333/0.974667`、MRR `0.7475095`。冻结副本含 8 个推理文件与 `FREEZE_MANIFEST.json`，目录 `0555`、文件 `0444`；源/目标权重 SHA 一致，新 CPU 进程加载为 `Qwen3Model`（595,776,512 参数，`torch.float32`）。冻结不等于 HF 或平台提交。

本地提交副本位于 `/home/seraphic/xir/versions/M07/`，8 个推理文件的大小和 SHA 与服务器冻结 manifest 一致；另含公开模型卡。公开 HF commit 为 `b5e765a48e30efb42502e2606fc2a932b6cc94b2`。匿名 API 回读确认远端 `model.safetensors` 为 2,383,139,480 bytes、LFS SHA-256 为 `c937d61d...986a2b`。HF 发布完成不等于平台已提交或评测。

## M10 冻结位置

```text
RUN_ID：m10_lr2e6_full_epoch_20260729_seed20260716
原训练输出：/root/data/LRAT/ccir/outputs/checkpoints/m10_lr2e6_full_epoch_20260729_seed20260716/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-lr2e6-full-20260729/
本地提交副本：/home/seraphic/xir/versions/M10/
训练 Git commit：c3325199d2cf418967c4015864816b67aca79d36
基础 M00 SHA-256：0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd
early_stop_v1 train SHA-256：158eb3843e8e022b5b0d7e64446ddf0782e2ad1aaa830f9f5aa3fb3b06c835c9
权重 SHA-256：cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
权重 MD5：7a5b45a2784ba830213d9c50fc7648a9
冻结 manifest SHA-256：d56ceafcc110ea37d480d1f137b7306a63422afc1b4e257e44119e27ada11b2a
HF：Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729
HF commit：9cdfdba51359ec65069a748cf8c00c55477016bb
```

M10 从 M00 使用与 M07 相同的 94,113 行 query-disjoint train、官方 `reweight_rate`、group size 6、temperature 0.02、seed `20260716` 和 11,764-step 完整一轮；实验变量是学习率 `2e-6`。首次训练在 7,385 step 后随服务器重启中断，恢复流程从完整 `checkpoint-6000` 还原模型、optimizer、scheduler 与 RNG 状态并 clean 到达 11,764/11,764；没有 OOM 或训练异常，最终 `COMPLETED` 存在且无 `RUNNING/FAILED`。

最终目录独立 dev1500 评测为 R@1/R@5/R@10/MRR `0.630000/0.905333/0.974667/0.752016`，结果 SHA-256 为 `29d07964f99eb774b7f8a36768c4c8b513f2b8394af6d9e2f08fd7cca12eb0c0`。相对 M07 的 MRR 点估计 `+0.004507`，相对 M12 为 `+0.003015`，两组逐 query bootstrap 95% CI 均跨零；locked test 未使用。正式 A榜随后给出 `Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，记录 `998853`，超过 M01 的 `40.1 / 44.8 / 21.0 / 15.4`。提升主要来自 Recall `+1.5`，Success `+0.1`，代价为 AvgSteps `+0.6`。

冻结工具只复制 8 个推理白名单文件，源/目标权重 SHA 一致；目录 `0555`、文件 `0444`，全新 CPU 进程加载为 `Qwen3Model`（595,776,512 参数，`torch.float32`）。公开 HF 由匿名 API 回读确认 commit、根目录文件、2,383,139,480-byte 权重和 LFS SHA 全部一致。正式 JSONL 为 `versions/M10/submission_m10_20260729.jsonl`，SHA-256 `193df8cde1decf491e53d2f9e89f5bb0c212ff54894844864eb2e6e30dee9ed2`。平台正式记录 `998853` 的提交时间为 `2026-07-29 23:36:07`。B榜使用多个、最高约 300B 的不同 Agent，跨 Agent 优势尚未被实测；基于 query-disjoint 隔离和正式 Recall 提升，M10 当前设为 B榜默认候选，M01 保留为回退。

## M12 冻结位置

```text
原训练输出：/root/data/LRAT/ccir/outputs/experiments/m12_full/checkpoints/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-searchidx-full-20260728/
本地提交副本：/home/seraphic/xir/versions/M12/
训练数据 SHA-256：e8f2ef5dfad4b26343710e7cb53760e25718cf9eeded11f41e65b28a00e663bb
权重 SHA-256：c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e
权重 MD5：9708a8653a78ec7e20c168e2ddda8483
HF：Seraphic663/lrat-qwen3-0.6b-searchidx-full-20260728
HF commit：5461393bba73194ff905977f8713bd5212f20320
```

M12 原先因 paired dev gate=false 保留为未归一化诊断；2026-07-28 用户明确选择用剩余 A 榜额度进行探索性验证，因此完成冻结、登记、HF、JSONL 与网页提交。平台记录 `998769` 为 `Total 40.1 / Recall 46.0 / Success 20.5 / AvgSteps 16.1`；相对 M01 是 `Recall +1.2 / Success -0.5 / AvgSteps +0.7`，展示综合分相同，按 Success tie-break 居后。它证明加权方向能换取更高召回，但没有证明综合表现或跨 Agent 泛化优于 M01。

## M01 冻结位置

```text
原训练输出：/root/data/LRAT/ccir/outputs/checkpoints/qwen3_dual_full_epoch1_20260716/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-1epoch-20260716/
本地正式版本包：/home/seraphic/xir/versions/M01/
```

M01 的三个模型权重副本与 HF LFS 摘要一致。服务器另保留 `checkpoint-11500/12000/12063` 用于恢复；未经用户确认不得删除。

2026-07-26 为 B 榜登记 M01 `model.safetensors` MD5 `0e1c9c01fe0e49d133614130c2184161`，对应 SHA-256 仍为 `b25b3b08...02501`、大小 2,383,139,480 bytes。最终 B 包的 `MODEL_IDENTITY.json`、`BUILD_MANIFEST.json` 和独立 `md5sum/sha256sum` 三处一致；该动作不创建新模型 ID，也不改变 M01 权重。

## M02 冻结位置

```text
原训练输出：/root/data/LRAT/ccir/outputs/checkpoints/qwen3_dual_epoch2_from_epoch1_20260717/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-2epoch-20260717/
本地提交副本：/home/seraphic/xir/versions/M02/
训练 Git commit：b0bd39e4980a1faf7e7282dff05fc8ac444244a3
官方数据 SHA-256：dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9
权重 SHA-256：1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050
```

冻结副本仅含 9 个顶层推理文件，均设为只读；不含 checkpoint、`training_args.bin`、`run_config.env`、训练状态或嵌套目录。源、本地提交副本与冻结副本逐文件一致，并已由 CPU 新进程独立加载为 `Qwen3Model`（595,776,512 参数）。公开 HF 仓库远端权重大小和 LFS SHA-256 已回读核验；HF commit 为 `28ab688b5e91d4e6bc61642950ec4bb354a73c23`。A榜于 `2026-07-19 08:59:22 UTC` 记录 `Total 38.2 / Recall 42.7 / Success 18.3 / AvgSteps 15.5`，StepScore 按展示 AvgSteps 近似为 `69.0`。

## M06 冻结位置

```text
原训练输出：/root/data/LRAT/ccir/outputs/checkpoints/qwen3_dual_epoch3_from_epoch2_pathfix_20260718/
服务器推理副本：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-3epoch-20260718/
训练 Git commit：8f5b4d702e16cf17a1449ef72868b0fef3f4a608
输入 M02 SHA-256：1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050
官方数据 SHA-256：dd75a3f1970438f0905a3e3e93e3d98dc1122cdb0e054ba87159a9368afbe1b9
权重 SHA-256：996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5
```

训练于 `2026-07-18 03:25:56 +08:00` 开始、`14:26:41 +08:00` 完成，共 12,063 steps，runtime `39,578.2804` 秒，train loss `0.4103184`，首次尝试成功。M06 冻结副本含 8 个源推理文件和 `FREEZE_MANIFEST.json`。目录权限为 `0555`，文件为 `0444`；源、目标与 manifest 的逐文件大小和 SHA-256 全部一致。冻结后再次由全新 CPU 进程加载为 `Qwen3Model` / `Qwen2TokenizerFast`（595,776,512 参数）。8 个推理文件已逐项同步到本地 `versions/M06/`，公开 HF 仓库远端权重大小和 LFS SHA-256 与服务器一致。A榜于 `2026-07-18 08:31:42 UTC` 记录 `Total 38.7 / Recall 43.4 / Success 19.0 / AvgSteps 15.8`，低于 M01 的 `40.1`；平台记录号为 `998655`。

## M01 正式 A榜基准

```text
团队：DefaultGroup
Total：40.1
Recall：44.8
Success：21.0
AvgSteps：15.4（越低越好）
StepScore：69.1
提交时间：2026-07-17 10:40:36 UTC
```

同一榜单的 `BaselineLRAT` 为 `34.0 / 36.1 / 15.5 / 16.8 / 66.5`。M01 的 Total 绝对提升 `6.1`，相对提升约 `17.9%`；Recall 提升 `8.7`，Success 提升 `5.5`，AvgSteps 降低 `1.4`，StepScore 提升 `2.6`。平台 Total 与规则公式的未四舍五入结果 `40.14` 一致。

M02 相对 M01 为 `Total -1.9 / Recall -2.1 / Success -2.7 / AvgSteps +0.1`；M06 相对 M01 为 `Total -1.4 / Recall -1.4 / Success -2.0 / AvgSteps +0.4`。因此退化在第2轮已经发生，第3轮虽比第2轮回升 `0.5`，仍未恢复到第1轮。M02/M06 的 StepScore 分别按展示 AvgSteps 近似为 `69.0/68.4`，不登记为平台正式显示值。所有榜单行使用 Agent model `Qwen3.5-4B`；这里的 Qwen3.5 是比赛 Agent，不是检索器基础模型，M00–M06 检索器仍严格基于 `Qwen3-Embedding-0.6B`。

## 注册规则

新增正式模型时必须记录：

1. 唯一 ID 和不可变 SHA-256。
2. 基础模型、数据 SHA、训练 Git commit 和完整方法。
3. 服务器冻结路径，不得直接引用 `RUNNING` 目录。
4. 本地诊断的真实口径，明确是否独立。
5. HF repo、HF commit、提交日期和正式榜单指标。
6. 状态只能使用“实验产物 / 候选 / 冻结 / 已提交 / 退役”。

## 2026-07-28 未编号诊断：offline flywheel v1

该路线不分配正式 M ID。父模型为既有 raw checkpoint-500，SHA `acd2eb20228caf9aff83d20afb6bc35d514cd378739cc117489b7165bc213373`；从同一父模型在同一 5,834 行官方 shard 上分别训练 uniform control 与 retriever-mined hard-negative candidate 500 step。

| 诊断产物 | 权重 SHA-256 | dev1500 R@1 / R@5 / R@10 / MRR | 生命周期 |
|---|---|---|---|
| control retry1 | `ba84e4774ef61cfdcadef7ab1376d3fadbb75f36d7fba1a34296679876eca22f` | `0.558667 / 0.882000 / 0.971333 / 0.697717` | clean completed；未冻结/HF/提交 |
| candidate retry1 | `fd875283ed060450800845750377db54157ab3d3a491148ea8c439d094269c99` | `0.558667 / 0.874667 / 0.968667 / 0.695414` | gate=false；未冻结/HF/提交 |

candidate 相对同源 control 的 MRR/R@5/R@10 均退化，第一轮 gate SHA `a6ebca8e00afa501794d523fc51bba7da166a49b09cafed778d86b8d6640ea2a` 为 false。没有第二轮或 full epoch；不得把 candidate 登记成正式模型或替代 M01。

## 2026-07-29 research-only 论文 Data Flywheel

提交 `06c9d6c669bcbf5433f0ed95cc4d49badd6b7cf2` 已实现论文式五轮 Agent trajectory flywheel，提交 `b5f6dec0301f6c8f06624d102e7dece04577d71e` 又完成真实 pair-builder 与精确两卡训练 smoke，并按论文第 11 页修正为 `1/N` weighted-loss reduction。当前仍未下载研究所需的外部 InfoSeekQA/Tongyi/Qwen-Judge 资产，也未生成正式 trajectory、training pairs 或 research checkpoint；synthetic smoke 权重与 M00 SHA 相同且已删除，因此不分配 M ID。该工作流固定 `competition_submission_eligible=false`；未来即使产生 research checkpoint，也不得进入本注册表的正式比赛模型、`ccir/models`、HF 或比赛提交生命周期。

---

## SUBMISSION_REGISTRY.md

# 比赛提交注册表

> A榜每天只能提交一次。任何 session 都不得仅凭聊天记忆判断当日额度，提交前必须让用户确认平台当天是否已有记录。

| 日期 | 模型 | 权重 SHA-256 | HF repo / commit | JSONL | 平台状态 | 正式成绩 |
|---|---|---|---|---|---|---|
| 2026-07-17 | M01 LRAT 1epoch | `b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501` | `Seraphic663/lrat-qwen3-0.6b-1epoch-20260716` / `36038a60ba0b0bb36e87b42646460ae159b3097f` | `versions/M01/submission_1epoch_20260717.jsonl` | 已评测 | `Total 40.1；Recall 44.8；Success 21.0；AvgSteps 15.4；StepScore 69.1` |
| 2026-07-18 | M06 LRAT 累计 3epoch | `996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5` | `Seraphic663/lrat-qwen3-0.6b-3epoch-20260718` / `b51ef2b1fb8a1bca671b551b030aa1b7f47c59f3` | `versions/M06/submission_3epoch_20260718.jsonl` | 已评测；记录 `998655` | `Total 38.7；Recall 43.4；Success 19.0；AvgSteps 15.8；StepScore未显示` |
| 2026-07-19 | M02 LRAT 累计 2epoch | `1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050` | `Seraphic663/lrat-qwen3-0.6b-2epoch-20260717` / `28ab688b5e91d4e6bc61642950ec4bb354a73c23` | `versions/M02/submission_2epoch_20260719.jsonl` | 已评测 | `Total 38.2；Recall 42.7；Success 18.3；AvgSteps 15.5；StepScore≈69.0` |
| 2026-07-23 | M07 LRAT early_stop_v1 独立 1epoch | `c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b` | `Seraphic663/lrat-qwen3-0.6b-earlystop-full-20260725` / `1da2f3ec2dae197736f59f1fc45257e0c161e4a7` | `versions/M07/submission_m07_20260725.jsonl` | 已评测；记录 `998718` | `Total 39.5；Recall 45.0；Success 19.8；AvgSteps 15.9；StepScore 68.2` |
| 2026-07-28 | M12 search_idx 未归一化加权独立 1epoch | `c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e` | `Seraphic663/lrat-qwen3-0.6b-searchidx-full-20260728` / `5461393bba73194ff905977f8713bd5212f20320` | `versions/M12/submission_m12_20260728.jsonl` | 已评测；记录 `998769` | `Total 40.1；Recall 46.0；Success 20.5；AvgSteps 16.1；StepScore≈67.8` |
| 2026-07-29 | M10 early_stop_v1、LR `2e-6`、独立 1epoch | `cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984` | `Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729` / `9cdfdba51359ec65069a748cf8c00c55477016bb` | `versions/M10/submission_m10_20260729.jsonl` | 已评测；记录 `998853` | `Total 40.6；Recall 46.3；Success 21.1；AvgSteps 16.0` |

## 2026-07-29 M10 A榜提交与正式结果

```text
模型：M10 LRAT early_stop_v1、LR 2e-6、独立 1epoch
服务器训练目录：/root/data/LRAT/ccir/outputs/checkpoints/m10_lr2e6_full_epoch_20260729_seed20260716/
服务器冻结目录：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-lr2e6-full-20260729/
本地提交目录：/home/seraphic/xir/versions/M10/
权重 SHA-256：cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
权重 MD5：7a5b45a2784ba830213d9c50fc7648a9
HF repo：Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729
HF commit：9cdfdba51359ec65069a748cf8c00c55477016bb
HF 可见性：public；匿名 API 回读通过
远端 model.safetensors：2,383,139,480 bytes
远端 LFS SHA-256：cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
JSONL：versions/M10/submission_m10_20260729.jsonl
JSONL SHA-256：193df8cde1decf491e53d2f9e89f5bb0c212ff54894844864eb2e6e30dee9ed2
平台状态：已评测；记录 998853；提交时间 2026-07-29 23:36:07
```

M10 已完成 11,764/11,764 step、最终目录独立 dev1500、原子冻结、新 CPU 进程加载、本地白名单、公开 HF 上传和匿名远端回读。dev1500 为 R@1/R@5/R@10/MRR `0.630000/0.905333/0.974667/0.752016`；paired bootstrap 95% CI 跨零，因此本地指标没有提前证明稳定提升。正式 A榜记录 `998853` 为 `Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，相对 M01 为 `Total +0.5 / Recall +1.5 / Success +0.1 / AvgSteps +0.6`。M10 成为当前正式 A榜最佳。

## 2026-07-28 M12 A 榜提交与待评测状态

用户选择把已完整训练但 dev gate=false 的 M12 作为探索性 A 榜候选，并在网页完成提交。2026-07-28 用户提供 Daily Results 截图，确认正式记录如下：

```text
模型：M12 LRAT search_idx 未归一化加权独立 1epoch
服务器冻结目录：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-searchidx-full-20260728/
本地提交目录：/home/seraphic/xir/versions/M12/
权重 SHA-256：c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e
权重 MD5：9708a8653a78ec7e20c168e2ddda8483
HF repo：Seraphic663/lrat-qwen3-0.6b-searchidx-full-20260728
HF commit：5461393bba73194ff905977f8713bd5212f20320
HF 可见性：public；匿名 API 回读通过
远端 model.safetensors：2,383,139,480 bytes
远端 LFS SHA-256：c961c32c3b9f5284fd60524ec7e4ad3080232c5fe74e49b67a3e721e2b253a4e
JSONL：versions/M12/submission_m12_20260728.jsonl
JSONL SHA-256：89dc411d93f6e3524a738a4cdcf4ef51bfa293ad5634db8294f4c4da54174ba5
平台状态：已评测
提交时间：2026-07-27 17:18:15 UTC
平台记录：998769
正式成绩：Total 40.1 / Recall 46.0 / Success 20.5 / AvgSteps 16.1
StepScore：约 67.8；由展示 AvgSteps 按公式推算，不冒充平台显示字段
```

M12 与 M01 的展示 Total 均为 `40.1`，按展示字段计算的综合分也同为约 `40.16`。M12 相对 M01 为 `Recall +1.2 / Success -0.5 / AvgSteps +0.7`；同分时规则先比较 Success，因此 M01 仍排在 M12 前面。M12 证明 search_idx 加权提高了召回，但没有提高 A 榜综合排名；这不自动改变 7 月 29 日 B 榜登记模型。

2026-07-24 的 trajectory provenance A/B/C 短训中，B/C 均未通过预声明门槛，`full_epoch_authorized=false`；没有产生新正式模型、HF repo、JSONL 或平台提交记录，因此本表不新增 M08 行，M07 状态保持不变。

## 2026-07-26 候选与 B 榜准备

当前没有新增 A 榜可提交模型。M12 实际 artifact 已通过全量纠错审计，79,496/94,113 行按 stable `event.search_idx` 改权重，但未归一化导致总权重放大 `1.1444×`；它是待 dev1500 验证的实验候选，不是提交就绪模型，暂不分配正式模型 ID、不冻结、不上传 HF、不生成 A 榜 JSONL。

归一化 `trajectory_search_idx_v2` 已部署为 M12 后自动门控实验。自动流程先重新审计并独立评测 M12 最终模型，以固定 M07 dev1500 做 full-epoch paired gate，再执行 raw/v2 1,000-step、dev1500、10,000 次 paired bootstrap 和最多 3 seeds；只有多 seed 门槛通过才运行 v2 full epoch，且 full 结果还需通过相同 M07 paired gate才标记 development candidate。自动流程不读取 locked test、不冻结、不 HF、不提交，因此本注册表暂不新增模型或平台行。

M01 B 榜最终复现包已生成并关闭原始数据追溯门槛：

```text
目录：/root/data/LRAT/ccir/submissions/submission_B_m01_final_20260726/
ZIP：/root/data/LRAT/ccir/submissions/submission_B_m01_final_20260726.zip
ZIP bytes：2387140606
ZIP SHA-256：8172a899636d986e9997c1be1be331bd0543878a25a07f16e0a4ab9cd7b4c7fd
ZIP MD5：62afa7cb7c500402c267af32da85ece7
checkpoint SHA-256：b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501
checkpoint MD5：0e1c9c01fe0e49d133614130c2184161
submission source Git：f7656b174ca569ea2ac345b9af2bc2cf5c348451
trajectory provenance：96,366 stable / 138 ambiguous / 0 mismatch / 96,504 negative IDs traceable，SHA `0dd510fc...b690`
offline corpus：11,215,099 行，SHA `4d795938...e6393`
corpus links：1,989,015 references / 959,042 unique IDs / 0 missing / 0 text mismatch
结构/哈希/ZIP CRC/vendor/train dry-run/competition CPU inference：通过
合规状态：full_raw_traceability_verified
平台状态：未提交
```

该 ZIP 已是 M01 的最终 B 榜复现候选。7 月 29 日群公告已明确 A榜无需 MD5，旧的模型 MD5 登记计划取消；任何实际 B 榜上传仍需用户当次明确授权，并在上传前复核平台入口、最终 checkpoint、推理结果与 GitHub commit。

2026-07-28 冻结预检已完成。官方 `Yuqi-Zhou/LRAT main=274d0abb32ce24e4da96594a11ad683f533be9ba`、LRAT-Train 与比赛页面尚无 B 榜代码、测试数据、输出 schema 或 validator。准备实现提交为 `8114f2915b5ad9920fd26d8bbcf592470d75b11d`，带正确 PYTHONPATH 的 solution 全套 49 项测试通过。服务器 `ccir/submissions/b_m01_preflight_20260728.json` 的 SHA-256 为 `67ab026c74fc88e56159fcbb7fa1dbfca82d627268ebc82b770b8a43bf56fca8`；模型/ZIP SHA 与 MD5、41 文件 package validator、ZIP CRC、30 GiB 空间门槛均通过，登记值明确为模型 MD5 `0e1c9c01fe0e49d133614130c2184161`，平台写入为 false。服务器依赖版本与 requirements 逐项一致，CPU query/document smoke 与训练 dry-run 再次通过；训练 dry-run 记录 SHA 为 `48bda2f919d1221211d24d3287b3496d293121cf0d572a7bd30526719e25e854`。完整历史阶段门槛见 `/home/seraphic/xir/control/CCIR/B_LEADERBOARD.md`。

## 2026-07-30 B榜 GitHub 复现仓库

仓库：`https://github.com/seraphic663/ccir-lrat-retriever`

可见性：public

默认分支：`main`

远端 commit：`d9671c2a211c4454ef91c268d91a6c9970e2cf07`

远端内容：34 个 blob 与本地 tracked 文件逐项一致

仓库根目录 `code/` 现为 M10 默认复现流程，M01 移入 `profiles/m01/` 作为回退。M10 先要求官方 96,504 pairs 完成 trajectory/corpus 全追溯，再做固定 normalized-query SHA-256 分组并训练 94,113 条逐字节来自官方 pairs 的行；没有生成或额外构造 trajectory。仓库不包含训练数据、模型权重、checkpoint、缓存、日志、外部 API 输出或凭据。

验收：GitHub 页面匿名可见，README 已补充官方 B榜 JSONL、HF/GitHub 对应关系及最终 ZIP 树形结构。版本切换后的本地和服务器 4 项单测通过；服务器用真实 M00、官方 pairs、全追溯 manifest、M10 train 与 split manifest 完成 dry-run，compliance gate 为 `full_raw_traceability_and_query_disjoint_split_verified`，最终命令含 `LR=2e-6 / max_steps=11764 / master_port=29531`。

最新群公告明确：A 榜无需 MD5；B 榜 7 月 30–31 日提交并取最后一次，提交时需额外提供 GitHub 仓库链接。M10 已以正式 A榜 `40.6` 超过 M01 `40.1`，当前锁定为 B榜默认 checkpoint；B榜多个、最高约 300B Agent 带来新的 query/行为分布，M01 仍保留为回退。实际 B 推理与平台提交仍须按官方合同和用户当次授权执行。

## 2026-07-30 M10 B榜最终准备

```text
B榜 JSONL（2026-07-30旧准备稿，已被次日 canonical 文件替代）：submission_B_m10_20260730.jsonl
JSONL bytes：156
JSONL SHA-256：77c1599ded5236254cffb8522ecc920089fd473dca305b33a06985dec5d4e37b
date：2026-07-30
hf_repo_id：Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729
github_repo_url：https://github.com/seraphic663/ccir-lrat-retriever
GitHub commit：d9671c2a211c4454ef91c268d91a6c9970e2cf07
平台状态：已准备，尚未上传

最终复现目录：ccir/submissions/submission_B_m10_final_20260730/
文件数：43
最终 ZIP：ccir/submissions/submission_B_m10_final_20260730.zip
ZIP bytes：2,387,245,663
ZIP SHA-256：4e1d0655964a17bb8b5979944b4e919712cf62dbf1ecfe19f7202ee6b0b94047
ZIP MD5：308fd66c69906dc87e99117b2f8f0b02
checkpoint SHA-256：cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
checkpoint MD5：7a5b45a2784ba830213d9c50fc7648a9
```

最终目录包含 `code/preprocess.py / train.py / inference.py / config.yaml`、额外的 provenance/split 工具、全量 checkpoint、requirements、README、数据来源、合规说明、来源 evidence、模型身份、逐文件 manifest 和离线 validator。43 文件 validator、ZIP CRC、逐文件 SHA、checkpoint/HF/GitHub 身份、真实训练 dry-run 以及 package checkpoint 的 CPU query inference smoke 均通过；推理输出 1 行、1024 维、L2 norm `1.0000000457`。该 ZIP 是复现阶段交付物，不是当前平台要求上传的三字段 JSONL。

## 2026-07-31 M10 B榜 canonical 最终提交物

```text
B榜 JSONL：versions/M10/submission_B_m10_20260731.jsonl
Windows 上传副本：C:\Users\31572\Downloads\submission_B_m10_20260731.jsonl
JSONL bytes：156
JSONL SHA-256：02c271173ad6bf5e438972bf9e617c66e1625bb13f5a7086c1b6bcf0167b3f49
date：2026-07-31
hf_repo_id：Seraphic663/lrat-qwen3-0.6b-lr2e6-full-20260729
HF commit：9cdfdba51359ec65069a748cf8c00c55477016bb
github_repo_url：https://github.com/seraphic663/ccir-lrat-retriever
GitHub commit：31949f6f53722f06e91bd2ded6ec7f3a48037bba
平台状态：已准备，尚未上传

最终复现目录：ccir/submissions/submission_B_m10_final_20260731/
文件数：45
最终 ZIP：ccir/submissions/submission_B_m10_final_20260731.zip
ZIP bytes：2,387,249,735
ZIP SHA-256：c46f15d34a09dfe3108971ce56f837f1a23bbe39e35ee3a541bb609aea159984
ZIP MD5：cfeeda75d7f7458a147c159f06099ca6
checkpoint SHA-256：cea87cca521abd46c3e45ae53cd3b8f65b8216348c373fcc9698d69f133a9984
checkpoint MD5：7a5b45a2784ba830213d9c50fc7648a9
```

公开仓库已收敛为唯一 M10 复现路径：`code/reproduce_m10.py` 固定所有输入哈希、双卡拓扑和训练参数，README 明确独立复现约 10.5 小时、约 80GB 空间及“指标容差复现而非 checkpoint 逐字节复现”。2026-07-31 canonical dry-run 重新完成 trajectory/corpus 全追溯、query-disjoint split 和训练命令校验，状态 `dry_run_verified`。独立完整训练的 dev1500 为 R@1/R@5/R@10/MRR `0.628000/0.906000/0.974667/0.750895`，与提交模型的绝对差均不超过 `0.002`；复现权重 SHA 不同，因此只登记为功能/统计复现。两套 validator、ZIP CRC、逐文件 manifest、匿名 HF LFS SHA 和 GitHub main 回读均通过。

## M07 提交准备证据

```text
服务器冻结目录：/root/data/LRAT/ccir/models/Qwen3-Embedding-0.6B-LRAT-earlystop-full-20260723/
本地提交目录：/home/seraphic/xir/versions/M07/
HF repo：Seraphic663/lrat-qwen3-0.6b-earlystop-full-20260725
HF commit：1da2f3ec2dae197736f59f1fc45257e0c161e4a7
可见性：public
远端 model.safetensors：2,383,139,480 bytes
远端 LFS SHA-256：c937d61d57840467c5b8867559cc645704313e97ace7dafdeeece52652986a2b
JSONL：versions/M07/submission_m07_20260725.jsonl
平台状态：已评测；提交时间 2026-07-24 UTC；记录号 998718
正式成绩：Total 39.5 / Recall 45.0 / Success 19.8 / AvgSteps 15.9 / StepScore 68.2
```

M07 相对 M01：Total -0.6 / Recall +0.2 / Success -1.2 / AvgSteps +0.5。Recall 创队内新高但 Success 和 AvgSteps 退步，query-disjoint 路线未能超越全量数据路线。

## 未提交的旧诊断文件

`archive/diagnostics/submissions/submission_earlystop_20260722.jsonl` 指向 `Seraphic663/lrat-qwen3-0.6b-earlystop-20260721`。2026-07-23 认证 HF 查询返回 404；其本地模型 SHA 为 `15c937bbf2f0b038842f33272f658090c7bdd4e568541afc8a17b5b03e58f05c`，只是 1000-step 诊断模型，不是 M07，也不是最新完整 epoch。该 JSONL 仅保留作历史证据，绝对不可提交。正式 M07 只使用上表的新 repo、commit 和 `versions/M07/submission_m07_20260725.jsonl`；旧的 `archive/diagnostics/submissions/submission_m07_20260723.jsonl` 不得提交。

## M06 提交准备证据

```text
本地提交目录：/home/seraphic/xir/versions/M06/
HF repo：Seraphic663/lrat-qwen3-0.6b-3epoch-20260718
HF commit：b51ef2b1fb8a1bca671b551b030aa1b7f47c59f3
可见性：public（匿名 API 回读成功）
远端 model.safetensors：2,383,139,480 bytes
远端 LFS SHA-256：996189d7ee1580cb7c9eb1b2fa550b72c1afd806f7fbdc7e6ac7305bf2e6e1b5
JSONL SHA-256：f70638affe6657935d0fdba647808db6c108877ac6a56fa699728597350154b6
平台状态：已评测；提交时间 2026-07-18 08:31:42 UTC；记录号 998655
```

M06 相对 M01：`Total -1.4 / Recall -1.4 / Success -2.0 / AvgSteps +0.4`。M02 相对 M01：`Total -1.9 / Recall -2.1 / Success -2.7 / AvgSteps +0.1`；相对 M06：`Total -0.5 / Recall -0.7 / Success -0.7 / AvgSteps -0.3`。这确认第2轮已经退化；第3轮小幅回升但仍低于第1轮。M02/M06 榜单行的 StepScore 均按展示 AvgSteps 近似推算，不作为平台正式显示字段登记。

## M02 提交准备证据

```text
本地提交目录：/home/seraphic/xir/versions/M02/
HF repo：Seraphic663/lrat-qwen3-0.6b-2epoch-20260717
HF commit：28ab688b5e91d4e6bc61642950ec4bb354a73c23
可见性：public（匿名 API 回读成功）
远端 model.safetensors：2,383,139,480 bytes
远端 LFS SHA-256：1075292002c27eee654f83e1d1d568a78d4cc9238bc48a36d4ceb5f35968e050
JSONL SHA-256：3107b79f45197a1ac9ddac5257943817416d94757964aa9d14ac6cac3e4ded7b
平台状态：已评测；提交时间 2026-07-19 08:59:22 UTC
正式成绩：Total 38.2 / Recall 42.7 / Success 18.3 / AvgSteps 15.5 / StepScore≈69.0
```

## 正式对照基线与评测配置

```text
BaselineLRAT：Total 34.0 / Recall 36.1 / Success 15.5 / AvgSteps 16.8 / StepScore 66.5
Baseline Qwen3 Embedding：Total 28.5 / Recall 28.5 / Success 10.6 / AvgSteps 17.8 / StepScore 64.3
Agent model：Qwen3.5-4B
```

Agent model 是比赛检索流程中的 Agent，不是提交的 retriever。M01/M02/M06 的 retriever 基础模型始终是 `Qwen/Qwen3-Embedding-0.6B`。

## 论文 Data Flywheel 隔离声明

2026-07-29 的论文式 Data Flywheel 实现 `06c9d6c669bcbf5433f0ed95cc4d49badd6b7cf2` 与 execution-smoke 修正 `b5f6dec0301f6c8f06624d102e7dece04577d71e` 需要外部 InfoSeekQA query、Tongyi-DeepResearch-30B-A3B 与 Qwen3-30B-A3B-Thinking-2507 推理，明确不符合本届比赛允许数据和模型范围。该工作流及未来 research checkpoint 永不生成比赛 JSONL、不得上传比赛 HF repo 或平台，也不改变 M01 B 榜登记身份。synthetic smoke 输出已删除，当前没有任何 research 模型或外部提交记录。

## M01 榜单证据

```text
Team：DefaultGroup
Total：40.1
Recall：44.8
Success：21.0
AvgSteps：15.4
StepScore：69.1
SubmitTime：2026-07-17 10:40:36 UTC（UTC+8 为 2026-07-17 18:40:36）
```

同表 `BaselineLRAT` 为 `Total 34.0 / Recall 36.1 / Success 15.5 / AvgSteps 16.8 / StepScore 66.5`。后续提交以 M01 的 `40.1` 为正式参考，不用训练重叠 `dev500` 指标代替。

## A榜提交格式

比赛平台上传一行 JSONL，而不是本地模型目录或 checkpoint ZIP：

```jsonl
{"date":"YYYY-MM-DD","hf_repo_id":"用户名/模型仓库"}
```

模型必须位于 HF model repo 根目录，至少包含 `config.json` 和 `model.safetensors`；当前 M01 仓库公开，无需添加私有仓库协作者。

## 提交前检查

1. 用户明确选择 `REGISTRY.md` 中的冻结模型。
2. 用户确认当日额度尚未使用，并授权本次外部提交。
3. 本地冻结模型、服务器副本和目标 HF LFS SHA-256 一致。
4. 仓库根目录没有 optimizer、scheduler、嵌套 checkpoint、密码或数据。
5. 远端模型可见性与文件大小正确；私有仓库需按比赛要求授权。
6. JSONL 的日期使用实际提交日，不复用旧日期。
7. 提交后登记平台状态和完整榜单指标；截止后不再修改对应 HF 仓库。
