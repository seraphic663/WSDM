# CCIR / LRAT 控制文档区

> 导航 + 架构与控制（由原 README、ARCHITECTURE、CONTROL 合并，2026-08-04）。

## 目录
- [导航（README.md）](#readmemd)
- [架构（README.md）](#architecturemd)
- [控制（README.md）](#controlmd)

## README.md

# CCIR Cup 控制文档

这里是 CCIR/LRAT 的当前控制文档区。它不再保存模型、数据或训练输出；版本实体统一在 [`../../versions/README.md`](../../versions/README.md)，历史材料统一在 [`../../archive/README.md`](../../archive/README.md)。

> WSDM Short 范围说明：下文涉及 A/B 榜、HF 和外部提交的内容只用于历史追溯，不是当前论文流程；不要据此执行比赛提交。

## 推荐阅读顺序

1. [`README.md`](README.md)：当前结论、事实源分工和不可混淆边界。
2. [`OPERATIONS.md`](OPERATIONS.md)：训练、评测、冻结和复现实验的短版流程。
3. [`STATUS.md`](STATUS.md)、[`REGISTRY.md`](REGISTRY.md)、[`LOG.md`](LOG.md)：动态状态、模型身份和历史记录。
4. [`LOG.md`](LOG.md)：实验历史摘要；需要细节时再进入归档报告。

当前保留的控制文件只有事实源、模型版本记录和少量导航文档；旧实验说明、旧会话提示、详细报告和 workflow 不再混在这里。


---

## ARCHITECTURE.md

# CCIR / LRAT 项目架构与跨环境操作图

> 本文说明“东西在哪里、谁负责、数据如何流动、认证在哪里结束”。动态结果看 `STATUS.md`，模型身份看 `REGISTRY.md`，外部平台历史看 `REGISTRY.md`。

## 1. 整体拓扑

```text
用户 / Codex客户端
        │
        ▼
本地 WSL2：/home/seraphic/xir                 Hugging Face
控制面、规则、代码草稿、原始副本、备份、HF CLI ──上传──▶ Model Repo
        │ SSH / SCP / rsync                              │
        ▼                                                │ repo id
训练服务器：/root/data/LRAT                              ▼
uv环境、2×A40、训练、评测、checkpoint             外部平台历史记录
                                                   （不属于 WSDM Short 流程）
```

四个环境相互独立：

1. **Codex客户端/session**：只有聊天上下文；新session不继承旧聊天。
2. **本地WSL**：持久共享文件和个人认证所在位置，是控制面。
3. **训练服务器**：共享root计算面，只放项目运行环境，不登录个人Codex或HF账号。
4. **HF与比赛平台**：外部状态；HF存模型，比赛平台只接收指向HF的JSONL。

## 2. 本地 WSL 控制面

```text
根目录：/home/seraphic/xir
系统：WSL2 Linux
Codex启动目录：必须是 /home/seraphic/xir
HF CLI：/home/seraphic/xir/.download-env/bin/hf
HF CLI版本：1.23.0
当前HF账号：Seraphic663
```

关键目录：

```text
AGENTS.md                         本地唯一总纲；含本地私密SSH信息，不上传
control/CCIR/                             当前事实源、规则和状态
control/CCIR/README.md              本文
control/CCIR/STATUS.md                    动态快照
control/CCIR/REGISTRY.md            模型SHA和生命周期
control/CCIR/REGISTRY.md       HF与外部平台历史记录
XIR archive/history.md                 历史控制文档和旧会话材料
XIR archive/workflows.md               阶段操作手册归档
XIR archive/diagnostics/               未登记模型和实验诊断产物
archive/05-patches/                   服务器补丁归档
data/raw/LRAT-Train/                      官方训练数据本地副本
versions/M00/                             官方基础模型本地副本
versions/M01/、M02/...                    正式模型包；模型文件和提交JSONL直接位于Mxx下
solution/                                 与服务器同步的项目代码副本
LRAT/                                     官方仓库本地只读参考clone，不等同服务器工作树
```

本地根目录本身不是Git仓库；真正的比赛Git工作树在服务器。不得在本地 `LRAT/` clone 中误以为已经修改了服务器代码。

## 3. 训练服务器计算面

```text
Host：ssh-cn-huabei1.ebcloud.com:34988
用户：root
项目：/root/data/LRAT
分支：ccir/dev
origin：https://github.com/Yuqi-Zhou/LRAT.git
磁盘：/root/data，2026-07-23 核查时约1.0T总量、364G可用；实时值看 `STATUS.md`
GPU：2 × NVIDIA A40
```

环境：

```text
uv：/root/.local/bin/uv 0.11.28
虚拟环境：/root/data/LRAT/.venv
Python：3.10.16
torch：2.7.0+cu126
CUDA runtime：12.6
transformers：4.53.2
vllm：0.9.0.1
deepspeed：0.17.2
依赖检查：237 packages compatible
UV缓存：/root/data/uv_cache
HF缓存：/root/data/huggingface_cache
```

服务器目录分层：

```text
/root/data/LRAT/.git/             服务器唯一比赛Git历史
/root/data/LRAT/.venv/            uv管理环境，不进Git
/root/data/LRAT/FlagEmbedding/    官方训练实现及必要修复
/root/data/LRAT/solution/         CCIR脚本、工具与测试
/root/data/LRAT/ccir/data/        raw/processed/smoke/cache，不进Git
/root/data/LRAT/ccir/models/      base、冻结模型、插值模型，不进Git
/root/data/LRAT/ccir/outputs/     checkpoints/eval/logs，不进Git
/root/data/LRAT/ccir/meta/        审计和快照；逐文件决定是否进Git
/root/data/LRAT/ccir/reports/     实验报告
/root/data/LRAT/ccir/workflows/   与本地同步的操作手册
```

服务器不得安装或登录个人Codex/Claude Code，不保存个人HF token。HF上传从本地WSL完成。

## 4. 路径映射

| 内容 | 本地WSL | 服务器 |
|---|---|---|
| 官方训练pairs | `data/raw/LRAT-Train/LRAT-training-pairs.jsonl` | `ccir/data/raw/LRAT-training-pairs.jsonl` |
| 500条诊断集 | `data/processed/dev.jsonl` | `ccir/data/smoke/dev.jsonl` |
| 官方base | `versions/M00/` | `ccir/models/Qwen3-Embedding-0.6B/` |
| 1epoch冻结模型 | `versions/M01/` | `ccir/models/Qwen3-Embedding-0.6B-LRAT-1epoch-20260716/` |
| 项目代码 | `solution/` | `solution/` |
| 项目文档 | `control/CCIR/`、`docs/` | `ccir/` |
| 训练输出 | 一般不整目录下载 | `ccir/outputs/checkpoints/<RUN_ID>/` |
| 评测输出 | 必要时选择性同步 | `ccir/outputs/eval/`、`logs/` |

同步原则：先报源、目标、文件数和大小；不使用 `--delete`；大文件前确认；上传后检查大小、行数和SHA。不要同步整个clone、`.git`、`.venv`或缓存。

2026-07-18实测服务器没有安装 `rsync`，调用本地 `rsync` 会因远端找不到命令而在传输前失败。不要仅为同步文档或模型擅自安装服务器依赖；小文件使用 `scp` 精确列出文件，大目录使用 `scp -r` 复制到新建的空 staging 目录，或在确认工具可用后使用SFTP。所有回退方式仍须执行文件清单、大小和SHA复核。

## 5. 三类认证

### SSH

SSH凭据只在本地 `AGENTS.md`。`lrat-ebcloud` 当前已配置，但新环境仍应先检查：

```bash
ssh -G lrat-ebcloud | grep -E '^(hostname|user|port) '
```

正确结果应指向 `ssh-cn-huabei1.ebcloud.com`、`root`、`34988`。2026-07-23 当前代理下默认 sntrup KEX 可能卡住，优先增加 `-o KexAlgorithms=curve25519-sha256`。若 `ssh -O check` 显示 master 存活但 10 秒探针仍超时，按 `AGENTS.md` 先退出失效 master 再重建；长扫描/哈希使用非复用独立连接。别名不可用时采用交互式直连：

```bash
ssh -p 34988 root@ssh-cn-huabei1.ebcloud.com
```

别名配置完成后，优先建立ControlMaster：

```bash
ssh -MNf lrat-ebcloud
ssh -O check lrat-ebcloud
```

复用失效时允许交互式密码重连。不得把密码放进脚本、日志、Git、服务器或HF仓库；使用 `sshpass/expect` 前另行确认。

### Hugging Face

HF认证只在本地WSL：

```bash
cd /home/seraphic/xir
.download-env/bin/hf --version
.download-env/bin/hf auth whoami
.download-env/bin/hf auth login --force
```

浏览器device flow成功后token由HF CLI写入用户缓存；不得读取、打印、复制或提交token内容。只记录账号名、repo、commit和LFS SHA。当前已验证账号为 `Seraphic663`。

### 外部平台历史

比赛平台没有项目 CLI 或可由 Codex 调用的认证。历史记录可能包含网页上传的单行 JSONL；这些记录不构成 WSDM Short 的实验结果，HF 与平台状态也不在本次论文流程内。

## 6. Git架构

服务器 `ccir/dev` 是唯一 LRAT 开发历史。`origin/main` 是官方参考，禁止 push。提交使用单次身份，不设置共享 root 的全局 Git 配置。

```text
官方代码 origin/main
        │
        └── ccir/dev
              ├── FlagEmbedding必要训练修复
              ├── solution代码与测试
              ├── ccir文档/报告/workflow
              └── 不跟踪的数据、模型、输出
```

同一时刻只允许唯一主会话写服务器Git。长期已知遗留修改见 `STATUS.md`，不得用 `git add .`、`git commit -a` 或清理命令混入。

## 7. 数据—模型—提交生命周期

```text
官方HF base + 官方HF LRAT-Train
              │ SHA核验
              ▼
服务器只读输入
              │ smoke
              ▼
唯一RUN_ID训练目录（RUNNING/FAILED/COMPLETED）
              │ 完整性、哈希、新进程加载、评测
              ▼
服务器冻结推理副本
              │ MODEL_REGISTRY登记为候选
              ▼
用户选择
              │ 本地干净submission目录
              ▼
HF model repo根目录 + 远端LFS SHA
              │ 单行JSONL
              ▼
外部平台历史指标（不属于 WSDM Short 流程）
```

任何层级都不能用目录名代替SHA。`RUNNING`目录不能提交；HF repo不能包含训练数据、optimizer、scheduler、token或密码。

## 8. 当前代码入口

训练与恢复：

```text
solution/scripts/run_qwen3_dual_train.sh
solution/scripts/run_qwen3_dual_supervised.sh
solution/src/find_valid_checkpoint.py
```

评测与收尾：

```text
solution/src/evaluate_qwen3_pairs.py
solution/scripts/eval_completed_qwen3_run.sh
solution/scripts/wait_then_eval_qwen3_run.sh
```

数据与模型实验：

```text
solution/src/inspect_data.py
solution/src/prepare_data.py
solution/src/clean_query_conflicts.py
solution/src/interpolate_safetensors.py
solution/scripts/eval_qwen3_interpolation_grid.sh
```

对应测试位于 `solution/tests/`。修改代码时优先扩展现有入口和测试，不另写不可追踪的一次性脚本。

## 9. 失败分流

| 现象 | 归属 | 首要动作 |
|---|---|---|
| SSH DNS/复用失败 | 本地连接 | 检查socket与域名，交互重连，不碰训练 |
| `.venv/bin/pip`不存在 | 环境判断 | 检查uv绝对路径，不重建环境 |
| GPU忙/未知进程 | 服务器资源 | 只读识别所有者，不杀进程 |
| checkpoint中断 | 训练 | 保留FAILED和日志，用同配置监督器恢复 |
| 配置变化后续训被拒 | 训练保护 | 新建RUN_ID，不绕过校验 |
| HF上传SSL断线 | 历史外部模型上传 | 查远端 commit 与进程连接，复用分块缓存 |
| HF repo SHA不一致 | 历史外部模型核对 | 停止外部发布，定位源模型 |
| JSONL日期错误 | 历史外部记录 | 按实际日期核对，不复用旧日期 |
| 本地指标与外部历史不一致 | 评测 | 检查数据重叠与分布，不把外部平台差异当作论文结论 |

## 10. 唯一主会话的最小闭环

主会话每次接续工作必须做到：

1. 从本地根目录启动并读取总纲、架构、状态、模型注册表和当前阶段 workflow。
2. 先做职责范围内的只读检查，再陈述准备操作。
3. 明确 Git/GPU/HF/提交阶段，避免同一主会话中的重叠操作。
4. 只使用注册模型ID、实际路径和SHA。
5. 完成后更新相应状态文件，给出 commit、路径、哈希、异常、下一阶段和仍需用户授权的动作。

聊天历史只用于理解用户意图，不作为模型、数据、提交或服务器状态的最终证据。

---

## CONTROL.md

# XIR 控制总览

这是项目的单页控制入口。它负责告诉人“现在是什么状态、该看哪里、哪些身份不能混淆”；完整登记和历史证据仍保留在事实源和归档区中。

## 当前结论

- M10 是当前正式评测中最强的 LRAT 配置：`Total 40.6 / Recall 46.3 / Success 21.1 / AvgSteps 16.0`，作为 WSDM 研究主比较基线。
- M01 是正式参考基线：`Total 40.1 / Recall 44.8 / Success 21.0 / AvgSteps 15.4`。
- M12 已完成探索性评测：展示总分 `40.1`，Recall `46.0`，但 Success `20.5`，不替代 M10/M01。
- M02、M06、M07 的模型身份和历史成绩保留；其余版本是否进入论文由受控实验决定。

## 事实源分工

| 问题 | 唯一入口 |
|---|---|
| 当前动态状态 | [`STATUS.md`](STATUS.md) |
| 模型 SHA、训练来源、冻结状态 | [`REGISTRY.md`](REGISTRY.md) |
| HF、JSONL 和外部平台历史记录 | [`REGISTRY.md`](REGISTRY.md) |
| 时间线和完整变更证据 | [`LOG.md`](LOG.md) |
| 详细实验 | [`../../archive/experiments.md`](../../archive/experiments.md) |
| 训练/评测/复现实验操作 | [`OPERATIONS.md`](OPERATIONS.md)；历史合同见 [XIR archive/workflows.md](/home/seraphic/xir/archive/workflows.md) |
| 按 M 阅读 | [`../../versions/README.md`](../../versions/README.md) |

## 当前版本入口

模型、历史 JSONL 和版本身份已经集中到 [`../../versions/`](../../versions/)。模型权重在对应 M 文件夹根部；最终身份仍以 `model.safetensors` 的 SHA-256 为准。

## 不可混淆的边界

- 外部平台 JSONL、HF repo、GitHub commit 和复现 ZIP 是不同历史交付物，不是 WSDM Short 的核心证据。
- 独立 LRAT 复现仓库是外部身份，不能因为本地目录移动而重新生成、覆盖或推送；完整副本在 `/home/seraphic/xir/ccir-lrat-retriever`。
- `dev1500` 是 query-disjoint 研发诊断，不替代平台隐藏评测。
- 服务器 `/root/data/LRAT` 是训练事实源；本地 `/home/seraphic/xir` 是控制面和白名单副本。
