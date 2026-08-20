# WSDM Short Research Workspace

这是一个面向 WSDM 2027 Short 的私有研究工作仓库，不是完整的 XIR 项目，也不是比赛提交仓库。完整项目、历史材料、服务器材料和恢复副本仍位于 `/home/seraphic/xir`。

当前根仓库远端是 [seraphic663/WSDM](https://github.com/seraphic663/WSDM)，默认分支为 `main`。根仓库只管理论文、研究分析、实验代码、事实记录和轻量说明；原始数据、模型权重、缓存、编译产物和两个独立 Git 仓库留在本地，不进入根仓库历史。

## 当前研究状态

截至 2026-08-20，WSDM Short 的 negative-reliability audit 已完成 Stage 0、Stage 1、Stage 2 对齐和 Stage 3 离线 dry run。

- 4,980 条轨迹映射到 830 个 BrowseComp-Plus 问题，恢复 638,215 个 candidate occurrences。
- 未访问候选的 evidence-qrel rate 为 13.77%；later-visited 与未 later-visited 的比例为 66.69% 与 12.14%。
- successful-visit mapping 为 98.88%，低于预注册的 99% coverage threshold。
- 联合门控结果为 **PIVOT**：不启动 retriever intervention training，不宣称 method gain 或机制证明。
- Stage 3 的 random、exposure-prioritized 和 later-visit 三个训练臂只是未训练的公平性 dry run。

当前论文材料是诊断优先的 WSDM Short 工作稿，而不是最终匿名投稿版本。作者可见的四页 ACM 草稿位于 [`shortpaper/latex/demo.pdf`](shortpaper/latex/demo.pdf)。

## 建议阅读顺序

1. [`shortpaper/analysis.md`](shortpaper/analysis.md)：内部研究问题、证据边界、竞争风险、venue 和迁移审计。
2. [`shortpaper/draft.md`](shortpaper/draft.md)：论文问题、方法、假设、实验设计和当前 claim ledger。
3. [`analysis/negative_reliability/README.md`](analysis/negative_reliability/README.md)：Stage 0/1 结果、PIVOT 门控、图表、manifest 和复现命令。
4. [`shortpaper/latex/README.md`](shortpaper/latex/README.md)：ACM `sigconf` 模板、匿名 review mode 和当前四页草稿说明。
5. [`control/README.md`](control/README.md)：项目状态、登记表和事实入口。
6. [`versions/README.md`](versions/README.md)：M00–M13 的人类可读版本总览。

## 根仓库内容边界

| 路径 | 根仓库状态 | 说明 |
|---|---|---|
| `shortpaper/` | 跟踪 | WSDM Short 内部分析、工作稿、LaTeX 源码和当前 PDF |
| `analysis/` | 跟踪 | 可复核的研究分析、摘要、图表和审计证据 |
| `solution/` | 跟踪 | 分析、训练、评测和测试代码；不含大型数据与模型 |
| `control/` | 跟踪 | 项目状态、事实登记、操作说明和实验边界 |
| `versions/README.md` | 跟踪 | 版本的人类可读总览 |
| `data/README.md` 及数据卡片 | 跟踪 | 数据来源、边界和复现说明；不跟踪数据 payload |
| `LRAT/` | 独立本地 Git | LRAT 官方 origin clone，不纳入 WSDM 根仓库 |
| `search_agent/` | 独立本地 Git | 师兄 Diagnosing Search Behavior 使用的 origin clone，不纳入 WSDM 根仓库 |

## 两个独立 origin 仓库

这两个目录物理上位于 WSDM 下面，但 Git 历史彼此隔离，也不作为 WSDM 子模块或普通文件上传。

- `LRAT/` 的 origin 是 [`Yuqi-Zhou/LRAT`](https://github.com/Yuqi-Zhou/LRAT.git)，用于 LRAT 官方代码、训练数据语义和检索器基线。
- `search_agent/` 的 origin 是 [`liuqi6777/search_agent`](https://github.com/liuqi6777/search_agent.git)，用于师兄论文的 ReAct harness、工具接口和轨迹处理说明。

根仓库中的 `git status` 不应进入这两个目录；需要更新它们时，必须在各自目录内单独检查分支、origin 和本地修改。当前 WSDM 根仓库不会替它们 pull、reset、commit 或 push。

## 参考资料与外部资源

下面按两条研究来源链组织。WSDM 根仓库只记录入口和分析代码，不镜像这些外部数据、模型或索引；外部资源的许可证和原始数据条款仍然适用。

### LRAT：Learning to Retrieve from Agent Trajectories

LRAT 是当前 WSDM 工作中的检索器基线和训练数据来源。它研究如何从多步 Agent search/browse trajectory 中构造 retriever supervision，并将其转化为训练 pairs。

- 论文：[Learning to Retrieve from Agent Trajectories](https://arxiv.org/abs/2604.04949)。
- 官方 GitHub：[Yuqi-Zhou/LRAT](https://github.com/Yuqi-Zhou/LRAT)。
- 项目主页：[LRAT Homepage](https://yuqi-zhou.github.io/LRAT-homepage/)。
- 官方 Hugging Face 资源集合：[Yuqi-Zhou/LRAT](https://huggingface.co/collections/Yuqi-Zhou/lrat)。
- 训练数据：[Yuqi-Zhou/LRAT-Train](https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train)。
- Qwen3-Embedding 模型：[LRAT-Qwen3-Embedding-0.6B](https://huggingface.co/Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B)。
- multilingual-e5 模型：[LRAT-multilingual-e5-large](https://huggingface.co/Yuqi-Zhou/LRAT-multilingual-e5-large)。

当前本地 `LRAT/` 是该官方 GitHub 仓库的独立工作副本；LRAT training pairs、trajectories、模型和 qrels 不属于 WSDM 根仓库提交内容。

### Diagnosing：Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents

Diagnosing 是师兄的独立研究来源，也是当前 WSDM negative-reliability audit 的外部轨迹和 measurement framework 来源。WSDM 使用其公开 trajectory release 与 BrowseComp-Plus qrels 做自己的重分析，不复用其未公开的论文指标分析代码。

- 论文：[Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents](https://arxiv.org/abs/2608.01913)。
- 官方 GitHub：[liuqi6777/search_agent](https://github.com/liuqi6777/search_agent)。
- 方法与复现说明：[diagnosing-search-behavior.md](https://github.com/liuqi6777/search_agent/blob/main/docs/diagnosing-search-behavior.md)。
- 完整轨迹数据：[bcp_search_agent_trajectory](https://huggingface.co/datasets/liuqi6777/bcp_search_agent_trajectory)。
- BrowseComp-Plus 预构建索引：[Tevatron/browsecomp-plus-indexes](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes)。

当前本地 `search_agent/` 是该官方 GitHub 仓库的独立工作副本；六个 Agent 的完整 trajectory JSONL 位于本地-only 的 `data/raw/bcp_search_agent_trajectory/`，不上传到 WSDM 根仓库。

### 共同依赖：BrowseComp-Plus

两条研究线都涉及 BrowseComp-Plus，但用途不同：LRAT 将其作为外部 benchmark 评测，Diagnosing 使用其 document-level evidence/gold qrels 做 trajectory-level diagnosis。

- 官方 GitHub：[texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)。
- 论文：[BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent](https://arxiv.org/abs/2508.06600)。
- 查询与标注数据：[Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus)。
- 固定语料：[Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)。

当前 WSDM 的 Stage 0/1 audit 只把 BrowseComp-Plus qrels 作为 relevance judgment；它不把 BrowseComp qrels 转移成 LRAT 的 row-level training labels，也不把 later visit 直接等同于 false negative。

## 数据边界

根仓库不上传任何数据 payload。具体边界和公开来源见 [`data/README.md`](data/README.md)。本地-only 内容包括：

- `data/raw/bcp_search_agent_trajectory/` 下六个完整 Agent trajectory JSONL；
- `data/raw/LRAT-Train/` 下的 LRAT training pairs 和 trajectory archive；
- `data/processed/` 下的 train/dev/test split JSONL；
- BrowseComp-Plus corpus、qrels、预构建索引和其他下载资产；
- `versions/M*/` 下的 model、Tokenizer 和机器登记文件。

因此，根仓库 clone 后不能直接运行完整 audit；需要先按 `data/README.md` 恢复本地数据，并把独立的 `LRAT/` 和 `search_agent/` 工作副本放到约定位置。分析报告中的路径和命令描述的是完整本地工作区，不是假设 GitHub clone 自带数据。

## 版本边界

`versions/README.md` 是根仓库唯一跟踪的版本文件。`registry.json`、各个 `M*/manifest.json`、`model.safetensors`、`tokenizer.json`、`vocab.json` 和 `merges.txt` 仍保留在本地，但不进入 WSDM 根仓库；完整模型身份和机器验收在本地工作区或 XIR 母本中完成。

## 本地验证

在 `/home/seraphic/WSDM` 下运行：

```bash
./solution/scripts/clean_python_cache.sh --dry-run
./solution/scripts/clean_python_cache.sh
```

Markdown 链接检查脚本和完整模型验收脚本属于本地工作区工具，不随根仓库上传。清理脚本默认处理 Python 字节码和常见测试缓存，不删除原始数据、模型、独立仓库或研究证据。

## Git 工作规则

- WSDM 根仓库使用 `main`；两个嵌套仓库使用各自的 Git 状态和 origin。
- 根仓库只按明确路径暂存，例如 `git add -- shortpaper analysis solution control`；不要使用 `git add .` 或 `git add -A`。
- 原始数据、模型、缓存、编译产物、服务器认证材料、组会材料、CCIR 历史 archive 和 XIR 完整材料不属于根仓库提交范围。
- 在 WSL 工作区内使用 WSL Git；Windows Git 通过 UNC 路径可能把跨系统文件属性误报为 `solution/` 修改。
- private repo 只提供代码和研究材料的访问控制，不改变第三方数据集、语料、qrels 或模型的原始许可边界。

## 安全边界

`shortpaper/server.md`、临时沟通稿和临时图片仅保留在本地；服务器密码、API key、token 和个人认证信息不得加入 Git。发现凭据时应先从暂存区和提交历史中排除，再另行决定是否轮换或清理本地文件。

## 与 XIR 的关系

WSDM 是面向论文的派生副本，XIR 是完整事实源和恢复源。不要在 WSDM 中猜测缺失的比赛提交文件、服务器快照、模型或历史目录；需要这些材料时先回到 `/home/seraphic/xir`，并保持两棵工作树边界清晰。
