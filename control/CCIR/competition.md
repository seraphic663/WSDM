# 基于Agent交互轨迹的检索模型优化

**赛题简介**：

本赛题旨在探索从LLM Agent的执行轨迹中训练检索器的方法。参赛者需利用Agent交互轨迹数据，从轨迹中提取检索信号以训练检索器，使检索器能更精准地服务于Agent的任务执行需求。赛题限定检索器基础模型为Qwen3-Embedding-0.6B，训练数据仅可使用赛题官方提供的Agent轨迹与离线语料库，不允许引入任何外部数据。比赛使用BC-Plus基准进行评测，考察检索器召回率（Recall）、Agent端到端成功率（Success_Rate）及Agent平均执行轮次（Avg_Steps）。B榜阶段将更换Agent模型，以考察检索器的泛化能力。

**赛题背景**：

随着大语言模型的快速发展，基于LLM的智能Agent在信息检索、代码生成、工具调用等任务中得到广泛应用。Agent通常依赖检索器从外部知识库中获取相关上下文以辅助推理与决策。然而，现有检索器大多基于通用语义相似度任务或人类点击日志训练，与Agent实际使用场景存在分布差异：Agent没有位置偏差、浏览文档后存在推理痕迹、执行的是"搜索-浏览-推理"的多步迭代循环。因此，如何利用Agent执行过程中产生的轨迹信号来优化检索器，使检索器更好地适应Agent的工作模式，具有重要的研究价值。

**赛题任务**：

本赛题要求参赛者从官方提供的Agent轨迹数据中提取训练信号，对Qwen3-Embedding-0.6B检索器进行训练，提升检索器在BC-Plus基准上的实际表现。参赛者需解决以下关键问题：

（1）如何从Agent轨迹中识别有效的正负样本；

（2）如何在有限数据条件下充分挖掘训练信号；

（3）如何使检索器的优化最终体现为Agent端到端表现的提升；

（4）如何保证检索器对不同Agent的泛化能力。

### 大赛赛程

2026年07月06日-7月29日：开放初赛A榜提交，此时间段内选手可每天提交1次模型参数；

2026年07月20日24点：截止大赛报名；

2026年07月29日：参与B榜的团队提交模型MD5；

2026年07月30日-7月31日：开放初赛B榜提交，此时间段内选手可提交测试集推理结果，以选手当天提交的最后一次进行评测；

2026年8月01日-08月06日：此时间段内工作人员会对B榜TOP3进行复现（复现版本与前期提交MD5和成绩均一致），如遇弃权、作弊等行为将顺延名次；

2026年08月07日：晋级队伍进行决赛答辩；

2026年8月14日-8月16日：获奖队伍出席2026年CCIR大会，进行颁奖典礼。

### 参赛规则

1. 作品提交：在A榜阶段参赛者在竞赛平台可每天提交1次模型参数；
2. 赛段晋级：初赛阶段每道赛题经审核后前3支团队晋级决赛；
3. 公平竞技：参赛者禁止在指定考核技术能力的范围外，利用规则漏洞或技术漏洞等不良途径提高成绩排名；
4. 竞赛数据：参赛人员不得将数据用于任何商业用途。若做科研使用，请注明数据来源于相关数据提供单位；
5. 知识产权：参赛作品知识产权归出题单位、参赛者、官方竞赛平台三方共享。

### 更多信息请参考

论文：https://arxiv.org/abs/2604.04949
主页：https://yuqi-zhou.github.io/LRAT-homepage/

### 参赛交流

![118e18875e80159d77d0070ac5de023d.png](https://minio.xir.cn/competitions/uploads/admin/editor/2026-07-08/118e18875e80159d77d0070ac5de023d-642796.png)



## 数据与评测

### 数据简介

赛题使用的全部训练数据来源于LRAT（Learning to Retrieve from Agent Trajectories）论文提供的Agent轨迹与离线语料库，参赛者可通过以下链接获取：

- 数据集：https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train
- 代码仓库：https://github.com/Yuqi-Zhou/LRAT
  数据包含Agent在执行BC-Plus基准任务时的完整轨迹（多轮查询、检索、浏览、推理、行动记录及任务完成状态），以及对应的离线文档语料库。

### 数据说明

竞赛数据以JSON Lines格式存储。Agent轨迹数据每条记录包含task_id、task_description、turns（多轮交互列表）、final_status等字段。离线语料库每条文档包含doc_id、title和content字段。

允许使用的数据：组委会提供的Agent轨迹数据与离线语料库，以及Qwen3-Embedding-0.6B原始预训练权重。禁止使用的数据：任何形式的外部数据，包括但不限于其他检索数据集、额外训练语料、其他预训练模型参数或推理结果、外部 API生成的数据。参赛者可以自由设计轨迹数据的利用方式（如构造正负样本对、对比学习、排序学习等），但所有训练数据必须可追溯至组委会提供的原始数据。

### 提交要求

A榜阶段：每队可每天提交1次模型参数。参赛者仅需提交训练后的检索器模型checkpoint（基于Qwen3-Embedding-0.6B），无需提交训练代码。组委会将使用统一评测脚本加载checkpoint，在A榜Agent（Qwen3.5-4B）上运行BC-Plus基准评测。

B榜阶段：B榜截止后，参赛者需提交完整可复现的训练代码（包括数据处理、模型训练、推理的全部流程）及最终检索器checkpoint。提交形式为压缩包（.zip格式），需包含环境配置说明（requirements.txt或Dockerfile）。组委会将在独立环境中复现训练流程，并在B榜Agent（与A榜不同）上进行评测，B榜Agent与测试代码将于B榜结束后公布。

## 模型提交说明

### 一、上传模型

1. 在 Hugging Face 创建 Model 仓库。
2. 将全量模型上传到仓库根目录，至少包含 `config.json`、`model.safetensors`。
3. 若为私有仓库，请添加 HF 用户 **Yuqi-Zhou** 为 **Read（只读）** 协作者：https://huggingface.co/Yuqi-Zhou

### 二、提交文件

在提交入口上传 **`.jsonl`** 文件，每行一条 JSON，字段如下：

| 字段         | 类型   | 必填 | 说明                          |
| ------------ | ------ | ---- | ----------------------------- |
| `date`       | string | ✅    | 日期，格式 `YYYY-MM-DD`       |
| `hf_repo_id` | string | ✅    | 仓库 ID，格式 `用户名/仓库名` |

**示例 `submission.jsonl`：**

```jsonl
{"date": "2026-07-01", "hf_repo_id": "yourname/qwen3-0601"}
```

### 三、规则

- 只收全量参数，不收 LoRA / adapter。
- 权重须在 repo 根目录。
- 每日提交一个 jsonl 文件，一行一条记录。
- 截止后请勿修改对应仓库内容。

### 提交示例

A榜提交结构：

```text
submission_A.zip
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer.json
└── README.md # 简要说明训练方法
```

B榜提交结构：

`````
submission_B.zip
├── code/
│ ├── preprocess.py # 数据预处理
│ ├── train.py # 训练主脚本
│ ├── inference.py # 推理脚本
│ └── config.yaml # 训练配置
├── checkpoint/ # 最终检索器权重
├── requirements.txt # Python依赖
└── README.md # 环境配置、训练与推理说明
`````

### 评测标准

A榜与B榜均基于BC-Plus基准进行评测，包含三项核心指标：

1. 检索器召回率（Recall）：检索器返回结果对 BC-Plus 标注证据文档的覆盖比例。
2. Agent端到端成功率（Success_Rate）：配备参赛检索器的Agent在BC-Plus测试任务上的成功完成比例。
3. Agent平均执行轮次（Avg_Steps）：Agent完成单个任务所需的平均交互轮次，越低越好。
   最终排名以综合得分为依据：

综合得分 = 0.4 × Recall + 0.4 × Success_Rate + 0.2 × Step_Score
其中，Step_Score = 1 − Avg_Steps / Max_Steps，本次评测中 Max_Steps = 50。即Agent 平均执行轮次越少，Step_Score 越高；当 Avg_Steps 达到 50 时，Step_Score 为 0。

排名以综合得分从高到低排列。若综合得分相同，则依次以 Success_Rate、Recall、Avg_Steps 原始值作为次级排序依据。

A榜使用Qwen3.5-4B作为Agent模型，B榜使用不同的Agent模型（具体信息于B榜结束后公布）。两榜成绩独立排名。

已在 GitHub 开源基于 Qwen3.5 的 A 榜评测 pipeline，参赛者可参考 https://github.com/Yuqi-Zhou/LRAT/tree/main/xir_a_leaderboard_eval 中的评测代码，在本地或自有服务器部署检索器、运行端到端评测并复现提交前的评测流程。



## 常见问题

Q：是否可以使用赛题提供数据以外的数据进行检索器训练？

> A：不可以。仅允许使用组委会提供的Agent轨迹数据、离线语料库及Qwen3-Embedding-0.6B原始预训练权重。使用任何外部数据均视为违规。

Q：检索器模型是否必须使用Qwen3-Embedding-0.6B？

> A：是。检索器基础模型必须为Qwen3-Embedding-0.6B，参赛者可进行全量微调、LoRA、Adapter等任意微调方式，最终提交的checkpoint需能完整加载并独立运行推理。若使用LoRA等PEFT方法，建议提交合并后的完整权重。

Q: A榜提交频率的具体规则是什么？

> A：每队每天可提交1次模型参数，按自然天统计(00:00至23:59，UTC+8)。建议在本地充分验证后再提交。

Q: B榜更换Agent的具体情况？

> A：A榜使用Qwen3.5-4B作为Agent模型。B榜将更换为不同的Agent模型（可能涉及不同系列或规模），以考察检索器的泛化能力。具体模型信息于B榜结束后公布。

Q：数据集如何划分？

> A：训练集提供完整Agent轨迹数据，用于参赛者提取训练信号；测试集仅提供任务描述，用于评测检索器在Agent上的实际表现。训练集与测试集的任务分布相互独立，无重叠。

Q：提交的checkpoint是否支持适配器（LoRA/Adapter）权重？

> A：支持。参赛者可提交完整权重或适配器权重（含配置文件），但需确保组委会评测脚本能正确加载。README中需清晰说明加载方式。
