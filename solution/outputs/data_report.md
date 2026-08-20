# LRAT training-pairs 数据报告

生成时间：2026-07-14

## 数据完整性

输入文件：`data/raw/LRAT-Train/LRAT-training-pairs.jsonl`

- 文件大小：3,883,089,616 bytes
- JSONL 记录数：96,504
- 非法 JSON 行：0
- 字段：`query`, `pos`, `neg`, `pos_id`, `neg_id`, `reasoning_len`, `satisfied`, `reweight_rate`
- 空 query：1 条
- 空正样本：0 条
- 空负样本：0 条
- 正负 ID 同时出现：0 条

## 关键统计

| 指标 | 均值 | 中位数 | P95 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|
| 负样本数 | 19.61 | 9 | 58 | 5 | 407 |
| query 字符数 | 44.53 | 40 | 84 | 0 | 348 |
| 正样本文本字符数（合计） | 1,851.25 | 2,311 | 2,846 | 11 | 3,815 |
| 负样本文本字符数（合计） | 37,622.73 | 21,984.5 | 113,054.35 | 82 | 1,028,372 |
| reasoning_len | 175.24 | 137 | 458 | 0 | 8,336 |
| reweight_rate | 1.00 | 1.013 | 1.826 | 0.0305 | 2.026 |

全量数据中 `satisfied=true` 为 96,504 条，未发现 `false` 样本。因此当前数据文件不能用来直接比较“删除 satisfied=false”这一策略；该字段在此发布版本中已经完成了筛选。

标准化 query 去重后有 80,890 个 query；除去唯一的空 query 后为 80,889 个。重复记录数为 15,614。

## 长度说明

报告中的 token 数是空白分词近似值，不是 Qwen tokenizer 的真实 token 数。真实训练前仍需用模型 tokenizer 检查长度和截断比例。

## Smoke split

使用 seed `20260714`，标准化 query 后只保留每个 query 的第一条记录：

- `data/processed/train.jsonl`：5,000 条
- `data/processed/dev.jsonl`：500 条
- train/dev 标准化 query 重叠：0
- 两个 split 均无空 query

## 当前结论

数据结构正常，可以进入小规模训练准备。但训练配置不能假定每条记录固定 9 个负样本；应按记录实际读取 `neg`。由于当前下载环境没有 `torch`、`transformers`、`datasets` 或 `FlagEmbedding`，且未发现可用 NVIDIA GPU，小规模训练尚未执行。
