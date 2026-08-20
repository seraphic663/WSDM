# XIR 项目入口

项目按四个入口理解：`control/` 管事实，`versions/` 管 M 版本总览，`shortpaper/` 管面向人的 WSDM Short 材料，`analysis/` 管可复核研究证据。训练代码、数据和两个独立 Git 仓库保持各自边界，不再复制到版本目录。

## 先看哪里

1. [`versions/README.md`](versions/README.md)：M00–M13 版本总览；当前优先看 M10。
2. [`control/README.md`](control/README.md)：当前状态、登记表和操作入口。
3. [`shortpaper/analysis.md`](shortpaper/analysis.md)：WSDM Short 内部分析、竞争边界、文献地图与迁移审计；[`shortpaper/draft.md`](shortpaper/draft.md) 是论文问题、方法和实验工作稿。
4. [`analysis/negative_reliability/README.md`](analysis/negative_reliability/README.md)：WSDM Stage 0/1 数据审计、PIVOT 门控、图表与复现入口；当前四页稿为 [`shortpaper/latex/demo.pdf`](shortpaper/latex/demo.pdf)。
5. CCIR 汇报、组会材料和历史 archive 仍在本地 WSDM 副本中，但不纳入这个根仓库。

## 一级目录

| 路径 | 用途 |
|---|---|
| `control/` | 当前状态、模型登记、项目日志和复现实验入口 |
| `versions/` | M00–M13 的 README 总览；模型文件和版本机器元数据只在本地 |
| `shortpaper/` | WSDM Short 研究材料 |
| `solution/` | 训练、评测和实验代码 |
| `data/` | 原始与处理数据的本地目录；根仓库只跟踪 README |
| `LRAT/` | 独立 LRAT origin clone；当前既有修改不清理，也不纳入 WSDM 根仓库 |
| `search_agent/` | 独立 search-agent origin clone；当前既有修改不清理，也不纳入 WSDM 根仓库 |
| `.download-env/` | 历史 Hugging Face 工具环境；WSDM Short 论文流程不依赖它 |

独立的 LRAT 复现仓库已从 WSDM 副本省略；完整副本仍在 `/home/seraphic/xir/ccir-lrat-retriever`。

根目录不再保留 `CCIR`、`server_patches`、`models`、`submissions` 的符号链接或兼容壳；正式版本总览统一在 `versions/README.md`，大体积诊断产物不纳入根仓库，完整历史在 `/home/seraphic/xir/`。

## 当前结论

- M10 是当前正式评测中最强的 LRAT 配置，作为 WSDM 研究的主比较基线。
- M01 是正式参考基线。
- WSDM negative-reliability audit 已恢复 638,215 个候选 occurrence；qrels-grounded 异质性成立，但 98.88% successful-visit mapping 未达到预注册 99% 门槛，因此当前结论是 PIVOT，不启动方法训练。
- `tmp/`、`.pyc`、`__pycache__` 和 `.cache` 已清除；旧 WIP 已归档。

## 验收

```bash
./solution/scripts/clean_python_cache.sh --dry-run
./solution/scripts/clean_python_cache.sh
```

清理脚本删除 Python 字节码和常见测试缓存；默认不删除通用 `.cache`，需要时显式加 `--include-cache`。完整本地 Markdown 链接检查脚本也保留在工作区，但不随这个根仓库上传。模型版本和 `versions/verify.py` 属于本地验收材料，不随这个不携带模型文件的根仓库上传。
