# WSDM Short 工作副本说明

本目录是面向可能的 WSDM Short 投稿的精简工作副本，不是完整 XIR 项目。完整项目、历史材料、比赛提交材料和恢复副本位于：

```text
/home/seraphic/xir
```

## 当前工作重点

- `shortpaper/`：WSDM Short 工作稿。
- `analysis/`：研究问题、来源核查和实验分析。
- `data/`：LRAT 原始数据与处理后的实验数据。
- `solution/src/`、`solution/experiments/`、`solution/scripts/`：短论文研究和复现实验代码。
- `LRAT/`：官方 LRAT 参考代码。
- `report/`、`ai_core_ccir_defense_20260807/`、`control/CCIR/`：CCIR 报告、答辩和事实材料，作为 WSDM 写作参考保留。

## 本副本有意省略的内容

以下内容已从 WSDM 副本删除；需要时到 `/home/seraphic/xir` 查找：

- A 榜/B 榜 submission JSONL 及其哈希登记。
- `archive/diagnostics/submissions/` 下的历史提交文件。
- `solution/b/` 下的 B 榜提交、复现和预检代码。
- `control/CCIR/B_LEADERBOARD.md` 下的 B 榜专用操作手册。
- B 榜专用测试，以及旧的 M12 提交脚本。
- `LRAT/xir_a_leaderboard_eval/` 下的 A 榜评测代码。
- `report/CCIR_report/output/` 下的 LaTeX 编译中间文件和日志；保留可直接阅读的 PDF。
- `archive/diagnostics/models/` 下的旧 1000-step 诊断模型。
- `ccir-lrat-retriever/` 下的 B 榜公开复现仓库本地副本。
- WSDM `archive/` 中的旧 `history.md`、`workflows.md`、`research.md`、榜单诊断、竞赛备份、服务器快照和未运行的 Flywheel 代码包。

因此，`versions/verify.py` 在本副本中是只校验模型、实验和登记路径的精简版；它有意不校验省略的 submission 文件。完整提交材料仍在 `/home/seraphic/xir`。

## 边界

- 不要把 WSDM 副本中缺少的比赛文件重新猜测或重建；先检查 `/home/seraphic/xir`。
- 不修改或删除 `/home/seraphic/xir` 中的完整材料，除非用户另行明确授权。
- 数据、实验结果、报告和答辩材料是否删除，必须单独判断，不能因为它们来自 CCIR 就默认删除。
