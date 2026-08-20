# WSDM Short 版本总览

这里是 WSDM Short 工作副本的版本入口。根仓库只保留这份人类可读的版本总览；模型权重、Tokenizer 文件、`registry.json` 和各版本 `manifest.json` 都是本地材料，不进入 WSDM 根 Git。

每个 `Mxx/` 是一个真实版本包。完整模型身份、登记证据和本地验收材料仍保留在当前工作区；需要完整模型或比赛提交材料时，去 `/home/seraphic/xir` 查找。

## 版本

| 版本 | 状态 | 结果/说明 |
|---|---|---|
| [M00](M00/) | 基础模型 | 原始 Qwen3，平台 baseline 28.5 |
| [M01](M01/) | 正式参考/回退 | 1 epoch；榜单记录和提交文件在 XIR 母本 |
| [M02](M02/) | 正式历史模型 | 2 epochs；榜单记录和提交文件在 XIR 母本 |
| [M03](M03/) | 诊断登记 | M01/M02 插值 alpha=0.25 |
| [M04](M04/) | 诊断登记 | M01/M02 插值 alpha=0.50 |
| [M05](M05/) | 诊断登记 | M01/M02 插值 alpha=0.75 |
| [M06](M06/) | 正式历史模型 | 3 epochs；榜单记录和提交文件在 XIR 母本 |
| [M07](M07/) | 正式历史模型 | early-stop full epoch；榜单记录和提交文件在 XIR 母本 |
| [M08](M08/) | 研究路线 | status weighting，未分配正式模型 |
| [M09](M09/) | 研究路线 | multi-positive，gate 未通过 |
| [M10](M10/) | 当前主版本 | early-stop full epoch，learning rate 2e-6；榜单记录和提交文件在 XIR 母本 |
| [M11](M11/) | 研究路线 | hard negatives，未分配正式模型 |
| [M12](M12/) | 探索性正式模型 | search_idx weighting；榜单记录和提交文件在 XIR 母本 |
| [M13](M13/) | 研究路线 | reasoning weighting，未分配正式模型 |

M08/M09/M10/M11/M12/M13 的实验源码仍在 `solution/experiments/`；模型和版本登记材料只在本地保留，不复制进 WSDM 根仓库。

## 当前事实源

- 版本的人类可读状态：本文件。
- 项目事实登记：[`../control/CCIR/REGISTRY.md`](../control/CCIR/REGISTRY.md)。
- 详细报告：本地 archive 材料，不纳入这个根仓库。
- 完整 submission 记录：`/home/seraphic/xir`。

## 本地验收边界

完整模型 SHA、版本 manifest 和 `versions/verify.py` 只用于当前本地工作区验收，不属于这个不携带模型文件的 WSDM private repo。原始模型和完整验收请在本地 WSDM/XIR 副本中进行。
