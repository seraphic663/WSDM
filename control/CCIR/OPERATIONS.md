# XIR 操作总览

这是日常操作的短版入口。旧训练、评测、审计和提交手册已从 WSDM 副本移出；如需历史合同，读取 [XIR archive/workflows.md](/home/seraphic/xir/archive/workflows.md)。

## 开始任何训练或评测前

1. 读 [`README.md`](README.md)、[`STATUS.md`](STATUS.md) 和 [`../../versions/`](../../versions/) 的统一模型登记。
2. 核对服务器 `/root/data/LRAT` 的 branch、HEAD、工作树、GPU、进程和磁盘。
3. 核对输入数据、基础模型、代码 commit 和实验 RUN_ID；不要把本地目录名当作模型身份。
4. 不触碰未知工作树修改，不杀未知进程，不把训练数据或凭据放进公开仓库。

## 训练与评测

- 训练、评测和审计的当前约束：[`STATUS.md`](STATUS.md)、[`LOG.md`](LOG.md) 和 `solution/experiments/`
- 实验源码仍在 `solution/experiments/`，版本包只保存正式模型、提交 JSONL 和 `manifest.json`。
- `locked test` 只在明确授权的最终评估使用，不能用于普通方案选择。

## 冻结与复现实验

- 结果冻结前必须独立加载、登记 SHA，并记录数据切分、代码 commit 和评测配置。
- 外部 HF/平台状态不纳入 WSDM Short 的核心证据；如需引用历史结果，必须单独核对发布权限。
- 外部平台结果不作为 WSDM Short 的核心证据；如需引用，必须单独核对数据和发布权限。

## 当前统一版验收

```bash
cd /home/seraphic/xir
python3 versions/verify.py
python3 verify_project_markdown_links.py
```

这些检查只读验证真实 M 路径、模型权重、实验路径、Markdown 链接和缓存残留，不修改外部仓库、HF、服务器或平台状态。
