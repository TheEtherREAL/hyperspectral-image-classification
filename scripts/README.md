# Scripts 运行入口速查 / Entry-point Quick Reference

这里只保留当前阶段仍需使用的入口；阶段 0–2 的一次性生成和冒烟工具仅在原开发机器本地归档，不进入精简的公共仓库。

## 推荐顺序

```powershell
# 1. 检查已有环境
.\.venv\Scripts\python.exe scripts\检查运行环境.py

# 2. 校验准备使用的 YAML
.\.venv\Scripts\python.exe scripts\检查配置.py `
  --config "configs\数据预处理\Pavia数据预处理.yaml" `
  --require-state

# 3. 只读执行七步数据管线
.\.venv\Scripts\python.exe scripts\运行数据预处理.py

# 4. 运行 HybridSN baseline（会写入 experiments/）
.\.venv\Scripts\python.exe scripts\运行HybridSN基线.py

# 5. 执行全部自动测试
.\.venv\Scripts\python.exe -m pytest

# 6. 运行 seed=1442 的五种光谱预处理公平对比
.\.venv\Scripts\python.exe scripts\运行HybridSN预处理对比.py

# 7. 运行研究架构分组对比（PCA/LDA/选带、LBP/Gabor、分类器、HybridSN）
.\.venv\Scripts\python.exe scripts\运行研究架构分组对比.py
```

## 当前文件角色

| 文件 | 类型 | 作用 |
|---|---|---|
| `检查运行环境.py` | 只读 | 显示 Python、PyTorch、CUDA 和 GPU 状态 |
| `检查配置.py` | 只读 | 校验 YAML、路线名和冻结状态是否存在 |
| `运行数据预处理.py` | 只读 | 按七步构建 Dataset/DataLoader，不训练模型 |
| `运行HybridSN基线.py` | 写入 | 训练、推理、checkpoint、指标、性能、可视化和运行清单 |
| `运行HybridSN预处理对比.py` | 写入 | 在带 validation 的固定划分上公平比较 PCA、whitening 和直接波段路线 |
| `运行研究架构分组对比.py` | 写入 | 执行 7 个传统模型并合并 3 个 HybridSN 结果，生成四组指标、效率图和分类图 |
| `sync_reference_notebooks.py` | 写入 | 幂等同步两本参考对齐 Notebook 的展示、结果回放和报告素材导出章节 |
| `build_preprocessing_comparison_notebook.py` | 写入 | 生成只读回放正式结果的预处理对比 Notebook |
| `build_architecture_comparison_notebook.py` | 写入 | 生成只读回放 10 种方法结果的研究架构分组 Notebook |
| `验证固定划分.py` | 只读 | 独立检查 split 的覆盖、互斥和复现性 |
| `验证预处理状态.py` | 只读 | 独立重建并核对冻结预处理状态 |
| `生成固定划分.py` | 写入 | 为新 seed 生成 paper30/fair 两套 split；默认拒绝覆盖历史文件 |
| `生成模型就绪数据.py` | 写入 | 拟合或复用冻结状态，并生成训练脚本可直接读取的模型就绪 NPZ |
| `生成预处理状态.py` | 写入 | 仅在新 PCA/patch 路线需要新状态时使用；默认拒绝覆盖 |
| `激活项目环境.ps1` | 环境入口 | 激活已有 `.venv`，不会重建环境 |

完整操作见根目录 `使用说明.md`；参数含义见 `configs/数据预处理/配置调参说明.md`。
