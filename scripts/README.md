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

# 4. 执行全部自动测试
.\.venv\Scripts\python.exe -m pytest
```

## 当前文件角色

| 文件 | 类型 | 作用 |
|---|---|---|
| `检查运行环境.py` | 只读 | 显示 Python、PyTorch、CUDA 和 GPU 状态 |
| `检查配置.py` | 只读 | 校验 YAML、路线名和冻结状态是否存在 |
| `运行数据预处理.py` | 只读 | 按七步构建 Dataset/DataLoader，不训练模型 |
| `验证固定划分.py` | 只读 | 独立检查 split 的覆盖、互斥和复现性 |
| `验证预处理状态.py` | 只读 | 独立重建并核对冻结预处理状态 |
| `生成预处理状态.py` | 写入 | 仅在新 PCA/patch 路线需要新状态时使用；默认拒绝覆盖 |
| `激活项目环境.ps1` | 环境入口 | 激活已有 `.venv`，不会重建环境 |

完整操作见根目录 `使用说明.md`；参数含义见 `configs/数据预处理/配置调参说明.md`。
