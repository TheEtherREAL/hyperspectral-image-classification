# 协作开发说明

## 1. 不提交的内容

- `.venv/` 与个人 IDE 配置；
- `data/raw/` 中的 `.mat` 原始数据；
- 新生成的大体积中间数据；
- checkpoint、运行日志和临时实验目录；
- 密钥、Token、个人路径或 `.env` 文件。

固定 Pavia split 和公开 PCA/LDA 小型冻结预处理状态属于复现契约，需要随仓库保留。

## 2. 开发前检查

```powershell
.\.venv\Scripts\python.exe scripts\检查运行环境.py
.\.venv\Scripts\python.exe -m pytest -q
```

协作者首次使用时应根据 `requirements.txt` 或 `requirements-lock.txt` 建立自己的环境，并按 `data/raw/README.md` 准备本地原始数据。不要把本机 `.venv` 或原始数据上传到仓库。

## 3. 分支与提交

- 每个功能使用独立分支，例如 `feature/hybridsn-model`；
- 一次提交只解决一个明确问题；
- 修改算法时同时增加或更新测试；
- 修改实验口径时同步更新 YAML、`docs/TASK_BOARD.md` 和 `docs/notes/DECISIONS.md`；
- 合并前运行完整测试，并说明新增配置、输出形状和验收结果。

## 4. 实验红线

- 不重新生成或覆盖 seed=1442 的固定划分；旧 seed 产物只作为历史记录保留；
- 标准化、PCA、LDA 等统计变换只在训练集拟合；
- 调参使用 `fair24_6_70` 的验证集，不使用测试集选择超参数；
- `paper30` 与 `fair24_6_70` 的结果不能混入同一公平比较表；
- 不把第三方 Keras 预训练模型、临时冒烟网络或外部论文数字写成本项目结果。

## 5. 结果进入论文前

正式结果至少应包含配置副本、随机种子、checkpoint、训练日志、OA、AA、Kappa、逐类准确率、混淆矩阵和分类图。论文、实施报告与答辩 PPT 必须引用同一份可追溯结果。

## 6. 首次发布与提交前检查

公共仓库应包含代码、测试、统一 YAML、核心文档、固定 split、小型冻结状态、数据来源清单和已审核的数据概览图；旧阶段工具、旧报告、原始数据和运行产物只保留在本地。

首次建立仓库时先检查暂存区，而不是直接推送：

```powershell
git init
git add .
git status --short
git diff --cached --stat
```

确认暂存区中没有 `.venv/`、`.mat/.zip` 原始数据、日志、checkpoint、`.env`、个人 IDE 配置和被 `.gitignore` 排除的本地历史目录。若边界不符合预期，应先修正 `.gitignore`，不要依赖推送后再删除。
