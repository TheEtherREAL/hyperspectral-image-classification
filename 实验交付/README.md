# 高光谱图像智能解译 —— 实验交付

本目录是高光谱图像分类大作业的**整理后的核心交付目录**，包含三个自包含 Jupyter Notebook（数据预处理、HybridSN 基线、传统分类方法对比）、对应的 YAML 配置、运行产物目录与实验报告。

> 实验报告（`report/实验报告.md`）按教师 Word 模板标题组织，覆盖**基本任务**（数据读取与预处理、PCA/LDA 降维、HybridSN 分类）与**进阶任务**（改进 HybridSN：BatchNorm + 残差连接 + 全局平均池化，以及与原始模型的控制变量对比）。改进模型定义见主仓库 `src/models/改进HybridSN.py`，对比脚本为 `scripts/运行HybridSN改进对比.py`。

## 目录结构

```
实验交付/
├── README.md                      # 本说明文档
├── configs/                       # 三阶段 YAML 配置（超参数集中管理，便于调参）
│   ├── stage1_data/               # 阶段一：数据读取 / 划分 / 预处理配置（3 套数据集）
│   ├── stage2_hybridsn/           # 阶段二：HybridSN 模型结构与训练配置（softmax/sigmoid 等）
│   └── stage3_traditional/        # 阶段三：传统分类方法配置
├── notebooks/                     # 三个自包含实验 Notebook（按顺序运行）
│   ├── 01_数据读取划分与预处理.ipynb
│   ├── 02_HybridSN基线训练验证测试.ipynb
│   └── 03_传统分类方法对比.ipynb
├── data/
│   └── splits/                    # 阶段一生成的划分（npz/json/csv）
├── outputs/                       # 运行产物（模型就绪数据、训练结果、指标与图）
│   ├── stage1/                    # 预处理后的模型就绪数据 + manifest
│   ├── stage2/                    # HybridSN 训练结果（checkpoint、曲线、混淆矩阵、分类图）
│   ├── stage3/                    # 传统方法结果
│   └── figures/                   # 汇总对比图
└── report/                        # 实验报告（Markdown）与报告插图
```

## 运行顺序与依赖

三个 Notebook 存在明确的先后依赖，请**按编号顺序**运行：

1. **`01_数据读取划分与预处理`**：读取原始 `.mat` 数据 → 分析数据集构成 → 生成划分 → 完成 PCA / LDA / 波段选择 / 原始四种预处理 → 输出模型就绪数据与 `manifest`。
2. **`02_HybridSN基线训练验证测试`**：读取阶段一的模型就绪数据 → 定义可配置 HybridSN → 训练 / 验证 / 测试 → 输出精度、学习曲线、混淆矩阵、分类图；支持 softmax / sigmoid 两种分类目标。
3. **`03_传统分类方法对比`**：读取阶段一的降维特征与划分 → 逐个测试 SVM / KNN / LDA / 随机森林 / XGBoost → 汇总对比。

### 数据来源

原始 `.mat` 文件（体积较大，未复制进本目录）位于：

```
g:/高光谱智能图像解译/hsi_project/data/raw/
├── PaviaU.mat / PaviaU_gt.mat                 # Pavia University（610×340×103，9 类）
├── Indian_pines_corrected.mat / Indian_pines_gt.mat  # Indian Pines（145×145×200，16 类）
└── Salinas_corrected.mat / Salinas_gt.mat      # Salinas（512×217×204，16 类）
```

如需更改数据路径，修改各 Notebook 顶部「0. 环境配置与路径验证」代码块中的 `RAW_DATA_DIR`。

### 运行环境

- Python 3.12，主要依赖：`numpy / scipy / scikit-learn / matplotlib / pyyaml / torch(>=2.0) / xgboost / joblib`。
- 建议使用已有的项目虚拟环境：

  ```
  g:/高光谱智能图像解译/hsi_project/.venv/Scripts/python.exe -m jupyter notebook
  ```

- 模型训练推荐 GPU（本机 `torch` 已带 CUDA 支持，`device` 会自动选择）。

## 关键约定

- **数据划分**：统一为 **24% 训练 / 6% 验证 / 70% 测试**（分层随机）。
- **无数据泄漏**：标准化、PCA、LDA、波段选择等统计量都**只在训练集上拟合**，再变换整幅图像。
- **核心超参数均以中文注释标明**，集中在各 Notebook 的配置代码块与 `configs/*.yaml` 中，方便直接调参。
