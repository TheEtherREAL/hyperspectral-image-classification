<div align="center">

# 基于改进 HybridSN 的高光谱图像分类研究

### Reproducible Hyperspectral Image Classification Pipeline

面向高光谱图像分类任务的可复现、可扩展 PyTorch 工程，覆盖 **HybridSN 基线复现、模型改进、传统方法对比、论文复现（3D-CNN / 3D-1D-CNN）与多数据集验证** 的完整流程。

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1%20%7C%20CUDA%2013.0-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passed-2EA44F?logo=pytest&logoColor=white)
![Datasets](https://img.shields.io/badge/datasets-3-5B5BD6)

[快速开始](#快速开始) · [核心成果](#核心成果) · [实验内容](#实验内容) · [项目结构](#项目结构) · [主要结果](#主要结果)

</div>

<p align="center">
  <img src="results/dataset_overviews/pavia_university_overview.png" alt="Pavia University false-color image and ground-truth map" width="920">
</p>

## 项目简介

本项目完成高光谱图像（Hyperspectral Image, HSI）智能解译大作业的**全流程实验**：从高光谱数据读取、训练/验证/测试样本构建、PCA/LDA 降维，到 HybridSN 网络的搭建与训练，再到模型改进、传统机器学习方法对比、论文复现与多数据集验证。核心主线是「**改进 HybridSN**」——在原始 HybridSN 基础上引入 BatchNorm、残差连接与全局平均池化（GAP）分类头，以约 97% 的参数削减保持了 99.7% 以上的分类精度。

所有实验共享同一套数据基座与评价口径，保证结果可复现、可追溯、可横向比较。

## 核心成果

| 模块 | 成果 |
|---|---|
| **HybridSN 基线** | 在 Pavia University 上取得 **OA 99.9532%、AA 99.9405%、Kappa 0.999381**（4,844,793 参数，测试集仅 14 个像元误分） |
| **改进 HybridSN** | BatchNorm + 残差连接 + GAP，参数量 **4,844,793 → 152,073（−96.9%）**，OA 99.7328%、AA 99.3404%，模型体积从 18.48 MiB 降至 0.58 MiB |
| **传统方法对比** | 相同 PCA15 特征与划分下，SVM 最优仅 93.50% OA，HybridSN 领先约 **6.4 个百分点**，验证谱空联合建模的优势 |
| **论文复现** | 复现 Zhang et al. (2020) 的 3D-CNN（771,041 参数）与 3D-1D-CNN（214,561 参数），参数量与论文逐层对齐，Pavia OA 分别达 99.00%、98.77% |
| **多数据集验证** | 改进 HybridSN 在 Indian Pines / Pavia University / Salinas 上 OA 均超过 98.8%，跨城市与农业场景保持稳定 |

> **口径说明**：以上结果均来自固定划分 `fair24_6_70`（24% 训练 / 6% 验证 / 70% 测试）与固定随机种子 `seed=1442`。随机像元划分下测试 patch 与训练中心像元在空间上可能相邻，因此这些精度是「随机像元划分」口径的结果，不能直接解读为严格跨区域泛化能力；更严格的验证需采用空间块划分（见 [展望](#展望)）。

## 实验内容

本实验覆盖课程大作业的三级任务：

- **基本任务**：高光谱数据读取与预处理、PCA/LDA 降维、HybridSN 网络搭建与分类。
- **进阶任务**：改进 HybridSN（BN + 残差 + GAP）、与传统机器学习方法横向对比、提升分类精度。
- **提高任务**：复现 3D-CNN / 3D-1D-CNN 论文、多模型对比、多数据集验证、进一步模型创新（改进 3D-1D-CNN）。

### 数据集

| 数据集 | 空间尺寸 | 波段数 | 地物类别 | 有标签像元 | 场景 |
|---|---:|---:|---:|---:|---|
| Pavia University | 610 × 340 | 103 | 9 | 42,776 | 城市 |
| Indian Pines | 145 × 145 | 220（有效 200） | 16 | 10,249 | 农业 |
| Salinas | 512 × 217 | 204 | 16 | 54,129 | 农业 |

原始 `.mat` 数据不进入仓库，来源与校验信息见 [data/raw/README.md](data/raw/README.md) 与 [data/raw/SOURCES.csv](data/raw/SOURCES.csv)。

### 方法要点

- **数据划分**：分层随机划分 `fair24_6_70`，seed=1442，训练/验证/测试中心像元身份全模型共享。
- **预处理**：仅用训练集拟合「逐波段标准化 → PCA（15 维）」；复现模型按论文要求不做 PCA、使用原始波段。
- **HybridSN**：3D 卷积联合提取空间—光谱特征 → 重排 → 2D 卷积 → 全连接分类头，输入 `N×1×15×25×25`。
- **改进 HybridSN**：每个卷积后加 BatchNorm、每个 3D/2D 卷积块加残差连接、用 GAP 替换庞大的 `Flatten(18496)→256→128` 分类头。
- **复现模型**：3D-CNN（五层 3D 卷积 + FC）、3D-1D-CNN（3D 主干 + 两层 1D 卷积替代大 FC），输入 `N×1×B×11×11`，原始波段。

## 快速开始

### 1. 准备环境

```powershell
git clone https://github.com/TheEtherREAL/hyperspectral-image-classification.git
cd hyperspectral-image-classification

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

显卡或 CUDA 版本不同时，请按 PyTorch 官方方式选择适配本机的构建。

### 2. 准备原始数据

将原始 `.mat` 放入 `data/raw/`（不在仓库中，需自行下载）：

```text
PaviaU.mat            PaviaU_gt.mat
Indian_pines_corrected.mat   Indian_pines_gt.mat
Salinas_corrected.mat        Salinas_gt.mat
```

### 3. 验证环境与配置

```powershell
.\.venv\Scripts\python.exe scripts\检查运行环境.py
.\.venv\Scripts\python.exe scripts\检查配置.py --config "configs\数据预处理\Pavia数据预处理.yaml" --require-state
.\.venv\Scripts\python.exe -m pytest -q
```

### 4. 运行实验

| 实验 | 命令 |
|---|---|
| 数据预处理主流程 | `scripts\运行数据预处理.py` |
| HybridSN 基线 | `scripts\运行HybridSN基线.py` |
| 改进 HybridSN 对比 | `scripts\运行HybridSN改进对比.py` |
| 论文复现（单模型） | `scripts\运行论文复现.py --config "configs\模型训练\Paper复现_Pavia.yaml" --model paper3d1dcnn` |
| 全部对比（5 模型 × 3 数据集） | `scripts\运行全部对比.py` |
| 汇总对比表与图 | `scripts\汇总论文复现对比.py` |

也可直接打开 `notebooks/` 下的交付 Notebook（01 数据预处理、02 HybridSN 基线、03 传统方法对比、`paper_reproduction.ipynb` 论文复现），选择 `.venv` 内核逐格执行。

## 项目结构

```text
├─ configs/      实验配置 YAML（数据预处理 + 模型训练）
├─ data/         固定划分 splits、冻结预处理状态与数据来源说明（原始 .mat 不入库）
├─ src/          核心代码（模型 / 数据 / 训练 / 评价 / 可视化）
├─ scripts/      运行入口（检查 / 生成 / 运行 / 汇总）
├─ tests/        单元测试（数据管线、模型结构、参数量对齐、评价接口）
├─ notebooks/    交付 Notebook 与结构学习 / 对比实验 Notebook
└─ results/      结果图表（报告插图、论文复现对比图、数据概览图、数据集分布图）
```

## 主要结果

### 五模型对比（Pavia University，统一 `fair24_6_70 + seed1442`）

| 模型 | 参数量 | OA / % | AA / % | Kappa | 误分 |
|---|--:|--:|--:|--:|--:|
| 原始 HybridSN | 4,844,793 | **99.9532** | **99.9405** | **0.999381** | 14 |
| 改进 HybridSN | **152,073** | 99.7328 | 99.3404 | 0.996459 | 80 |
| 3D-CNN | 771,041 | 99.0048 | 98.3692 | 0.986806 | 298 |
| 3D-1D-CNN | 214,561 | 98.7744 | 97.9467 | 0.983756 | 367 |
| 改进 3D-1D-CNN | 201,993 | 98.0230 | 96.9256 | 0.973792 | 592 |

> 精度排序：原始 HybridSN > 改进 HybridSN > 3D-CNN > 3D-1D-CNN > 改进 3D-1D-CNN。改进 HybridSN 以约 97% 参数削减换取 0.22pp 的 OA 代价；复现模型在原始波段（不 PCA）与 11×11 邻域下仍达 98%–99%，验证了 3D 卷积谱空联合特征提取的有效性。

### 改进 HybridSN 多数据集验证

| 数据集 | OA / % | AA / % | Kappa |
|---|--:|--:|--:|
| Indian Pines | 98.8458 | 94.7670 | 0.986840 |
| Pavia University | **99.6996** | **99.3527** | **0.996018** |
| Salinas | 99.5120 | 99.3052 | 0.994565 |

改进 HybridSN 在三个数据集上 OA 均超过 98.8%。Indian Pines 的 AA（94.77%）低于 OA，源于 Oats、Stone-Steel-Towers 等少数类训练样本不足；这也说明在类别不平衡数据集上应结合 AA、逐类精度与混淆矩阵综合评判。

结果图见 [results/figures/](results/figures/)（报告插图）与 [results/论文复现结果/](results/论文复现结果/)（各模型混淆矩阵、学习曲线、分类图与汇总对比图）。

## 参考论文

- Roy S. K. et al. *HybridSN: Exploring 3D–2D CNN Feature Hierarchy for Hyperspectral Image Classification.* IEEE GRSL, 2019.
- Zhang B., Zhao L., Zhang X. *Three-dimensional convolutional neural network model for tree species classification using airborne hyperspectral images.* Remote Sensing of Environment, 2020.

## 展望

- 采用**空间块划分**与多随机种子重复实验，评估真实的跨区域泛化能力。
- 对少数类引入类别加权损失或数据增强，缓解类别不平衡下的精度回落。
- 在易混类上引入注意力机制或更细的谱段选择，进一步提升边界像元判别能力。
- 更系统地开展消融实验，分离 BatchNorm、残差连接与 GAP 各自的贡献。

---

<div align="center">

**当前成果：HybridSN 基线、改进 HybridSN（BN+残差+GAP）、传统方法对比、3D-CNN / 3D-1D-CNN 论文复现与三数据集验证均已闭环。**

</div>
