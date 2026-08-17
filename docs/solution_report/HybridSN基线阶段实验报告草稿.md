# 本科实验报告

**实验名称：** 大作业5：高光谱智能解译  
**课程名称：** 人工智能基础理论与技术 / 人工智能导论  
**实验时间：** 课后  
**任课教师：** 刘欢  
**助教：** 阮子航  
**实验类型：** 综合设计  
**姓名（组长）：** 待填写  
**学号（组长）：** 待填写  
**学院（组长）：** 待填写  
**组员：** 待填写  
**组号：** 待填写  

> 文档状态：HybridSN baseline 阶段草稿（2026-08-17）。标题顺序严格沿用教师《实验报告-大作业-组号(1).doc》；最终提交时应把本文内容填入原 Word 模板，并补充组员信息、过程截图、分工权重和个人心得。

# 实验目的

## （一）基本任务

1. 复用已经冻结的 Pavia University 数据读取、固定划分、训练集标准化和 PCA 降维结果，验证数据处理产物能够直接服务于模型训练与推理。
2. 使用 PyTorch 实现并训练 HybridSN，通过 3D 卷积联合学习光谱—空间特征，再通过 2D 卷积与全连接层完成九类地物分类。
3. 建立可重复执行的 baseline：固定配置、随机种子和 checkpoint 选择规则，输出 OA、AA、Kappa、逐类准确率、混淆矩阵、训练曲线和分类图。
4. 统计参数量、训练时间、推理时间、吞吐率和显存占用，为后续 HybridSN 优化、消融和其他模型对比建立参照。

## （二）进阶任务的衔接

本阶段不修改原始 HybridSN，而是冻结可复用的训练—推理—评价接口。后续改进模型必须复用相同数据身份、预处理指纹和评价代码，在带 validation 的公平比较配置上选择超参数，并报告多随机种子的均值与标准差。

# 实验环境

## 1. 数据

- 数据集：Pavia University。
- 原始影像：610×340 像元，103 个光谱波段；有标签像元 42,776 个，包含 9 类地物。
- 固定划分：`paper30`，split seed=1442；训练集 12,832 个，测试集 29,944 个，无验证集。
- 光谱处理：逐波段标准化后 PCA 降至 15 维，`whiten=false`；标准化与 PCA 均只在训练中心像元上拟合。
- 空间表示：25×25 邻域块，边界使用零填充，中心像元标签作为监督信号。
- 模型就绪产物：`data/processed/pavia_university/paper30__seed1442__standard_pca15_patch25/model_ready_dataset.npz`。
- 预处理指纹：`884942048935b51bf385e1d6c124eee59ea96bb04ac1470ac12ed783131dd61e`。

## 2. 软件及深度学习环境

| 项目 | 本次正式运行 |
|---|---|
| 操作系统 | Windows 11 10.0.26100 |
| Python | 3.12.13（项目 `.venv`） |
| PyTorch | 2.12.1+cu130 |
| NumPy | 2.5.2 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| CUDA 构建版本 | 13.0 |
| 训练精度 | float32 |
| 自动测试 | 77 passed |

系统默认 Anaconda 环境存在 NumPy 2 与旧二进制扩展不兼容，因此本实验只使用项目 `.venv`，避免环境混用导致结果不可复现。

## 3. 实验程序与附件

- 模型：`src/models/HybridSN模型.py`。
- 可复用训练、推理与 checkpoint：`src/training/hybridsn_baseline.py`。
- 指标和空间审计：`src/evaluation/classification_metrics.py`。
- 结果图：`src/visualization/hybridsn_results.py`。
- 一键入口：`scripts/运行HybridSN基线.py`。
- 结构学习 Notebook：`notebooks/HybridSN模型结构学习.ipynb`。
- 正式运行目录：`experiments/pavia_university__hybridsn__seed1442__20260817-1251/`。

# 实验原理

## 1. 高光谱分类与谱空联合建模

高光谱像元包含连续波段反射信息，同类地物通常具有相似光谱，但不同类别可能出现“同物异谱”和“异物同谱”。只使用中心像元光谱会忽略道路形状、建筑边缘和地块纹理，因此本实验以中心像元周围的 25×25 邻域作为输入，让模型同时观察光谱和空间上下文。

## 2. 训练集标准化与 PCA

对训练中心像元的第 (b) 个波段计算均值 \(\mu_b\) 和标准差 \(\sigma_b\)，将光谱标准化为

\[
z_b=\frac{x_b-\mu_b}{\sigma_b}.
\]

PCA 在训练标准化光谱上求解主方向，将 103 维光谱映射为 15 维。验证或测试数据只使用同一组 \(\mu_b\)、\(\sigma_b\) 和 PCA 投影执行变换，不重新拟合。这样能防止测试统计量进入预处理参数。

## 3. HybridSN 结构

PyTorch 3D 卷积输入采用 `N×C×D×H×W`。本实验的输入是 `N×1×15×25×25`，依次通过：

| 层 | 核/通道 | 输出 shape（batch=N） |
|---|---|---|
| Conv3D-1 + ReLU | 1→8，(7,3,3) | N×8×9×23×23 |
| Conv3D-2 + ReLU | 8→16，(5,3,3) | N×16×5×21×21 |
| Conv3D-3 + ReLU | 16→32，(3,3,3) | N×32×3×19×19 |
| 3D→2D reshape | 32×3→96 | N×96×19×19 |
| Conv2D + ReLU | 96→64，3×3 | N×64×17×17 |
| Flatten | 64×17×17 | N×18,496 |
| 全连接分类头 | 18,496→256→128→9 | N×9 logits |

两个隐藏全连接层后使用 Dropout 0.4。模型不使用池化、BatchNorm 和数据增强，共有 4,844,793 个可训练参数。输出是 logits，不在模型内部添加 Softmax，因为 `CrossEntropyLoss` 已包含 log-softmax 与负对数似然计算。

## 4. 评价指标

- OA（Overall Accuracy）：全部测试样本中预测正确的比例。
- 每类准确率：混淆矩阵第 (i) 类对角元素除以该类测试样本数。
- AA（Average Accuracy）：九个类别准确率的算术平均，降低大类样本对 OA 的支配。
- Cohen's Kappa：对随机一致性进行校正，

\[
\kappa=\frac{p_o-p_e}{1-p_e},
\]

其中 \(p_o\) 为观察一致率，\(p_e\) 由混淆矩阵行列边际概率计算。

# 实验内容和结果分析

## 1. 实验过程记录

### 1.1 运行前验收

先使用项目环境运行全仓测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最终自动测试结果为 `77 passed`，覆盖数据读取、固定划分、无泄漏预处理、模型逐层 shape、参数量、有限梯度、分类指标、空间回填和训练/推理 traceability。

### 1.2 封闭测速

在不访问测试集的条件下运行 1 epoch：

```powershell
.\.venv\Scripts\python.exe scripts\运行HybridSN基线.py `
  --epochs 1 --skip-test `
  --output-dir artifacts\smoke_hybridsn_20260817
```

该封闭运行只用于确认 GPU、DataLoader、反向传播和 checkpoint 链路，不作为正式分类结果；种子统一后，以正式 seed=1442 运行记录为准。

### 1.3 正式训练与一次性测试

```powershell
.\.venv\Scripts\python.exe scripts\运行HybridSN基线.py `
  --output-dir experiments\pavia_university__hybridsn__seed1442__20260817-1251
```

固定训练参数如下：

| 参数 | 数值 |
|---|---:|
| 训练 seed | 1442 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| weight decay | 0 |
| batch size | 256 |
| epochs | 100 |
| checkpoint 间隔 | 10 epochs |
| 最终模型选择 | 固定第 100 epoch |

`paper30` 不包含 validation，因此不能依靠验证集早停；本实验也没有使用 test 选择最佳 epoch。训练 100 轮后固定 `checkpoint_final.pt`，再对 test 评测一次。

## 2. 收敛过程

首轮 train loss 为 0.543229、train accuracy 为 82.06%；第 2、3 轮 loss 分别降至 0.064950 和 0.031088。第 40 轮首次达到 train accuracy=100%，末轮 train loss=0.003414、train accuracy=99.95%。训练中存在若干短时损失波动，随后恢复；这与固定学习率下 Adam 更新和 Dropout 随机性一致。报告保留完整曲线，没有挑选局部最优训练轮次。

![训练损失和训练准确率](../../experiments/pavia_university__hybridsn__seed1442__20260817-1251/loss_curve.png)

## 3. 分类准确性

| 指标 | 结果 |
|---|---:|
| 测试样本数 | 29,944 |
| 测试损失 | 0.004360 |
| OA | 99.9532% |
| AA | 99.8790% |
| Cohen's Kappa | 0.999380 |
| 误分样本数 | 14 |

逐类结果如下：

| 类别 | 测试样本 | 正确数 | 准确率 |
|---|---:|---:|---:|
| Asphalt | 4,642 | 4,640 | 99.9569% |
| Meadows | 13,055 | 13,055 | 100.0000% |
| Gravel | 1,469 | 1,466 | 99.7958% |
| Trees | 2,145 | 2,143 | 99.9068% |
| Painted metal sheets | 942 | 940 | 99.7877% |
| Bare Soil | 3,520 | 3,520 | 100.0000% |
| Bitumen | 931 | 926 | 99.4629% |
| Self-Blocking Bricks | 2,577 | 2,577 | 100.0000% |
| Shadows | 663 | 663 | 100.0000% |

![测试集混淆矩阵](../../experiments/pavia_university__hybridsn__seed1442__20260817-1251/confusion_matrix.png)

![逐类测试准确率](../../experiments/pavia_university__hybridsn__seed1442__20260817-1251/per_class_accuracy.png)

## 4. 误差分析

本次共误分 14 个测试像元：

1. 2 个 Asphalt 被预测为 Self-Blocking Bricks；
2. 3 个 Gravel 被预测为 Trees；
3. 1 个 Trees 被预测为 Asphalt，1 个被预测为 Meadows；
4. 2 个 Painted metal sheets 被预测为 Shadows；
5. 5 个 Bitumen 被预测为 Shadows。

从类别属性看，Trees 与 Meadows 都包含植被光谱特征，Bitumen 与 Shadows、Asphalt 与铺装材料之间也可能在局部谱空上下文中混淆。25×25 patch 会让相邻地物影响中心像元判断。由于这仍是单次种子结果，不足以判断类别差异是否稳定，后续需要多 seed 均值和标准差。

## 5. 分类图

下图左侧为全部有标签像元真值，中间只回填固定测试集预测，右侧为同一模型对全部有标签像元的推理结果。右图仅用于空间展示，正式指标仍只由中间的 test split 计算。

![Pavia University 分类图](../../experiments/pavia_university__hybridsn__seed1442__20260817-1251/classification_map.png)

## 6. 性能结果

| 性能项 | 结果 |
|---|---:|
| 训练总时间 | 494.36 s（8.24 min） |
| 平均每 epoch | 4.931 s |
| 平均训练吞吐 | 2,604.1 samples/s |
| 测试端到端推理时间 | 2.359 s |
| 测试端到端吞吐 | 12,692.4 samples/s |
| 测试端到端单样本时间 | 0.0788 ms/sample |
| 纯模型 batch256 推理 | 7.040 ms/batch |
| 纯模型计算吞吐 | 36,362.4 samples/s |
| 训练峰值显存（allocated） | 363.02 MiB |
| 训练峰值显存（reserved） | 648.00 MiB |
| 模型参数内存（float32） | 18.48 MiB |
| final checkpoint | 55.49 MiB |

端到端吞吐包含 patch 动态提取、DataLoader、CPU→GPU 传输和模型计算；纯模型计算基准复用已送入 GPU 的 batch。因此前者更接近实际批量推理，后者用于比较模型结构本身。

## 7. 高准确率的空间重叠审计

本实验在算法上保证标准化与 PCA 只在训练集拟合，test 也没有参与 epoch 选择；但随机像元划分与大 patch 会产生另一类实验偏差：训练和测试中心像元虽然不同，它们的 25×25 空间上下文高度重叠。

审计结果如下：

| 空间审计项 | 结果 |
|---|---:|
| 测试 patch 内至少含 1 个训练中心像元 | 29,944 / 29,944（100%） |
| 测试 patch 内至少含 1 个同类训练中心像元 | 29,944 / 29,944（100%） |
| 每个测试 patch 内训练中心像元数中位数 | 92 |
| 每个测试 patch 内训练中心像元数均值 | 100.56 |
| 每个测试 patch 内同类训练中心像元数中位数 | 89 |

因此，99.9532% OA 是有效的 `paper30` 课程/论文兼容口径结果，但不能直接解释为模型对新区域的泛化能力。后续应保留本 baseline，同时新增空间块划分或不重叠区域划分对照；模型改进优先在 `fair24_6_70` 的 validation 上选择方案，避免在 test 上追逐极小差异。

## 8. 结果文件完整性

正式运行目录至少包含：

- `config.yaml`、`config_source.yaml`：生效配置与源配置；
- `environment.json/txt`：软件、GPU、命令、Git 状态和数据 SHA-256；
- `train.log`、`history.json/csv`：逐轮过程；
- `checkpoint_final.pt`、`checkpoint_last.pt`：最终与恢复 checkpoint；
- `metrics.json`、`per_class_accuracy.csv`：定量结果；
- `performance.json`：时间、吞吐和显存；
- `predictions_test.npz`、`classification_maps.npz`：可追溯预测；
- `spatial_overlap_audit.json`：空间重叠证据；
- `loss_curve.png`、`confusion_matrix.png`、`per_class_accuracy.png`、`classification_map.png`：报告图；
- `run_manifest.json`：文件大小与 SHA-256 清单。

# 参考文献

[1] S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, “HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification,” *IEEE Geoscience and Remote Sensing Letters*, vol. 17, no. 2, pp. 277–281, 2020, doi: 10.1109/LGRS.2019.2918719。项目本地材料：`../../../HybridSNExploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification.pdf`。

[2] 教师课程资料：《大作业5：高光谱大作业》与《大作业安排20260814》；实验报告模板《实验报告-大作业-组号(1)》注重原理、实验过程、结果展示、误差分析以及代码/图表附件的一致性。

[3] Scikit-learn developers, *PCA documentation*；本项目使用 `PCA(svd_solver="full", whiten=False)` 的训练集拟合语义。

[4] PyTorch contributors, *Conv3d, Conv2d, CrossEntropyLoss and Adam documentation*。

# 心得体会

本节应由各组员结合本人实际工作独立完成，避免统一套话。建议至少回答：

1. 从“能前向传播”到“可复现正式实验”，最容易忽略的配置、数据身份或记录问题是什么；
2. 为什么 99.99% OA 仍需检查空间重叠，而不能只凭高分判断模型泛化好；
3. 在本次实现中对 3D→2D 特征重排、logits/交叉熵接口和训练/测试隔离有哪些新的理解；
4. 后续本人负责的模型改进、消融或多数据集工作准备如何开展。

# 附件与 Word 排版清单

最终 Word 报告应继续使用教师原始模板，不改变封面、主标题顺序和评分说明。建议插入以下证据：

1. 项目环境检查与最新全仓 `pytest` 通过截图；
2. 模型 Notebook 中模型结构、逐层 shape 和参数量输出截图；
3. 正式训练日志的首 3 轮、中间波动轮和第 100 轮截图；
4. 本文四张正式结果图，并在图下注明 run 名称和 seed；
5. 指标与性能表，不只贴图片；
6. 空间重叠审计表及限制说明；
7. 代码附件、配置、run manifest、checkpoint、预测 NPZ 和参考文献；
8. 组内分工与权重（总和 100%），由组长按实际工作量填写。
