# 单人任务看板

状态含义：`待开始`、`进行中`、`待复核`、`已完成`、`受阻`。

## 阶段 0：环境与工作区

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 建立固定工作区 | 已完成 | `实验交付/` |
| 固定实验协议 | 已完成 | `EXPERIMENT_PROTOCOL.md` |
| 建立独立 Python 环境 | 已完成 | `.venv`，Python 3.12.13 |
| 安装 GPU PyTorch | 已完成 | PyTorch 2.12.1 + CUDA 13.0 |
| GPU 前向与反向测试 | 已完成 | RTX 5070 测试通过 |

## 阶段 1：数据获取与审计

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 下载 Pavia University | 已完成 | 影像与标签 `.mat` |
| 下载 Indian Pines corrected | 已完成 | 影像与标签 `.mat` |
| 接入 Salinas corrected | 已完成 | 补充影像与标签 `.mat` |
| 确认并获取 Houston 版本 | 待开始 | Houston 2013 数据与标签 |
| 校验文件名、变量名与维度 | 已完成 | `dataset_inventory.json` |
| 输出类别样本统计 | 已完成 | CSV 与标签图 |

## 阶段 2：数据划分与预处理

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 固定 train/val/test 划分 | 已完成 | Pavia `paper30`、`fair24_6_70` 的 `.npz`、元数据、统计表与验收脚本 |
| 实现标准化 | 已完成 | 仅训练集拟合、可保存/加载的逐波段变换器 |
| 实现 PCA | 已完成 | Pavia 两协议的 15 维状态与解释方差元数据 |
| 实现 LDA | 已完成 | sklearn SVD-LDA、训练集专用拟合、LDA8 冻结状态与一行配置切换 |
| 实现邻域切片 | 已完成 | 动态 pixel/patch 输出及边界单元测试 |
| 封装 Dataset/DataLoader | 已完成 | 传统特征矩阵与 PyTorch 统一数据接口 |
| 整理统一运行入口 | 已完成 | 七步中英双语 Python 主程序、统计可视化 Notebook 与入口说明 |
| 小模型试运行 | 已完成 | 临时网络完整 1 epoch GPU 联通验收与 JSON 日志（非正式实验） |
| 阶段封板与工程整理 | 已完成 | 中文核心入口、YAML 配置目录、统一使用说明及历史工具归档 |
| 同步核心报告与论文 | 已完成 | 唯一实施报告、论文阶段结果和教师作业要求对应表 |

### 阶段 2 当前验收记录（2026-08-13）

- 固定种子：`1442`。
- `paper30`：12,832 个训练样本、0 个验证样本、29,944 个测试样本。
- `fair24_6_70`：10,265 个训练样本、2,567 个验证样本、29,944 个测试样本。
- 两套协议共享同一测试集；`fair24_6_70` 的训练集与验证集合并后等于 `paper30` 的训练集。
- 固定产物：`data/splits/pavia_university__{paper30|fair24_6_70}__seed1442.{npz,json}` 及对应 `__stats.csv`。
- 划分自动验收：`tests/test_dataset_splits.py` 与 `scripts/验证固定划分.py`；相关检查已纳入当前全套测试。
- 预处理实现：`src/datasets/高光谱预处理.py`，支持 `none/PCA/LDA × pixel/patch`、传统特征矩阵、PyTorch Dataset/DataLoader 及状态保存/加载。
- 唯一 Notebook 入口：`notebooks/高光谱数据预处理主流程.ipynb`，统一包含固定划分约束、两协议关系、冻结预处理、Dataset/DataLoader、批次契约和描述性可视化；逐框解读为 `docs/数据预处理Notebook代码框解读.md`。
- 可复用绘图接口：`src/visualization/预处理分析.py`，包含 split 数量/空间图、训练光谱统计、PCA/LDA 解释比与散点、训练 patch 画廊和 batch 类别构成。
- 统一命令行入口：`scripts/运行数据预处理.py`，按七步只读执行并逐步显示中英双语目的、输入、输出和验收；运行说明为根目录 `使用说明.md`。
- 数据预处理唯一 YAML 入口：`configs/数据预处理/Pavia数据预处理.yaml`；三个选择字段与冻结状态规则见同目录 `配置调参说明.md`。七份旧路线预设已移入 `归档/阶段2历史配置/`。
- 整理结果：日常 `scripts/` 只保留环境、配置、只读运行、验证和受控状态生成入口；阶段 0–2 一次性工具仅在原开发机器本地归档，不进入公共仓库。
- 已冻结状态：`paper30` 与 `fair24_6_70` 的 `standard+pca15+patch25`，以及 LDA8 的 fair pixel/patch25 和 paper30 patch25 参数与元数据，位于 `data/processed/pavia_university/`。
- 数据管线冒烟验收：临时网络在 RTX 5070 上完整遍历 10,265 个训练样本和 2,567 个验证样本；训练参数更新、验证参数稳定性与 `test_set_evaluated=false` 均已核验。
- 验收日志：短批次和完整 1 epoch JSON 诊断日志保存在原开发机器本地，不作为论文结果或公共仓库内容。
- 阶段 2 封板时核心自动测试为 48 项通过：11 项统一/模型配置与历史路线检查、2 项数据读取、8 项固定划分、17 项预处理、4 项可视化接口、1 项复现性和 5 项七步主入口测试。其后阶段 3.1 新增的模型测试见下节；波段选择、LBP、Gabor 仍属后续路线。

## 阶段 3：HybridSN 基线

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 核对原论文与教师参考仓库 | 已完成 | Pavia 逐层结构、shape、参数量口径与差异说明 |
| 复现 HybridSN 结构 | 已完成 | `src/models/HybridSN模型.py` 与 13 项模型测试 |
| 固化数据—模型两文件流程 | 已完成 | 数据 Notebook 保存模型就绪 NPZ；模型 Notebook 只读加载 |
| 搭建训练、验证与评价流程 | 已完成 | 可恢复 CLI、训练/推理、指标、性能、图表和运行清单 |
| Pavia 调试实验 | 已完成 | 1 epoch、test 封存，5.29 s/epoch，数据与 CUDA 链路正常 |
| Pavia 正式实验 | 进行中 | 统一 seed=1442 已完成；还需多 seed 稳定性与均值±标准差 |
| 输出完整结果 | 已完成 | OA/AA/Kappa、逐类精度、曲线、矩阵、分类图、性能和空间审计 |

### 阶段 3.1 当前验收记录（2026-08-14）

- 输入资料：HybridSN 原论文、教师《大作业5：高光谱大作业》、本项目冻结实验协议与数据接口，以及教师指定参考仓库 `High_spectrum_BIT`。参考仓库仅用于核对 PyTorch 维度顺序、3D→2D reshape 和 logits 输出写法；其 PCA20、patch11、50/50 划分、注意力模块及其他偏离论文的设置未带入基线。
- 固定数据契约：`paper30`、split seed `1442`、训练集专用 `standard + PCA15`、`patch25`，输入 `N×1×15×25×25`，输出 `N×9` logits，模型内部无 Softmax。
- 原论文结构：`Conv3D(1→8,k=7×3×3)` → `Conv3D(8→16,k=5×3×3)` → `Conv3D(16→32,k=3×3×3)` → reshape `32×3→96` → `Conv2D(96→64,k=3×3)` → Flatten → `18496→256→128→9`，隐藏层使用 ReLU，两处 Dropout 为 0.4，无池化、BatchNorm 和数据增强。
- Pavia 逐层输出：`N×8×9×23×23`、`N×16×5×21×21`、`N×32×3×19×19`、`N×96×19×19`、`N×64×17×17`、`N×18496`、`N×256`、`N×128`、`N×9`。
- Pavia 参数量为 `4,844,793`。论文表中 `5,122,176` 对应 30 波段、16 类的 Indian Pines 设置，不能直接作为 Pavia 模型参数量验收值。
- `notebooks/高光谱数据预处理主流程.ipynb` 已生成并复用同指纹 `model_ready_dataset.npz`，保存变换后立方体、坐标、标签与固定 split 索引，不物化约 1.5 GiB 的重复 patch，也不覆盖固定划分或 PCA 冻结状态。
- `notebooks/HybridSN模型结构学习.ipynb` 已完成模型输入、逐层检查、损失/优化器、训练控制和最终指标接口；`RUN_TRAINING=False`、`RUN_FINAL_TEST=False` 保证 Notebook 本身不重复训练或重新评价测试集，后续章节只读回放阶段 3.2 的冻结正式结果。
- `tests/test_hybridsn_model.py` 的 13 项检查全部通过，覆盖精确层配置、参数量、逐层 shape、logits、单 batch 交叉熵反传、输入契约及错误参数。阶段 3.1 当时为 `61 passed`；阶段 3.2 的当前总数见下一节。

### 阶段 3.2 首个正式 baseline 记录（2026-08-17）

- 运行入口：`scripts/运行HybridSN基线.py`；固定配置为 `paper30 + split seed 1442 + standard + PCA15 + patch25`，训练 seed=1442，Adam `1e-3`，batch 256，100 epochs，无增强、BatchNorm 或测试集选模。
- 运行结果：训练 12,832、测试 29,944；最终固定第 100 epoch checkpoint 的 OA=99.9532%、AA=99.8790%、Kappa=0.999380，测试交叉熵 0.004360；总计误分 14 个像元。
- 性能：RTX 5070 Laptop GPU 上训练总计 494.36 s，平均 4.931 s/epoch；端到端测试推理 2.359 s、12,692 samples/s；纯模型 batch256 推理约 7.040 ms/batch；峰值训练显存分配 363.02 MiB、保留 648 MiB。
- 可复现产物：配置源与生效副本、环境、日志、history CSV/JSON、final/last checkpoint、metrics、performance、测试预测、run manifest、自动实验记录、loss 曲线、混淆矩阵、逐类精度和分类图均已写入本地 `experiments/pavia_university__hybridsn__seed1442__20260817-1251/`。
- 解释限制：25×25 随机像元 split 空间审计显示，100% 测试 patch 内含至少一个训练中心像元且 100% 含同类训练中心像元；该高分仅作为课程/论文兼容 baseline，不作为跨区域泛化结论。后续新增空间块划分对照，并在 `fair24_6_70` 上完成多 seed 调参与模型比较。
- 工程验收：新增通用分类指标、空间回填、空间重叠审计、训练/推理/checkpoint 与结果图接口；seed 迁移后阶段 3.2 为 `77 passed`。

### 阶段 3.3 光谱预处理公平对比（2026-08-17）

- 运行入口：`scripts/运行HybridSN预处理对比.py`；使用 `fair24_6_70 + split seed1442`，训练 10,265、验证 2,567、测试 29,944，训练 seed1442，五种方法统一 30 epoch 并按 validation OA 选 checkpoint。
- 受控变量：所有路线统一输出 15 个通道，输入均为 `N×1×15×25×25`，使用相同 4,844,793 参数 HybridSN、Adam `1e-3`、batch256；只改变训练集拟合的光谱表示。
- 路线与结果：标准化 PCA15（Validation OA 100%，Test OA 99.9532%，14 错）、无标准化 PCA15（94.5851%，94.9773%，1,504 错）、PCA15 whitening（99.9610%，99.9132%，26 错）、均匀 15 原始波段（99.9221%，99.9299%，21 错）、Fisher 15 原始波段（99.7273%，99.7095%，87 错）。
- 验证结论：标准化 PCA15 的 validation OA 唯一最高，按预先规则选为当前预处理方案；测试结果只作冻结模型描述，不用于反向选方法。
- 方法结论：逐波段标准化应保留；均匀全谱 15 波段可达到较高精度但本次未超过 PCA；集中在第 7–21 波段的 Fisher 选择较弱；LDA8 因原版 7/5/3 光谱卷积最少需要 13 通道而不纳入纯预处理主表。
- 正式运行：`experiments/hybridsn_preprocessing_comparison__fair24_6_70__seed1442__20260817-1300/`。
- 产物：`notebooks/HybridSN预处理方法对比实验.ipynb`、`docs/solution_report/HybridSN预处理对比实验报告草稿.md`，以及正式运行目录中的汇总表、曲线、逐类精度、混淆矩阵和分类图。

### 阶段 3.4 研究架构分组对比（2026-08-17）

- 分组：A 降维/选带（PCA15、LDA8、均匀15波段），B 空间特征（无空间、LBP、Gabor、融合），C 分类器（RBF-SVM、HistGradientBoosting），D 传统/HybridSN 范式对照。
- 固定协议：`fair24_6_70`，split/model seed 统一 1442；预处理只在训练中心像元拟合，传统超参数预先固定，测试集不参与选择。
- 真实结果：纯 PCA15+SVM Test OA 93.7016%；LBP 99.5124%；Gabor 99.5692%；LBP+Gabor 融合 SVM 99.9365%；PCA15 HybridSN 99.9532%。
- 分类器结论：固定融合特征时 SVM 比 HistGB 高 0.5610 个百分点；HistGB 测试推理更快。当前未安装 XGBoost，报告明确区分 HistGB 与 XGBoost。
- 产物：`notebooks/研究架构分组对比实验.ipynb`、`docs/solution_report/研究架构分组对比实验报告草稿.md`、正式运行目录与 Word 素材目录。
- 验收：新增自包含 LBP/Gabor 实现及 4 项测试；当前全仓 `81 passed`，结果 Notebook 9 个代码单元全部执行且 0 error output。

## 阶段 4：改进与消融

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 调研候选模块 | 待开始 | 选择理由和预期作用 |
| 实现第一项改进 | 待开始 | 代码与单项消融 |
| 必要时实现第二项改进 | 待开始 | 代码与组合消融 |
| 比较精度与复杂度 | 待开始 | 均值、标准差、参数量、时间 |

## 阶段 5：对比模型与三数据集实验

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 复现 3D-CNN / 3D-1D-CNN | 待开始 | 模型代码和测试 |
| Pavia 公平比较 | 待开始 | 首轮对比表 |
| 扩展 Indian Pines | 待开始 | 多模型结果 |
| 扩展 Houston | 待开始 | 多模型结果 |
| 汇总多种子实验 | 待开始 | 主结果表与补充表 |

## 阶段 6：分析与最终交付

| 任务 | 状态 | 验收产出 |
|---|---|---|
| 统一科研绘图 | 待开始 | 标准化图表 |
| 完成结果讨论 | 待开始 | 回答四个研究问题 |
| 撰写研究论文 | 进行中 | 数据方法与阶段 3.1 模型结构已写，正式分类结果待补 |
| 撰写解题报告 | 进行中 | 阶段 3.1 结构与流程已同步，模型结果待补 |
| 制作答辩 PPT | 待开始 | 答辩版本 |
| 整理工程包 | 已完成 | 美化 README、依赖、运行命令、2.3 MiB 精简提交边界与发布前验收 |
| 准备 GitHub 协作规则 | 已完成 | `.gitignore`、`.gitattributes`、`CONTRIBUTING.md` 和仓库体积审计 |
