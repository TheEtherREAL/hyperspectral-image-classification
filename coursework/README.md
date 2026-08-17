# 高光谱图像分类大作业（seed 1442）

本目录是最终可交付实验入口。旧 `notebooks/`、`experiments/` 与历史结果保留为研究记录；新流程只通过下列三个编号 Notebook 组织：

1. `notebooks/01_数据读取划分与预处理.ipynb`：三数据集适配、固定划分、原始/PCA/LDA/波段选择、阶段一交接清单。
2. `notebooks/02_HybridSN基线训练验证测试.ipynb`：配置化 HybridSN、Softmax 主实验、Sigmoid 消融、训练曲线、混淆矩阵与分类图。
3. `notebooks/03_传统特征与分类器对比.ipynb`：光谱、LBP、Gabor、SVM、XGBoost 分组对比。

统一约束：按类别分层的 24% 训练 / 6% 验证 / 70% 测试，随机种子固定为 1442；所有预处理统计量只在训练集中心像元上拟合。测试集只做最终报告，不用于调参或选择模型。

## 目录

- `configs/stage1_data/`：三套数据集配置，改 `dataset.name` 即可切换。
- `configs/stage2_hybridsn/`：模型结构、分类目标、优化器和训练超参数。
- `configs/stage3_traditional/`：传统方法参数。
- `outputs/`：可复用数据、CSV、模型、指标和 PNG 报告图。
- `report/`：按教师 Word 模板生成的报告及可编辑 Markdown 草稿。

Pavia University 是默认完整演示数据集；Indian Pines 和 Salinas 已生成同一 seed、同一比例的冻结划分，并可使用对应 YAML 运行相同流程。

