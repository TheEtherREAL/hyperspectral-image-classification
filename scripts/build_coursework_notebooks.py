"""Build the three ordered coursework notebooks from reproducible source cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "coursework/notebooks"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def common_setup() -> str:
    return r'''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import yaml
from IPython.display import Image, display

cwd = Path.cwd().resolve()
PROJECT_ROOT = next(path for path in (cwd, *cwd.parents) if (path / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SEED = 1442
print("PROJECT_ROOT =", PROJECT_ROOT)
print("frozen random seed =", SEED)
'''


def notebook_1():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown(r'''
# 01 数据读取、固定划分与多路线预处理

对应参考仓库 `Highspectrum.ipynb` / `HybridSN.ipynb` 的数据处理顺序，但修正了其中的硬编码路径、外部特征文件依赖和数据泄漏风险。支持 Pavia University、Indian Pines、Salinas 三套 `.mat` 数据。

本 Notebook 的唯一随机协议：**按类别分层 24% 训练、6% 验证、70% 测试，seed=1442**。划分文件一旦生成，后续模型不再重新抽样。
'''),
        code(common_setup()),
        markdown("## 0. 环境、数据集与路径配置验证"),
        code(r'''
from src.coursework.stage1 import load_stage1_config, route_config, run_stage1
from src.datasets.高光谱预处理 import load_hsi_data
from src.datasets.数据集注册 import DATASETS

DATASET_KEY = "pavia_university"  # 可改为 indian_pines / salinas
RUN_STAGE1 = False                 # True 会重新拟合并覆盖确定性 stage1 输出
CONFIG_PATH = PROJECT_ROOT / f"coursework/configs/stage1_data/{DATASET_KEY}.yaml"
OUTPUT_DIR = PROJECT_ROOT / f"coursework/outputs/stage1/{DATASET_KEY}"
values = load_stage1_config(CONFIG_PATH)
checks = []
for key, spec in DATASETS.items():
    checks.append({
        "dataset": key,
        "cube_file": spec.data_file,
        "cube_exists": (PROJECT_ROOT / "data/raw" / spec.data_file).is_file(),
        "label_file": spec.label_file,
        "label_exists": (PROJECT_ROOT / "data/raw" / spec.label_file).is_file(),
        "classes": len(spec.class_names),
    })
pd.DataFrame(checks)
'''),
        markdown("## 1. 数据集加载与简单取样"),
        code(r'''
selected_config = route_config(values, values["selected_route"])
data = load_hsi_data(PROJECT_ROOT, selected_config)
print("cube:", data.cube.shape, data.cube.dtype)
print("label map:", data.label_map.shape, data.label_map.dtype)
print("labeled pixels:", data.labels.size)
first = data.train_indices[0]
row, col = data.coordinates[first]
print("sample identity:", {"sample_index": int(first), "coordinate": (int(row), int(col)), "raw_label": int(data.labels[first])})
display(Image(filename=str(OUTPUT_DIR / "figures/01_dataset_overview.png")))
'''),
        markdown("## 2. 数据集构成分析"),
        code(r'''
records = []
for class_id, class_name in enumerate(data.spec.class_names, start=1):
    row = {"class_id": class_id, "class_name": class_name, "total": int(np.sum(data.labels == class_id))}
    for split_name, indices in data.indices_by_split.items():
        row[split_name] = int(np.sum(data.labels[indices] == class_id))
    records.append(row)
composition = pd.DataFrame(records)
display(composition)
display(Image(filename=str(OUTPUT_DIR / "figures/02_class_and_split_distribution.png")))
'''),
        markdown(r'''
## 3. 固定数据划分（后续禁止改动）

参考仓库常用 30%/70% 训练测试口径。为了让超参数选择不接触测试集，把 30% 建模池再按 80%/20% 分为训练/验证，得到 24%/6%/70%。这是分层的**像元级随机划分**，三组互斥并覆盖全部有标签像元。
'''),
        code(r'''
split_rows = []
assigned = []
for split_name, indices in data.indices_by_split.items():
    assigned.append(indices)
    split_rows.append({"split": split_name, "samples": len(indices), "actual_fraction": len(indices) / len(data.labels)})
assigned = np.concatenate(assigned)
assert np.unique(assigned).size == data.labels.size
assert np.array_equal(np.sort(assigned), np.arange(data.labels.size))
assert int(data.split_metadata["protocol"]["seed"]) == 1442
pd.DataFrame(split_rows)
'''),
        markdown("## 4. 原始波段、PCA、LDA 与波段选择"),
        code(r'''
route_table = []
for key, route in values["routes"].items():
    route_table.append({"route": key, **route})
pd.DataFrame(route_table).fillna("")
'''),
        markdown(r'''
- `raw_all_bands`：不降维，保留全谱；仍用训练集统计量逐波段标准化。
- `pca15`：无监督线性降维；仅用训练中心像元拟合。
- `lda8/lda15`：有监督投影，上限为类别数减一；仅用训练标签拟合。
- `uniform15`：按波段序号均匀保留，标签无关。
- `fisher15`：只用训练标签计算 Fisher 分数，选出判别性最高的原始波段。
'''),
        code(r'''
if RUN_STAGE1:
    stage1_result = run_stage1(PROJECT_ROOT, CONFIG_PATH, OUTPUT_DIR, overwrite=True)
manifest = json.loads((OUTPUT_DIR / "stage1_manifest.json").read_text(encoding="utf-8"))
route_summary = pd.read_csv(OUTPUT_DIR / "preprocessing_route_summary.csv")
display(route_summary)
for name in ("route_pca15.png", "route_lda8.png", "route_uniform15.png", "route_fisher15.png"):
    path = OUTPUT_DIR / "figures" / name
    if path.is_file():
        display(Image(filename=str(path), width=720))
'''),
        markdown("## 5. 阶段一交接文件"),
        code(r'''
handoff = manifest["selected_artifact"]
print("selected route:", manifest["selected_route"])
print("model-ready NPZ:", PROJECT_ROOT / handoff["model_ready"])
print("preprocessing state:", PROJECT_ROOT / handoff["state"])
print("effective YAML:", OUTPUT_DIR / "selected_preprocessing.yaml")
assert manifest["frozen_protocol"]["immutable_for_downstream_experiments"] is True
assert manifest["frozen_protocol"]["seed"] == 1442
handoff
'''),
        markdown(r'''
## 任务性质的准确解释

高光谱数据不是“一堆没有空间关系的点”，而是一个 `H×W×B` 数据立方体：每个二维位置是一个像元，每个像元有长度为 `B` 的光谱向量。标签图只给部分像元赋类别，0 表示未标注背景。

本课题的学习单位是**有标签中心像元**：传统方法读取中心像元光谱并可拼接 LBP/Gabor 邻域特征；HybridSN 读取以该像元为中心的 `25×25×B` patch，输出中心像元的一个地物类别。因此它是“高光谱像元分类 / patch-based dense classification”，可以生成与语义分割相似的分类图，但**不是标准端到端语义分割**：模型并非一次输入整幅图并输出每个位置的掩膜，训练/测试也按有标签像元抽样。分类图中未标注区域没有可用于定量评价的真值。
'''),
    ]
    return nb


def notebook_2():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown(r'''
# 02 HybridSN 基线：训练、验证、测试与可视化

模型接收阶段一的冻结 PCA15 + `25×25` patch。结构与 HybridSN 参考流程一致：三层 3D 卷积提取联合光谱—空间特征，reshape 后用 2D 卷积融合，再进入全连接分类器。结构、优化器、epoch 和分类目标均由 YAML 控制。
'''),
        code(common_setup()),
        markdown("## 0. 配置和阶段一产物验证"),
        code(r'''
from src.coursework.stage2 import load_stage2_config, run_stage2
from src.datasets.高光谱预处理 import PreprocessingConfig
from src.training.hybridsn_baseline import load_model_ready_artifact

RUN_TRAINING = False  # True 会按 YAML 重新训练两组；默认读取已执行结果
STAGE1_MANIFEST = PROJECT_ROOT / "coursework/outputs/stage1/pavia_university/stage1_manifest.json"
stage1 = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
preprocess_config = PreprocessingConfig(**stage1["selected_artifact"]["config"])
artifact = load_model_ready_artifact(PROJECT_ROOT / stage1["selected_artifact"]["model_ready"], preprocess_config)
print("dataset:", artifact.dataset_name)
print("cube:", artifact.transformed_cube.shape)
print("split sizes:", len(artifact.train_indices), len(artifact.validation_indices), len(artifact.test_indices))
assert artifact.split_seed == 1442 and artifact.split_protocol == "fair24_6_70"
'''),
        markdown("## 1. 模型输入取样"),
        code(r'''
from src.training.hybridsn_baseline import build_datasets
datasets = build_datasets(artifact)
sample = datasets["train"][0]
{key: (tuple(value.shape), str(value.dtype)) for key, value in sample.items()}
'''),
        markdown("## 2. YAML 可调 HybridSN 模型定义"),
        code(r'''
import torch
from src.models.可配置HybridSN import ConfigurableHybridSN, HybridSNArchitecture

SOFTMAX_CONFIG = PROJECT_ROOT / "coursework/configs/stage2_hybridsn/pavia_softmax_baseline.yaml"
SIGMOID_CONFIG = PROJECT_ROOT / "coursework/configs/stage2_hybridsn/pavia_sigmoid_ablation.yaml"
softmax_values = load_stage2_config(SOFTMAX_CONFIG)
architecture = HybridSNArchitecture.from_mapping(softmax_values)
model = ConfigurableHybridSN(input_bands=artifact.output_bands, patch_size=artifact.patch_size, num_classes=artifact.num_classes, architecture=architecture)
dummy_output = model(torch.zeros(2, 1, artifact.output_bands, artifact.patch_size, artifact.patch_size))
print(model)
print("output:", tuple(dummy_output.shape))
print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
'''),
        markdown(r'''
## 3. Softmax 与 Sigmoid 分类目标

两组网络都输出 logits。Softmax 主实验使用 `CrossEntropyLoss`，它显式建模九类互斥分布；Sigmoid 消融使用 one-hot 标签的 `BCEWithLogitsLoss`，把九类视作一对多二分类，预测仍取最大 logit。对单标签互斥地物分类，Softmax 是更自然的默认选择；Sigmoid 在本次单次划分上取得更高分，只能作为实测消融结果，不能据此声称其概率校准或普遍泛化更好。
'''),
        code(r'''
from src.models.可配置HybridSN import build_classification_objective
print(build_classification_objective("softmax", artifact.num_classes))
print(build_classification_objective("sigmoid", artifact.num_classes))
'''),
        markdown("## 4. 训练与验证流程（验证集选择最佳 epoch，测试集不调参）"),
        code(r'''
SOFTMAX_OUT = PROJECT_ROOT / "coursework/outputs/stage2/pavia_softmax_baseline"
SIGMOID_OUT = PROJECT_ROOT / "coursework/outputs/stage2/pavia_sigmoid_ablation"
if RUN_TRAINING:
    run_stage2(PROJECT_ROOT, SOFTMAX_CONFIG, SOFTMAX_OUT, overwrite=True)
    run_stage2(PROJECT_ROOT, SIGMOID_CONFIG, SIGMOID_OUT, overwrite=True)

summaries = []
for output in (SOFTMAX_OUT, SIGMOID_OUT):
    summaries.append(json.loads((output / "summary.json").read_text(encoding="utf-8")))
pd.DataFrame(summaries)[["objective", "best_epoch", "validation_oa", "test_oa", "test_aa", "test_kappa", "test_errors", "training_seconds", "test_inference_seconds"]]
'''),
        markdown("## 5. Epoch 曲线与性能对比"),
        code(r'''
for name, output in (("Softmax", SOFTMAX_OUT), ("Sigmoid", SIGMOID_OUT)):
    print(name)
    display(Image(filename=str(output / "learning_curves.png"), width=900))
display(Image(filename=str(PROJECT_ROOT / "coursework/outputs/report_figures/hybridsn_objective_comparison.png"), width=900))
'''),
        markdown("## 6. 测试指标、混淆矩阵、逐类准确率和分类图"),
        code(r'''
for title, output in (("Softmax baseline", SOFTMAX_OUT), ("Sigmoid ablation", SIGMOID_OUT)):
    print(title)
    display(Image(filename=str(output / "confusion_matrix.png"), width=650))
    display(Image(filename=str(output / "per_class_accuracy.png"), width=760))
    display(Image(filename=str(output / "classification_maps.png"), width=1100))
'''),
        markdown("## 7. 重要有效性边界：随机像元划分的空间邻域重叠"),
        code(r'''
audit = json.loads((SOFTMAX_OUT / "spatial_overlap_audit.json").read_text(encoding="utf-8"))
pd.DataFrame([
    {"measure": "任意训练中心落入测试 patch", **audit["any_training_center_in_query_patch"]},
    {"measure": "同类训练中心落入测试 patch", **audit["same_class_training_center_in_query_patch"]},
])
'''),
        markdown(r'''
随机像元划分会让相邻训练/测试 patch 高度重叠，因此极高 OA 不能直接等价为对新场景的泛化能力。报告必须同时保留这一审计。建设性改进顺序：增加空间块划分或跨场景验证；做多 seed 重复并报告均值±标准差；再比较轻量化、BatchNorm、数据增强和注意力模块。
'''),
    ]
    return nb


def notebook_3():
    nb = nbf.v4.new_notebook()
    methods = [
        "PCA15_SVM",
        "PCA15_XGBoost",
        "PCA15_LBP_SVM",
        "PCA15_Gabor_SVM",
        "PCA15_LBP_Gabor_SVM",
        "PCA15_LBP_Gabor_XGBoost",
    ]
    nb.cells = [
        markdown(r'''
# 03 传统特征、分类器与完整分组对比

对应研究架构中的传统分支：PCA/原始光谱 → LBP/Gabor 空间特征 → SVM/XGBoost。每个方法使用独立代码框，并在同一个 seed=1442 固定划分上输出 OA、AA、Kappa、混淆矩阵、逐类准确率与分类图。
'''),
        code(common_setup()),
        markdown("## 0. 配置与公共特征缓存"),
        code(r'''
from src.coursework.stage3 import prepare_traditional_context, run_traditional_method, save_method_comparison

RUN_METHODS = False  # True 时逐组重新训练；默认加载本次已完成结果
CONFIG_PATH = PROJECT_ROOT / "coursework/configs/stage3_traditional/pavia_traditional.yaml"
OUTPUT_DIR = PROJECT_ROOT / "coursework/outputs/stage3/pavia_traditional"
values = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
assert values["experiment"]["seed"] == 1442
context = prepare_traditional_context(PROJECT_ROOT, CONFIG_PATH) if RUN_METHODS else None
pd.DataFrame([{"method": key, **value} for key, value in values["methods"].items()])
'''),
        markdown("## 1. 每格一种传统方法"),
    ]
    for index, method in enumerate(methods, start=1):
        nb.cells.append(markdown(f"### 1.{index} `{method}`"))
        nb.cells.append(code(f'''
METHOD_KEY = "{method}"
if RUN_METHODS:
    summary = run_traditional_method(context, METHOD_KEY, OUTPUT_DIR)
else:
    summary = json.loads((OUTPUT_DIR / METHOD_KEY / "summary.json").read_text(encoding="utf-8"))
pd.Series(summary)
'''))
    nb.cells.extend([
        markdown("## 2. 传统方法总表与分类图"),
        code(r'''
traditional = pd.read_csv(OUTPUT_DIR / "traditional_method_comparison.csv")
display(traditional[["display_name", "feature_dimension", "validation_oa", "test_oa", "test_aa", "test_kappa", "training_seconds", "test_inference_seconds", "test_throughput_samples_per_second"]])
display(Image(filename=str(OUTPUT_DIR / "traditional_method_comparison.png"), width=1050))
for method in ("PCA15_SVM", "PCA15_LBP_Gabor_SVM", "PCA15_LBP_Gabor_XGBoost"):
    print(method)
    display(Image(filename=str(OUTPUT_DIR / method / "classification_maps.png"), width=1050))
'''),
        markdown("## 3. 预处理严格控制变量：同一 RBF-SVM"),
        code(r'''
PREPROCESSING_DIR = PROJECT_ROOT / "coursework/outputs/comparisons/preprocessing_svm"
preprocessing = pd.read_csv(PREPROCESSING_DIR / "preprocessing_svm_comparison.csv")
display(preprocessing[["route", "output_bands", "validation_oa", "test_oa", "test_aa", "test_kappa", "training_seconds", "all_labeled_inference_seconds"]])
display(Image(filename=str(PREPROCESSING_DIR / "preprocessing_svm_comparison.png"), width=950))
'''),
        markdown("## 4. 与 HybridSN 的统一比较"),
        code(r'''
REPORT_FIGURES = PROJECT_ROOT / "coursework/outputs/report_figures"
combined = pd.read_csv(REPORT_FIGURES / "all_method_comparison.csv")
display(combined)
display(Image(filename=str(REPORT_FIGURES / "all_method_accuracy_comparison.png"), width=1100))
display(Image(filename=str(REPORT_FIGURES / "accuracy_efficiency_tradeoff.png"), width=900))
display(Image(filename=str(REPORT_FIGURES / "research_architecture.png"), width=1100))
'''),
        markdown(r'''
## 5. 结果分析与建设性改进

1. 单独光谱分类不足：PCA15+SVM OA 为 93.70%，增加 LBP 或 Gabor 后分别达到 99.51% 和 99.57%，说明局部空间纹理是主要增益来源。
2. 传统融合很强：PCA15+LBP+Gabor+SVM 达 99.9365%，与 HybridSN 同量级，但 RBF-SVM 测试推理约 55.74 秒，显著慢于 HybridSN 约 7 秒，也远慢于 XGBoost 的亚秒级推理。
3. XGBoost 是效率方案：融合 XGBoost OA 99.2052%，精度低于融合 SVM，却提供约 7 万样本/秒吞吐。
4. 降维不是越“监督”越好：在仅使用中心光谱的统一 SVM 下，原始全波段 94.06%、PCA15 93.70%、LDA8 91.34%、均匀15波段 90.36%、Fisher15 70.59%。Fisher 单波段排序忽略波段间冗余与互补，是明显失败路线；后续应改为 mRMR、互信息、SPA 或可学习注意力选带。
5. 下一轮优化应优先验证泛化而非继续挤压随机划分 OA：空间块划分、多 seed 重复、跨数据集迁移；随后再做轻量化 HybridSN、BatchNorm/残差、光谱—空间注意力和数据增强。
'''),
    ])
    return nb


def write_notebook(path: Path, notebook) -> None:
    notebook.metadata = {
        "kernelspec": {
            "display_name": "HSI Coursework (.venv)",
            "language": "python",
            "name": "hsi-coursework",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbf.write(notebook, path)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "01_数据读取划分与预处理.ipynb", notebook_1())
    write_notebook(NOTEBOOK_DIR / "02_HybridSN基线训练验证测试.ipynb", notebook_2())
    write_notebook(NOTEBOOK_DIR / "03_传统特征与分类器对比.ipynb", notebook_3())


if __name__ == "__main__":
    main()
