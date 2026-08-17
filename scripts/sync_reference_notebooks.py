"""Synchronize the two course notebooks with the reference-repository workflow.

The script performs an idempotent, structural update.  Cells inserted by this
script carry explicit tags so that rerunning it replaces, rather than duplicates,
the reference-aligned sections.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NOTEBOOK = PROJECT_ROOT / "notebooks" / "\u9ad8\u5149\u8c31\u6570\u636e\u9884\u5904\u7406\u4e3b\u6d41\u7a0b.ipynb"
MODEL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "HybridSN\u6a21\u578b\u7ed3\u6784\u5b66\u4e60.ipynb"
MANAGED_TAGS = {
    "reference-aligned",
    "reference-overview",
    "reference-correlation",
    "reference-reducer-comparison",
    "report-export",
    "formal-results-replay",
    "formal-results-figures",
    "formal-results-audit",
}

SEED_SOURCE_REPLACEMENTS = (
    ("seed345", "seed1442"),
    ("seed=345", "seed=1442"),
    ("seed 345", "seed 1442"),
    ("== 345", "== 1442"),
    ("default_rng(345)", "default_rng(1442)"),
    ("seed42", "seed1442"),
    ("seed=42", "seed=1442"),
    ("seed 42", "seed 1442"),
    ("'training_seed': 42", "'training_seed': 1442"),
    ("get('loader_seed', 42)", "get('loader_seed', 1442)"),
    (
        "assert len(misclassified_table) == 3",
        "assert len(misclassified_table) == int(metrics['test_samples'] - np.trace(np.asarray(metrics['confusion_matrix'])))",
    ),
)


def source_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str, tag: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [tag]},
        "source": source_lines(text),
    }


def code(text: str, tag: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": [],
        "source": source_lines(text),
    }


def read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    used_ids: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        cell_id = cell.get("id")
        if not cell_id or cell_id in used_ids:
            digest = hashlib.sha1(
                f"{index}:{cell.get('cell_type')}:{cell_source(cell)}".encode("utf-8")
            ).hexdigest()[:12]
            cell_id = f"cell-{digest}"
            cell["id"] = cell_id
        used_ids.add(cell_id)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def cell_source(cell: dict[str, Any]) -> str:
    return "".join(cell.get("source", []))


def remove_managed_cells(notebook: dict[str, Any]) -> None:
    kept = []
    for cell in notebook["cells"]:
        tags = set(cell.get("metadata", {}).get("tags", []))
        if tags.isdisjoint(MANAGED_TAGS):
            kept.append(cell)
    notebook["cells"] = kept


def migrate_seed_references(notebook: dict[str, Any]) -> None:
    """Update seed-bearing source text while leaving historical outputs untouched."""
    for cell in notebook["cells"]:
        source = cell_source(cell)
        for old, new in SEED_SOURCE_REPLACEMENTS:
            source = source.replace(old, new)
        cell["source"] = source_lines(source)


def find_cell_index(notebook: dict[str, Any], needle: str) -> int:
    for index, cell in enumerate(notebook["cells"]):
        if needle in cell_source(cell):
            return index
    raise ValueError(f"Cannot find notebook cell containing: {needle!r}")


def insert_after(notebook: dict[str, Any], needle: str, cells: list[dict[str, Any]]) -> None:
    index = find_cell_index(notebook, needle)
    notebook["cells"][index + 1:index + 1] = cells


def update_data_notebook() -> None:
    notebook = read_notebook(DATA_NOTEBOOK)
    migrate_seed_references(notebook)
    remove_managed_cells(notebook)

    notebook["cells"][0]["source"] = source_lines(
        """# 高光谱数据预处理主流程：参考仓库对齐版 / Reference-aligned workflow

本 Notebook 对齐参考 `Highspectrum.ipynb` 的“数据读取 → 伪彩色与标签统计 →
波段关系 → PCA/LDA → patch → 模型输入 → 结果导出”展示链，同时继续执行本项目
更严格的冻结划分和**仅训练集拟合**约束。它会复用已有预处理状态，不会重新划分样本，
也不会用验证集或测试集拟合标准化、PCA、LDA。

最终产物包括 `model_ready_dataset.npz`，以及可直接放入实验报告的 PNG/CSV 图表。"""
    )

    notebook["cells"].insert(
        1,
        markdown(
            """## 与参考 `Highspectrum.ipynb` 的流程对应关系

| 参考流程 | 本 Notebook 对应内容 | 口径说明 |
|---|---|---|
| PaviaU MAT 读取与尺寸检查 | 步骤 2 | 路径来自 YAML，禁止硬编码 |
| 伪彩色影像、类别数量、Ground Truth 与叠加图 | 参考对齐展示 A | 使用固定色表并导出报告图 |
| 波段相关性分析 | 统计 2C | 只用训练中心像元，避免查看测试光谱统计 |
| PCA/LDA 特征降维与分量图 | 步骤 3–4、统计 4C | 加载仅由训练集拟合的冻结状态 |
| patch 构造与训练/测试划分 | 步骤 5–7 | `25×25` patch 按需提取，不复制海量邻域 |
| 后续分类与分类图 | `HybridSN模型结构学习.ipynb` | 本 Notebook 只准备模型输入，不假装输出分类结果 |

参考本中的 Gabor/LBP/GLCM 与传统分类器属于后续特征优化路线，本阶段不混入
HybridSN baseline；未来会在同一冻结 split 上作为独立实验进行对比。""",
            "reference-aligned",
        ),
    )

    insert_after(
        notebook,
        "Split: {split_counts}",
        [
            markdown(
                """### 参考对齐展示 A：伪彩色、类别数量、标签图与叠加图

对应参考 `Highspectrum.ipynb` 的数据可视化部分。伪彩色使用第 70、28、21
波段（Python 下标 69、27、20），每个通道采用 2%–98% 分位拉伸。
类别表保留背景类 0，便于核对 `610×340=207,400` 个像元的组成。""",
                "reference-overview",
            ),
            code(
                """# 参考流程所需的报告级数据总览；不拟合任何统计模型。
import pandas as pd
from IPython.display import display

REPORT_ASSET_DIR = PROJECT_ROOT / 'results/notebook_outputs/reference_aligned/data_preprocessing'
REPORT_ASSET_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LABEL_COLORS = (
    '#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
    '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#6A3D9A',
)
reference_label_cmap = ListedColormap(REFERENCE_LABEL_COLORS)

def percentile_stretch(channel: np.ndarray) -> np.ndarray:
    low, high = np.percentile(channel, (2, 98))
    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0, 1).astype(np.float32)

rgb_band_indices = (69, 27, 20)
false_color = np.stack(
    [percentile_stretch(data.cube[:, :, band]) for band in rgb_band_indices],
    axis=-1,
)

pixel_counts = np.bincount(data.label_map.reshape(-1), minlength=10)
class_rows = [{
    '类别编号': 0,
    '中文类别': '背景/未标注',
    'English class': 'Background / unlabeled',
    '像元数': int(pixel_counts[0]),
    '占全图比例': float(pixel_counts[0] / data.label_map.size),
}]
for class_id, english_name in enumerate(data.spec.class_names, start=1):
    class_rows.append({
        '类别编号': class_id,
        '中文类别': PAVIA_CLASS_NAMES_ZH[class_id - 1],
        'English class': english_name,
        '像元数': int(pixel_counts[class_id]),
        '占全图比例': float(pixel_counts[class_id] / data.label_map.size),
    })
class_distribution_table = pd.DataFrame(class_rows)
assert int(class_distribution_table['像元数'].sum()) == data.label_map.size
display(class_distribution_table.style.format({'占全图比例': '{:.2%}'}))

masked_labels = np.ma.masked_where(data.label_map == 0, data.label_map)
overview_figure, axes = plt.subplots(1, 3, figsize=(16, 7.2))
axes[0].imshow(false_color)
axes[0].set_title('PaviaU 伪彩色 / False color\\n波段 70-28-21')
axes[1].imshow(data.label_map, cmap=reference_label_cmap, vmin=0, vmax=9, interpolation='nearest')
axes[1].set_title('真实标签 / Ground truth')
axes[2].imshow(false_color)
axes[2].imshow(masked_labels, cmap=reference_label_cmap, vmin=0, vmax=9,
               interpolation='nearest', alpha=0.72)
axes[2].set_title('影像与标签叠加 / Overlay')
for axis in axes:
    axis.axis('off')
overview_figure.suptitle('Pavia University 数据总览（参考仓库流程对齐）', fontsize=15)
overview_figure.tight_layout()
overview_figure.savefig(REPORT_ASSET_DIR / '01_PaviaU数据总览.png', dpi=220, bbox_inches='tight')
class_distribution_table.to_csv(
    REPORT_ASSET_DIR / '01_PaviaU类别数量.csv', index=False, encoding='utf-8-sig'
)
plt.show()
print(f'报告素材目录 / Report assets: {REPORT_ASSET_DIR}')""",
                "reference-overview",
            ),
        ],
    )

    insert_after(
        notebook,
        "train_spectra = data.cube",
        [
            markdown(
                """### 统计 2C：训练集波段相关性 / Train-only band correlation

对应参考 Notebook 的波段相关性分析。相关矩阵只由 12,832 个训练中心像元计算；
它用于说明相邻波段冗余和降维动机，不参与测试结果解释。""",
                "reference-correlation",
            ),
            code(
                """# 只用训练中心像元计算 103 个原始波段之间的 Pearson 相关系数。
band_correlation = np.corrcoef(train_spectra, rowvar=False)
assert band_correlation.shape == (data.cube.shape[-1], data.cube.shape[-1])
assert np.allclose(np.diag(band_correlation), 1.0)

band_correlation_figure, axis = plt.subplots(figsize=(8.5, 7.2))
image = axis.imshow(band_correlation, vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
axis.set_title('训练集原始波段相关矩阵 / Train-only band correlation')
axis.set_xlabel('波段编号 / Band index')
axis.set_ylabel('波段编号 / Band index')
band_correlation_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
band_correlation_figure.tight_layout()
band_correlation_figure.savefig(
    REPORT_ASSET_DIR / '03_训练集波段相关矩阵.png', dpi=220, bbox_inches='tight'
)
plt.show()

upper_triangle = np.abs(band_correlation[np.triu_indices_from(band_correlation, k=1)])
print({
    'mean_absolute_interband_correlation': float(upper_triangle.mean()),
    'fraction_abs_correlation_ge_0.95': float(np.mean(upper_triangle >= 0.95)),
})""",
                "reference-correlation",
            ),
        ],
    )

    insert_after(
        notebook,
        "reducer_scatter_figure.tight_layout()",
        [
            markdown(
                """### 统计 4C：冻结 PCA15 与 LDA8 分量对照 / Frozen PCA–LDA comparison

参考 Notebook 同时展示 PCA 和 LDA。本项目不在此处重新拟合，而是读取相同
`paper30/seed1442` 下已经冻结的 PCA15、LDA8 状态。二者都只由训练样本拟合，
测试集仅在正式模型冻结后用于一次最终评估。HybridSN baseline 固定使用 PCA15。""",
                "reference-reducer-comparison",
            ),
            code(
                """# 加载同一 split 上的冻结 LDA8，仅用于降维对照可视化，不改变 baseline 输入。
lda_state_dir = (
    PROJECT_ROOT / 'data/processed/pavia_university'
    / 'paper30__seed1442__standard_lda8_patch25'
)
lda_pipeline = HSIPreprocessingPipeline.load_state(
    lda_state_dir / 'preprocessing_state.npz',
    lda_state_dir / 'metadata.json',
)
assert lda_pipeline.fit_metadata_['fit_scope']['validation_and_test_used_for_fit'] is False
assert int(lda_pipeline.fit_metadata_['fit_scope']['training_samples']) == data.train_indices.size
lda_cube = lda_pipeline.attach_transformed_cube(data.cube)
assert lda_cube.shape == (610, 340, 8)

reducer_comparison_table = pd.DataFrame([
    {
        '方法': 'PCA', '分量数': transformed_cube.shape[-1],
        '拟合样本': int(pipeline.fit_metadata_['fit_scope']['training_samples']),
        '验证/测试参与拟合': False,
        'baseline输入': True,
    },
    {
        '方法': 'LDA', '分量数': lda_cube.shape[-1],
        '拟合样本': int(lda_pipeline.fit_metadata_['fit_scope']['training_samples']),
        '验证/测试参与拟合': False,
        'baseline输入': False,
    },
])
display(reducer_comparison_table)

reducer_comparison_figure, axes = plt.subplots(2, 3, figsize=(13, 9))
for row, (method, cube_for_display) in enumerate((('PCA15', transformed_cube), ('LDA8', lda_cube))):
    for component_index in range(3):
        axes[row, component_index].imshow(cube_for_display[:, :, component_index], cmap='viridis')
        axes[row, component_index].set_title(f'{method} 分量 {component_index + 1}')
        axes[row, component_index].axis('off')
reducer_comparison_figure.suptitle('同一冻结划分上的 PCA/LDA 空间分量对照')
reducer_comparison_figure.tight_layout()
reducer_comparison_figure.savefig(
    REPORT_ASSET_DIR / '06_PCA15与LDA8分量对照.png', dpi=220, bbox_inches='tight'
)
reducer_comparison_table.to_csv(
    REPORT_ASSET_DIR / '06_PCA15与LDA8口径对照.csv', index=False, encoding='utf-8-sig'
)
plt.show()""",
                "reference-reducer-comparison",
            ),
        ],
    )

    final_index = find_cell_index(notebook, "## 如何阅读这些统计图")
    notebook["cells"][final_index:final_index] = [
        markdown(
            """## 步骤 9：集中导出实验报告素材 / Export report-ready artifacts

除 Notebook 内嵌输出外，本节把关键图表以 220 dpi PNG、统计表以 UTF-8-BOM CSV
写入固定目录。这样后续在 Word 中插图时无需截图，也不会引用临时运行目录。""",
            "report-export",
        ),
        code(
            """# 将前面已经验收过的 Figure 对象集中导出，文件名按报告叙事顺序编号。
figures_to_export = {
    '02_固定划分数量.png': split_count_figure,
    '02_固定划分空间分布.png': split_map_figure,
    '04_训练集逐类光谱.png': spectral_figure,
    '05_降维解释比例.png': reducer_ratio_figure,
    '05_训练集前两分量分布.png': reducer_scatter_figure,
    '07_九类代表性Patch.png': patch_gallery_figure,
    '08_首个训练批次类别分布.png': batch_distribution_figure,
}
for filename, figure in figures_to_export.items():
    figure.savefig(REPORT_ASSET_DIR / filename, dpi=220, bbox_inches='tight')

split_rows = []
for class_id, english_name in enumerate(data.spec.class_names, start=1):
    row = {
        '类别编号': class_id,
        '中文类别': PAVIA_CLASS_NAMES_ZH[class_id - 1],
        'English class': english_name,
    }
    for split_column, split_name in enumerate(SPLIT_NAMES):
        row[split_name] = int(counts[class_id - 1, split_column])
    row['total'] = int(counts[class_id - 1].sum())
    split_rows.append(row)
split_table = pd.DataFrame(split_rows)
split_table.to_csv(REPORT_ASSET_DIR / '02_逐类划分数量.csv', index=False, encoding='utf-8-sig')

preprocessing_summary = pd.DataFrame([
    {'环节': '原始数据', '结果': str(tuple(data.cube.shape)), '约束': 'PaviaU, 103 bands'},
    {'环节': '固定划分', '结果': 'train=12,832; validation=0; test=29,944', '约束': 'paper30, seed=1442'},
    {'环节': '标准化+PCA', '结果': str(tuple(transformed_cube.shape)), '约束': '仅训练集拟合; PCA15; no whiten'},
    {'环节': '模型输入', '结果': '(N, 1, 15, 25, 25)', '约束': '动态 patch; constant padding'},
    {'环节': '模型就绪产物', '结果': str(MODEL_READY_PATH.relative_to(PROJECT_ROOT)), '约束': config.fingerprint()},
])
preprocessing_summary.to_csv(
    REPORT_ASSET_DIR / '00_预处理流程摘要.csv', index=False, encoding='utf-8-sig'
)
display(preprocessing_summary)
print('已导出 / Exported:')
for path in sorted(REPORT_ASSET_DIR.iterdir()):
    print(f'  - {path.relative_to(PROJECT_ROOT)}')""",
            "report-export",
        ),
    ]

    write_notebook(DATA_NOTEBOOK, notebook)


def update_model_notebook() -> None:
    notebook = read_notebook(MODEL_NOTEBOOK)
    migrate_seed_references(notebook)
    remove_managed_cells(notebook)

    notebook["cells"][0]["source"] = source_lines(
        """# HybridSN baseline：结构、训练、推理与分类图 / Reference-aligned workflow

本 Notebook 对齐参考 `HybridSN.ipynb` 的“加载数据 → 构造 patch → 定义网络 →
训练 → 测试指标 → 全图推理 → 分类图”主流程。默认模式不重复训练或再次使用测试集，
而是严格回放已经完成的 100 轮正式 baseline 运行；若需要新实验，使用文末给出的
统一命令行入口，确保 checkpoint、日志、性能与图表一次性归档。

固定口径：Pavia University、paper30/seed1442、训练 seed1442、PCA15、25×25 patch、
Adam(1e-3)、batch256、100 epochs、无数据增强、无 BatchNorm。"""
    )
    notebook["cells"].insert(
        1,
        markdown(
            """## 与参考 `HybridSN.ipynb` 的流程对应关系

| 参考流程 | 本 Notebook 对应内容 | 本项目改进 |
|---|---|---|
| `loadData/applyPCA/createImageCubes` | 第 1–2 节 | 直接复用冻结 `model_ready_dataset.npz`，不在测试数据上拟合 PCA |
| HybridSN 网络定义与 summary | 第 3 节 | 模型唯一实现位于 `src/models`，逐层 shape 断言 |
| 训练循环、loss 曲线 | 第 4–6、8 节 | 正式运行保留 100 轮历史、checkpoint 和环境指纹 |
| OA、AA、Kappa、逐类精度、混淆矩阵 | 第 7–9 节 | 统一测试一次，指标可由预测文件复核 |
| 遍历像元生成分类图 | 第 9 节 | 批量推理，并同时输出测试像元图和全部有标签像元图 |
| 结果保存 | 第 10 节 | PNG/CSV/JSON/Markdown 集中归档，便于实验报告和后续模型对比 |

参考代码中的 PCA20、11×11 patch、50/50 随机划分与本课程冻结 baseline 参数不同；
这里保持**流程高度一致**，但不改动已经确定的实验协议。""",
            "reference-aligned",
        ),
    )

    result_index = find_cell_index(notebook, "## 流程验收")
    notebook["cells"][result_index:result_index] = [
        markdown(
            """## 8. 正式 100 轮实验结果回放 / Replay the completed formal run

本节默认加载已经生成并冻结的正式运行，不重新训练，也不重新推理测试集。
它会核对运行目录必须包含 checkpoint、预测、指标、性能和分类图，并验证
预处理指纹与当前 `model_ready_dataset.npz` 一致。""",
            "formal-results-replay",
        ),
        code(
            """# 自动优先选择指定正式运行；只有文件完整、已正式测试的运行才允许回放。
import pandas as pd
import shutil
from IPython.display import Image, display

PREFERRED_FORMAL_RUN = (
    PROJECT_ROOT / 'experiments/pavia_university__hybridsn__seed1442__latest'
)
required_formal_files = {
    'checkpoint_final.pt', 'history.csv', 'metrics.json', 'performance.json',
    'predictions_test.npz', 'classification_maps.npz', 'loss_curve.png',
    'confusion_matrix.png', 'per_class_accuracy.png', 'classification_map.png',
    'spatial_overlap_audit.json', 'environment.json', '实验记录.md',
}

def is_complete_formal_run(path: Path) -> bool:
    if not path.is_dir() or any(not (path / name).is_file() for name in required_formal_files):
        return False
    candidate_metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
    candidate_performance = json.loads((path / 'performance.json').read_text(encoding='utf-8'))
    return (
        candidate_metrics.get('test_set_used_for_model_selection') is False
        and candidate_performance.get('formal_test_evaluated') is True
        and candidate_performance.get('training', {}).get('epochs') == 100
    )

candidates = [PREFERRED_FORMAL_RUN] + sorted(
    (PROJECT_ROOT / 'experiments').glob('pavia_university__hybridsn__seed1442__*'),
    reverse=True,
)
FORMAL_RUN = next((path for path in candidates if is_complete_formal_run(path)), None)
if FORMAL_RUN is None:
    raise FileNotFoundError('No complete 100-epoch HybridSN formal run was found.')

metrics = json.loads((FORMAL_RUN / 'metrics.json').read_text(encoding='utf-8'))
performance = json.loads((FORMAL_RUN / 'performance.json').read_text(encoding='utf-8'))
spatial_audit = json.loads((FORMAL_RUN / 'spatial_overlap_audit.json').read_text(encoding='utf-8'))
environment = json.loads((FORMAL_RUN / 'environment.json').read_text(encoding='utf-8'))
history_df = pd.read_csv(FORMAL_RUN / 'history.csv', encoding='utf-8-sig')

assert len(history_df) == 100 and int(history_df['epoch'].iloc[-1]) == 100
assert metrics['test_samples'] == artifact['test_indices'].size == 29944
assert metrics['preprocessing_fingerprint'] == str(artifact['config_fingerprint'].item())
assert environment['selected_device'] == 'cuda'

headline_metrics = pd.DataFrame([
    {'指标': 'OA', '数值': metrics['oa'], '报告值': f"{metrics['oa']:.4%}"},
    {'指标': 'AA', '数值': metrics['aa'], '报告值': f"{metrics['aa']:.4%}"},
    {'指标': 'Kappa', '数值': metrics['kappa'], '报告值': f"{metrics['kappa']:.6f}"},
    {'指标': 'Test loss', '数值': metrics['test_loss'], '报告值': f"{metrics['test_loss']:.8f}"},
    {'指标': 'Test samples', '数值': metrics['test_samples'], '报告值': f"{metrics['test_samples']:,}"},
])
display(headline_metrics)
print(f'正式运行目录 / Formal run: {FORMAL_RUN.relative_to(PROJECT_ROOT)}')
print(f"Checkpoint: {metrics['checkpoint']} ({metrics['checkpoint_sha256'][:12]}...)")""",
            "formal-results-replay",
        ),
        markdown(
            """## 9. 训练曲线、混淆矩阵、逐类精度与分类图

以下图像来自同一正式运行目录。分类图依次给出 Ground Truth、仅测试像元预测、
全部有标签像元预测；背景保持黑色，因此不会把未标注区域伪装成分类结果。""",
            "formal-results-figures",
        ),
        code(
            """# 逐类结果表；类别索引在模型内部是 0–8，报告使用原始标签 1–9。
PAVIA_CLASS_NAMES_ZH = (
    '沥青路面', '草地', '砾石', '树木', '涂漆金属板',
    '裸土', '沥青材料', '自锁砖', '阴影',
)
per_class_table = pd.DataFrame(metrics['per_class'])
per_class_table.insert(2, 'class_name_zh', PAVIA_CLASS_NAMES_ZH)
per_class_table['accuracy_percent'] = per_class_table['accuracy'] * 100
display(per_class_table.style.format({
    'accuracy': '{:.8f}',
    'accuracy_percent': '{:.4f}%',
}))

figure_files = (
    ('训练损失曲线 / Loss curve', 'loss_curve.png'),
    ('测试混淆矩阵 / Confusion matrix', 'confusion_matrix.png'),
    ('逐类精度 / Per-class accuracy', 'per_class_accuracy.png'),
    ('分类结果图 / Classification maps', 'classification_map.png'),
)
for title, filename in figure_files:
    print(f'\\n{title}')
    display(Image(filename=str(FORMAL_RUN / filename), width=1150))""",
            "formal-results-figures",
        ),
        code(
            """# 从独立保存的测试预测重新计算错误像元，证明指标可审计而非只依赖 PNG。
with np.load(FORMAL_RUN / 'predictions_test.npz', allow_pickle=False) as prediction_file:
    labels_test = prediction_file['labels'].astype(np.int64)
    predictions_test = prediction_file['predictions'].astype(np.int64)
    coordinates_test = prediction_file['coordinates'].astype(np.int64)

wrong_positions = np.flatnonzero(labels_test != predictions_test)
misclassified_rows = []
for position in wrong_positions:
    true_index = int(labels_test[position])
    predicted_index = int(predictions_test[position])
    row, column = map(int, coordinates_test[position])
    misclassified_rows.append({
        'row': row,
        'column': column,
        'true_raw_label': true_index + 1,
        'true_class': PAVIA_CLASS_NAMES_ZH[true_index],
        'predicted_raw_label': predicted_index + 1,
        'predicted_class': PAVIA_CLASS_NAMES_ZH[predicted_index],
    })
misclassified_table = pd.DataFrame(misclassified_rows)
assert len(misclassified_table) == 3
assert np.isclose(np.mean(labels_test == predictions_test), metrics['oa'])
print(f'测试集错误像元 / Misclassified test pixels: {len(misclassified_table)} / {len(labels_test):,}')
display(misclassified_table)""",
            "formal-results-figures",
        ),
        markdown(
            """## 10. 性能记录、解释边界与后续对比基线

性能同时报告端到端测试推理（含 DataLoader/patch 提取）和纯模型吞吐，两者不可混写。
此外，`paper30` 是随机像元划分而非空间分区划分；25×25 邻域会在空间上重叠，
因此极高精度只代表当前协议下的 baseline，不等价于跨区域泛化能力。""",
            "formal-results-audit",
        ),
        code(
            """# 将速度、显存和空间重叠审计整理成可直接进入报告的表格。
training_perf = performance['training']
test_perf = performance['test_inference']
compute_perf = performance['model_compute_benchmark']

performance_table = pd.DataFrame([
    {'项目': '可训练参数', '数值': performance['model']['trainable_parameters'], '单位': 'parameters'},
    {'项目': '100轮训练总耗时', '数值': training_perf['total_seconds'], '单位': 's'},
    {'项目': '平均每轮耗时', '数值': training_perf['mean_epoch_seconds'], '单位': 's/epoch'},
    {'项目': '训练峰值显存 allocated', '数值': training_perf['peak_memory_allocated_bytes'] / 1024**2, '单位': 'MiB'},
    {'项目': '端到端测试耗时', '数值': test_perf['elapsed_seconds'], '单位': 's'},
    {'项目': '端到端测试吞吐', '数值': test_perf['throughput_samples_per_second'], '单位': 'samples/s'},
    {'项目': '纯模型 batch256 延迟', '数值': compute_perf['milliseconds_per_batch'], '单位': 'ms/batch'},
    {'项目': '纯模型吞吐', '数值': compute_perf['throughput_samples_per_second'], '单位': 'samples/s'},
])
display(performance_table.style.format({'数值': '{:,.4f}'}))

overlap_table = pd.DataFrame([
    {
        '审计项': '测试 patch 内至少一个训练中心',
        '比例': spatial_audit['any_training_center_in_query_patch']['fraction_with_at_least_one'],
        '中位数/patch': spatial_audit['any_training_center_in_query_patch']['median'],
        '均值/patch': spatial_audit['any_training_center_in_query_patch']['mean'],
    },
    {
        '审计项': '测试 patch 内至少一个同类训练中心',
        '比例': spatial_audit['same_class_training_center_in_query_patch']['fraction_with_at_least_one'],
        '中位数/patch': spatial_audit['same_class_training_center_in_query_patch']['median'],
        '均值/patch': spatial_audit['same_class_training_center_in_query_patch']['mean'],
    },
])
display(overlap_table.style.format({'比例': '{:.4%}', '中位数/patch': '{:.2f}', '均值/patch': '{:.2f}'}))

comparison_baseline = pd.DataFrame([{
    'method': 'HybridSN baseline',
    'reducer': 'PCA15',
    'patch_size': 25,
    'split_protocol': 'paper30',
    'split_seed': 1442,
    'training_seed': 1442,
    'epochs': training_perf['epochs'],
    'parameters': performance['model']['trainable_parameters'],
    'oa': metrics['oa'],
    'aa': metrics['aa'],
    'kappa': metrics['kappa'],
    'test_inference_seconds': test_perf['elapsed_seconds'],
    'test_throughput_samples_per_second': test_perf['throughput_samples_per_second'],
}])
display(comparison_baseline)""",
            "formal-results-audit",
        ),
        markdown(
            """> **结果解释边界**：标准化与 PCA 没有使用测试标签或测试统计，测试集也没有参与
> epoch/checkpoint 选择；但随机像元划分使空间邻域高度相关。报告中应把该结果表述为
> “固定 paper30 协议下的内部测试性能”，不可据此声称未知区域或跨场景仍有 99.99% 精度。""",
            "formal-results-audit",
        ),
        markdown(
            """## 11. 导出报告素材与新实验入口

本节把正式运行中的图表、指标、性能和错误像元复制到固定报告目录，并生成一行
baseline 对比表。以后优化模型时按相同列追加实验结果即可。默认不会启动新训练。""",
            "report-export",
        ),
        code(
            """# 复制而不改写正式运行归档；报告目录只作为便于 Word 插图的稳定入口。
MODEL_REPORT_DIR = PROJECT_ROOT / 'results/notebook_outputs/reference_aligned/hybridsn_baseline'
MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

for filename in (
    'loss_curve.png', 'confusion_matrix.png', 'per_class_accuracy.png',
    'classification_map.png', 'metrics.json', 'performance.json',
    'spatial_overlap_audit.json', '实验记录.md',
):
    shutil.copy2(FORMAL_RUN / filename, MODEL_REPORT_DIR / filename)
headline_metrics.to_csv(MODEL_REPORT_DIR / 'headline_metrics.csv', index=False, encoding='utf-8-sig')
per_class_table.to_csv(MODEL_REPORT_DIR / 'per_class_accuracy.csv', index=False, encoding='utf-8-sig')
misclassified_table.to_csv(MODEL_REPORT_DIR / 'misclassified_test_pixels.csv', index=False, encoding='utf-8-sig')
performance_table.to_csv(MODEL_REPORT_DIR / 'performance_summary.csv', index=False, encoding='utf-8-sig')
overlap_table.to_csv(MODEL_REPORT_DIR / 'spatial_overlap_summary.csv', index=False, encoding='utf-8-sig')
comparison_baseline.to_csv(MODEL_REPORT_DIR / 'baseline_for_future_comparison.csv', index=False, encoding='utf-8-sig')

print(f'报告素材目录 / Report assets: {MODEL_REPORT_DIR}')
for path in sorted(MODEL_REPORT_DIR.iterdir()):
    print(f'  - {path.name}')

print('\\n如需从头创建新的正式运行，请在项目根目录执行：')
print(r'.venv\\Scripts\\python.exe scripts\\运行HybridSN基线.py --output-dir experiments\\<new-run-name>')""",
            "report-export",
        ),
    ]

    # The old gate remains useful for study, but clarify what the default path does.
    try:
        gate_index = find_cell_index(notebook, "测试集保持封存 / Test set remains sealed.")
    except ValueError:
        gate_index = find_cell_index(notebook, "第 8 节只读回放已冻结正式结果")
    gate_source = cell_source(notebook["cells"][gate_index]).replace(
        "print('测试集保持封存 / Test set remains sealed.')",
        "print('本单元不重新评估测试集；第 8 节只读回放已冻结正式结果。')",
    )
    notebook["cells"][gate_index]["source"] = source_lines(gate_source)

    write_notebook(MODEL_NOTEBOOK, notebook)


def main() -> None:
    update_data_notebook()
    update_model_notebook()
    print(f"Updated: {DATA_NOTEBOOK}")
    print(f"Updated: {MODEL_NOTEBOOK}")


if __name__ == "__main__":
    main()
