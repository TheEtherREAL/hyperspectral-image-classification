# 🛰️ Hyperspectral Image Classification Study Datasets

This dataset contains three standard hyperspectral image (HSI) study datasets widely used in remote sensing and classification research.

## 📊 Datasets Included

### 1. Indian Pines 🌾
- **Scene**: Agricultural area, Indiana, USA (June 1992)
- **Sensor**: AVIRIS (Airborne Visible/Infrared Imaging Spectrometer)
- **Spatial Resolution**: 145×145 pixels (20m per pixel)
- **Spectral Bands**: 200 bands (0.4-2.5 μm wavelength range)
- **Land Cover Classes**: 16 agricultural types
- **Files**:
  - `Indian_pines_corrected.mat` - HSI cube (5.7 MB)
  - `Indian_pines_gt.mat` - Ground truth labels (1.1 KB)

**Classes**: Alfalfa, Corn-notill, Corn-mintill, Corn, Grass-pasture, Grass-trees, Grass-pasture-mowed, Hay-windrowed, Oats, Soybean-notill, Soybean-mintill, Soybean-clean, Wheat, Woods, Buildings-Grass-Trees-Drives, Stone-Steel-Towers

### 2. Pavia University 🏛️
- **Scene**: Urban area around University of Pavia, Italy (2003)
- **Sensor**: ROSIS (Reflective Optics System Imaging Spectrometer)
- **Spatial Resolution**: 610×340 pixels (1.3m per pixel)
- **Spectral Bands**: 103 bands (0.43-0.86 μm wavelength range)
- **Land Cover Classes**: 9 urban types
- **Files**:
  - `PaviaU.mat` - HSI cube (33 MB)
  - `PaviaU_gt.mat` - Ground truth labels (11 KB)

**Classes**: Asphalt, Meadows, Gravel, Trees, Painted metal sheets, Bare Soil, Bitumen, Self-Blocking Bricks, Shadows

### 3. Salinas 🥬
- **Scene**: Agricultural area, Salinas Valley, California, USA (1998)
- **Sensor**: AVIRIS
- **Spatial Resolution**: 512×217 pixels (3.7m per pixel)
- **Spectral Bands**: 204 bands (0.4-2.5 μm wavelength range)
- **Land Cover Classes**: 16 vegetable types
- **Files**:
  - `Salinas_corrected.mat` - HSI cube (25 MB)
  - `Salinas_gt.mat` - Ground truth labels (4.2 KB)

**Classes**: Broccoli_green_weeds_1, Broccoli_green_weeds_2, Fallow, Fallow_rough_plow, Fallow_smooth, Stubble, Celery, Grapes_untrained, Soil_vinyard_develop, Corn_senesced_green_weeds, Lettuce_romaine_4wk, Lettuce_romaine_5wk, Lettuce_romaine_6wk, Lettuce_romaine_7wk, Vinyard_untrained, Vinyard_vertical_trellis

## 💻 Usage Example

```python
import scipy.io as sio
import numpy as np

# Load Indian Pines dataset
data = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

print(f"HSI Cube shape: {data.shape}")      # (145, 145, 200)
print(f"Labels shape: {labels.shape}")       # (145, 145)
print(f"Number of bands: {data.shape[2]}")   # 200
print(f"Number of classes: {len(np.unique(labels))}")  # 17 (including background)

# Access a single pixel's spectral signature
pixel_spectrum = data[50, 50, :]  # All 200 bands for pixel (50, 50)

# Access a specific band
band_100 = data[:, :, 99]  # 2D image at band 100

# Load Pavia University dataset
pavia_data = sio.loadmat('PaviaU.mat')['paviaU']
pavia_labels = sio.loadmat('PaviaU_gt.mat')['paviaU_gt']

# Load Salinas dataset
salinas_data = sio.loadmat('Salinas_corrected.mat')['salinas_corrected']
salinas_labels = sio.loadmat('Salinas_gt.mat')['salinas_gt']
```

## 🔐 Data Integrity Verification

MD5 checksums for file verification:

```
Indian_pines_corrected.mat: 66dbc9f4a9b7c9b1445f60a87b505101
Indian_pines_gt.mat:        9414943dac1d80faaa9165c8b460510c
PaviaU.mat:                 165a3c7488995f54a19add47c7eed4cd
PaviaU_gt.mat:              b8c3ba44b077c26e24220463aa855bd3
Salinas_corrected.mat:      485d8802f4a6b4ebc0767d48dd0da06b
Salinas_gt.mat:             7b8da653a61bb0271b27b37fb926390f
```

Verify checksums in Python:
```python
import hashlib

def verify_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Example
print(verify_file('Indian_pines_corrected.mat'))
```

## 📚 Data Format

- **Format**: MATLAB `.mat` files (version 5.0)
- **HSI Cubes**: 3D NumPy arrays (height × width × bands)
  - Data type: `uint16` or `int16`
  - Values: Raw reflectance values (not normalized)
- **Ground Truth**: 2D NumPy arrays (height × width)
  - Data type: `uint8`
  - Values: Class labels (0 = unlabeled background, 1-N = class IDs)

## 🎓 Citation & Original Sources

These datasets are widely used benchmarks in the hyperspectral imaging research community.

**Indian Pines**:
- Original Source: Purdue University Research Repository
- Sensor: NASA AVIRIS
- Public Domain

**Pavia University**:
- Original Source: Computational Intelligence Group, University of the Basque Country
- Sensor: ROSIS (DLR)
- Public Domain

**Salinas**:
- Original Source: NASA Jet Propulsion Laboratory
- Sensor: AVIRIS
- Public Domain

If you use these datasets in your research, please cite:
```
@misc{hsi_benchmarks_2024,
  title={Hyperspectral Image Classification Benchmark Datasets},
  author={Standard HSI Research Community},
  year={2024},
  publisher={Kaggle},
  note={Curated collection of Indian Pines, Pavia University, and Salinas datasets}
}
```

## 🔬 Research Applications

These datasets are commonly used for:
- Land cover classification
- Crop type identification
- Urban material mapping
- Dimensionality reduction (PCA, ICA, etc.)
- Deep learning for remote sensing
- Spectral unmixing
- Feature extraction
- Benchmark comparison of classification algorithms

## 📊 Dataset Statistics

| Dataset | Size | Spatial | Spectral | Classes | Labeled Pixels | Scene Type |
|---------|------|---------|----------|---------|----------------|------------|
| **Indian Pines** | 5.7 MB | 145×145 | 200 | 16 | 10,249 | Agricultural |
| **Pavia University** | 33 MB | 610×340 | 103 | 9 | 42,776 | Urban |
| **Salinas** | 25 MB | 512×217 | 204 | 16 | 54,129 | Agricultural |

## 🔗 Related Resources

- **Complete HSI Classification Project**: [GitHub Repository](https://github.com/douglas-martins/hsi-study)

## 📄 License

These datasets are in the **public domain** and have been widely used as benchmarks in the hyperspectral imaging research community for over two decades. They are freely available for academic and commercial use.

## 🙏 Acknowledgments

- NASA Jet Propulsion Laboratory (AVIRIS sensor data)
- Purdue University (Indian Pines dataset curation)
- Gruppo di Telecomunicazioni, University of Pavia (Pavia University dataset)
- Computational Intelligence Group, University of the Basque Country (dataset distribution)
- **Sydney Matheus de Souza**, for providing his complete preprocessing pipeline and pre-trained CNN-3D models for all three datasets

---

**Total Dataset Size**: ~123 MB (6 data files + 3 pre-trained CNN-3D models)
**Last Updated**: February 2025
**Maintainer**: Douglas Martins (douglasfabiamartins)
