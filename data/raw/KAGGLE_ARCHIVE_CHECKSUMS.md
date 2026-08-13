# Data Integrity Checksums

Use these MD5 hashes to verify file integrity after download.

## MD5 Checksums

```
Indian_pines_corrected.mat  66dbc9f4a9b7c9b1445f60a87b505101
Indian_pines_gt.mat         9414943dac1d80faaa9165c8b460510c
PaviaU.mat                  165a3c7488995f54a19add47c7eed4cd
PaviaU_gt.mat               b8c3ba44b077c26e24220463aa855bd3
Salinas_corrected.mat       485d8802f4a6b4ebc0767d48dd0da06b
Salinas_gt.mat              7b8da653a61bb0271b27b37fb926390f
```

## Verification

### Using Python:
```python
import hashlib

def verify_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Verify a file
print(verify_file('Indian_pines_corrected.mat'))
# Expected: 66dbc9f4a9b7c9b1445f60a87b505101
```

### Using Command Line (macOS/Linux):
```bash
md5sum *.mat
```

### Using Command Line (macOS specific):
```bash
md5 *.mat
```

## File Sizes

```
Indian_pines_corrected.mat   5.7 MB
Indian_pines_gt.mat          1.1 KB
PaviaU.mat                  33.0 MB
PaviaU_gt.mat               11.0 KB
Salinas_corrected.mat       25.0 MB
Salinas_gt.mat               4.2 KB
```

Total: ~64 MB

## Data Shapes

```
Indian Pines:
  - HSI Cube:      (145, 145, 200)  dtype: uint16
  - Ground Truth:  (145, 145)       dtype: uint8

Pavia University:
  - HSI Cube:      (610, 340, 103)  dtype: uint16
  - Ground Truth:  (610, 340)       dtype: uint8

Salinas:
  - HSI Cube:      (512, 217, 204)  dtype: int16
  - Ground Truth:  (512, 217)       dtype: uint8
```
