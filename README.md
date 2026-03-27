# Deep-Learning-based-Brain-Tumor-Segmentation
# Claude Generated README.md
3D U-Net model for brain tumor segmentation using the BraTS dataset (Task01_BrainTumour). Performs 5-fold cross-validation and reports Dice and HD95 metrics for Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).

---

## Requirements

- Python 3.10+
- PyTorch
- MONAI
- scikit-learn
- medpy
- matplotlib
- pandas

Install all dependencies:

```bash
pip install torch monai scikit-learn medpy matplotlib pandas
```

---

## Dataset

This project uses the **Medical Segmentation Decathlon** Task01 Brain Tumour dataset.

Expected directory structure:

```
Task01_BrainTumour/
├── imagesTr/        # 4-channel NIfTI images (.nii.gz)
├── labelsTr/        # Segmentation labels (.nii.gz)
├── imagesTs/        # Test images (unused in training)
└── dataset.json
```

---

## Running Locally

### Basic run (CPU)

```bash
python 3D-Unet.py --data_dir ./Task01_BrainTumour
```

### Full run with all outputs saved

```bash
python 3D-Unet.py \
  --data_dir ./Task01_BrainTumour \
  --num_epochs 25 \
  --num_folds 5 \
  --batch_size 1 \
  --roi_size 128 128 128 \
  --num_workers 4 \
  --save_model \
  --save_plots \
  --save_visualizations \
  --save_results \
  --output_csv brats_results.csv
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `/content/data/Task01_BrainTumour` | Path to dataset |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--num_epochs` | `25` | Epochs per fold |
| `--num_folds` | `5` | Number of CV folds |
| `--batch_size` | `1` | Training batch size |
| `--roi_size` | `128 128 128` | Random crop size (D H W) |
| `--num_workers` | `4` | DataLoader workers |
| `--save_model` | flag | Save best model per fold |
| `--save_plots` | flag | Save training curve plots |
| `--save_visualizations` | flag | Save prediction images |
| `--save_results` | flag | Save metrics to CSV |
| `--output_csv` | `brats_results.csv` | Output CSV filename |

### Outputs

| Path | Contents |
|---|---|
| `best_model_fold{N}.pth` | Best model weights for fold N |
| `plots/fold{N}_curves.png` | Loss and Dice curves |
| `visualizations/fold{N}_prediction.png` | MRI / ground truth / prediction |
| `brats_results.csv` | Per-fold and average metrics |

---

## Running on Google Colab

### Step 1 — Open a new notebook and install dependencies

```python
!pip install monai medpy
```

> PyTorch, numpy, matplotlib, scikit-learn, and pandas are pre-installed on Colab.

### Step 2 — Upload the dataset

Option A — upload the `.tar` file directly:

```python
from google.colab import files
uploaded = files.upload()   # select Task01_BrainTumour.tar
```

Then extract it:

```python
!tar -xf Task01_BrainTumour.tar -C /content/data/
```

Option B — mount Google Drive (recommended for large files):

```python
from google.drive import drive
drive.mount('/content/drive')
!tar -xf /content/drive/MyDrive/Task01_BrainTumour.tar -C /content/data/
```

### Step 3 — Upload the script

```python
from google.colab import files
files.upload()   # select 3D-Unet.py
```

### Step 4 — Run training

```python
!python 3D-Unet.py \
  --data_dir /content/data/Task01_BrainTumour \
  --num_epochs 25 \
  --num_folds 5 \
  --batch_size 1 \
  --num_workers 2 \
  --save_model \
  --save_plots \
  --save_visualizations \
  --save_results
```

> **GPU:** Go to **Runtime > Change runtime type > H100 GPU** before running. Training will automatically use CUDA if available.

### Step 5 — Download results

```python
from google.colab import files
files.download('brats_results.csv')
```

To download all plots:

```python
import shutil
shutil.make_archive('plots', 'zip', 'plots')
files.download('plots.zip')
```
