# Setup and Execution Guide

**📁 Access Datasets, Weights & Results**: [Google Drive Folder](https://drive.google.com/drive/folders/1F7e-BxpRG240z-PROqJWcSeC7YIfi0pI?usp=sharing)

---

## 📋 Prerequisites

- Google Colab account (recommended) OR local machine with GPU
- Google Drive with sufficient storage (~5GB for datasets + models)
- Python 3.8 or higher
- CUDA-compatible GPU (for training)

---

## 🗂️ Step 1: Prepare Your Google Drive

### 1.1 Create Folder Structure

Create the following folders in your Google Drive:

```
MyDrive/
└── x24144801-Thesis/
    ├── Datasets/
    │   ├── RMHD_combined/
    │   │   ├── RMHD_combinedsorted_train.csv
    │   │   ├── RMHD_combinedsorted_val.csv
    │   │   └── RMHD_combinedsorted_test.csv
    │   ├── RMHD_Kaggle/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── test.csv
    │   └── 6476179.zip (SWMH dataset)
    ├── Weights/ (will store trained models)
    └── Results Sheets/ (will store SHAP analysis outputs)
```

### 1.2 Upload Your Datasets

Upload your dataset CSV files to the corresponding folders:
- RMHD Combined: 3 files (train, val, test)
- RMHD Kaggle: 3 files (train, val, test)
- SWMH: 1 zip file containing swmh.tar.gz

**Dataset Format Requirements:**
- Each CSV must have columns: `text` and `label`
- `text`: The Reddit post or text content
- `label`: The mental health category label

---

## 🚀 Step 2: Clone the Repository

First, clone the repository to get all notebook files:

```bash
git clone https://github.com/Mansoor07Tariq/TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS.git
cd TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS
```

**For Google Colab**: Upload the notebooks directly to Colab, or clone in a Colab cell:
```python
!git clone https://github.com/Mansoor07Tariq/TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS.git
%cd TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS
```

---

## 🚀 Step 3: Environment Setup

### Option A: Using Google Colab (Recommended)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Enable GPU**:
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` under Hardware accelerator
   - Choose `High-RAM` if available
3. **Upload Notebook**:
   - Click `File` → `Upload notebook`
   - Select one of the training notebooks from the cloned repository

### Option B: Using Local Machine

1. **Install Python 3.8+**: Download from [python.org](https://python.org)

2. **Create Virtual Environment**:
```bash
python -m venv mental_health_env
source mental_health_env/bin/activate  # On Windows: mental_health_env\Scripts\activate
```

3. **Install Jupyter**:
```bash
pip install jupyter notebook
```

4. **Install PyTorch with CUDA** (check [pytorch.org](https://pytorch.org) for your CUDA version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Step 4: Install Dependencies

Run this command in a notebook cell or terminal:

```bash
!pip install afinn transformers datasets evaluate accelerate shap openpyxl safetensors scikit-learn matplotlib seaborn pandas numpy
```

**Verify Installation**:
```python
import torch
import transformers
import shap
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

## 🏋️ Step 5: Training Models (Choose Your Dataset)

### 4.1 Train on RMHD Combined (8 classes)

1. Open: `RMHD_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`
2. **Mount Google Drive** (Cell 1):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   - Click the link and authorize access
3. **Update Dataset Paths** (Cell 3):
   ```python
   train_path = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_combined/RMHD_combinedsorted_train.csv"
   val_path   = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_combined/RMHD_combinedsorted_val.csv"
   test_path  = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_combined/RMHD_combinedsorted_test.csv"
   ```
4. **Run All Cells**: Click `Runtime` → `Run all`
5. **Training Time**: ~4-6 hours on Colab T4 GPU
6. **Model Saved To**: `/content/drive/MyDrive/...` (automatically saved)

### 4.2 Train on RMHD Kaggle (5 classes)

1. Open: `RMHDkaggle_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`
2. **Mount Google Drive** (Cell 1)
3. **Update Dataset Paths** (Cell 3):
   ```python
   train_path = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_Kaggle/train.csv"
   val_path   = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_Kaggle/val.csv"
   test_path  = "/content/drive/MyDrive/x24144801-Thesis/Datasets/RMHD_Kaggle/test.csv"
   ```
4. **Run All Cells**: Click `Runtime` → `Run all`
5. **Training Time**: ~3-5 hours on Colab T4 GPU

### 4.3 Train on SWMH (5 Reddit subreddits)

1. Open: `swmh_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`
2. **Mount Google Drive** (Cell 1)
3. **Extract Dataset** (Cell 2):
   ```python
   path_zip = "/content/drive/MyDrive/x24144801-Thesis/Datasets/6476179.zip"
   !unzip -o "$path_zip" -d "/content/"
   !tar -xvzf /content/swmh.tar.gz -C /content/
   ```
4. **Update Dataset Paths** (Cell 6):
   ```python
   train_path = "/content/swmh/train.csv"
   val_path   = "/content/swmh/val.csv"
   test_path  = "/content/swmh/test.csv"
   ```
5. **Run All Cells**: Click `Runtime` → `Run all`
6. **Training Time**: ~3-5 hours on Colab T4 GPU

### Training Tips:

- **Monitor Training**: Watch the loss and accuracy outputs
- **Early Stopping**: Training stops automatically if validation loss doesn't improve
- **Out of Memory**: Reduce batch size from 8 to 4 in Cell 15:
  ```python
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  ```
- **Save Checkpoints**: Best model is automatically saved to Google Drive

---

## 📊 Step 6: Model Evaluation

### 5.1 In-Domain Evaluation

1. Open: `Evaluation code.ipynb`
2. **Mount Google Drive** (Cell 1)
3. **Extract SWMH Dataset** (Cell 2) if needed
4. **Install Dependencies** (Cell 3)
5. **Update Model Paths** (Cell 6):
   ```python
   RMHD_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/RMHDCombined8_roberta_large_constructive_learning_affinnemotion_scl_v1.zip'
   SWMH_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/pytorch_modelroberta_emotion_wloss_constructive.bin'
   RMHD_kaggle_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/rmhdkaggle_pytorch_modelroberta_emotion_wloss_constructive.bin'
   ```
6. **Run Evaluation Cells** (Cells 7-9):
   - Cell 7: SWMH model on SWMH test set
   - Cell 8: RMHD model on RMHD test set
   - Cell 9: RMHD Kaggle model on RMHD Kaggle test set

**Expected Output**:
- ✅ Accuracy and Macro F1 scores
- ✅ Classification report with per-class metrics
- ✅ Excel files: `SWMH_predictions.xlsx`, `RMHD_predictions.xlsx`, `RMHDKaggle_predictions.xlsx`

### 5.2 Cross-Dataset Evaluation

7. **Run Cross-Evaluation Cells** (Cells 13-15):
   - Tests each model on other datasets
   - Generates 6 additional Excel files with cross-dataset results

**Execution Time**: ~30-45 minutes for all evaluations

---

## 🔍 Step 7: SHAP Interpretability Analysis

### 6.1 Run SHAP Analysis

1. Open: `SHAP_ANALYSIS.ipynb`
2. **Mount Google Drive** (Cell 1)
3. **Extract SWMH Dataset** (Cell 4) if needed
4. **Install Dependencies** (Cell 2):
   ```python
   !pip install afinn transformers datasets evaluate openpyxl shap
   ```
5. **Update Model Paths** (Cell 8):
   ```python
   RMHD_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/RMHDCombined8_roberta_large_constructive_learning_affinnemotion_scl_v1.zip'
   SWMH_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/pytorch_modelroberta_emotion_wloss_constructive.bin'
   RMHD_kaggle_weights = '/content/drive/MyDrive/x24144801-Thesis/Weights/rmhdkaggle_pytorch_modelroberta_emotion_wloss_constructive.bin'
   ```
6. **Run Evaluation** (Cell 9):
   - Evaluates all three models
   - Generates prediction Excel files
7. **Run SHAP Analysis** (Cell 10):
   - Analyzes 50 random samples per model
   - Generates visualizations and Excel files

**Expected Output**:
- ✅ SHAP visualizations showing token importance
- ✅ Excel files with top 20 influential tokens per sample
- ✅ Separate sheets for correct/incorrect predictions
- ✅ Files saved to: `/content/drive/MyDrive/x24144801-Thesis/Results Sheets/`

**Execution Time**: ~1-2 hours (SHAP is computationally intensive)

### 7.2 Customize SHAP Analysis

To analyze more/fewer samples, modify Cell 10:
```python
run_shap_analysis(model_swmh, file_swmh, {...}, sample_size=100)  # Change from 50 to 100
```

---

## 📁 Step 8: Download and Review Results

### 8.1 Training Outputs (from each training notebook)

**Location**: `/content/drive/MyDrive/x24144801-Thesis/Weights/`

Files generated:
- `*.zip` or `*.bin`: Model weights
- `metrics.json`: Evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization

### 8.2 Evaluation Outputs

**Location**: Current Colab directory or local folder

Files generated:
- `SWMH_predictions.xlsx`: In-domain SWMH predictions
- `RMHD_predictions.xlsx`: In-domain RMHD predictions
- `RMHDKaggle_predictions.xlsx`: In-domain RMHD Kaggle predictions
- `SWMHmodel_on_RMHDmapped.xlsx`: Cross-evaluation results
- `RMHDmodel_on_SWMHmapped.xlsx`: Cross-evaluation results
- (4 more cross-evaluation files)

### 8.3 SHAP Outputs

**Location**: `/content/drive/MyDrive/x24144801-Thesis/Results Sheets/`

Files generated:
- `SWMH_Model_SHAP_Analysis.xlsx`
- `RMHD_Model_SHAP_Analysis.xlsx`
- `RMHD_Kaggle_Model_SHAP_Analysis.xlsx`

Each Excel file contains 3 sheets:
- **ALL**: All analyzed samples
- **CORRECT**: Only correct predictions
- **INCORRECT**: Only incorrect predictions

### 8.4 Download Files

**From Google Colab**:
```python
from google.colab import files
files.download('SWMH_predictions.xlsx')
```

**From Google Drive**: Navigate to the folders and download manually

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory (OOM) Error**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in training configuration (Cell 15):
  ```python
  per_device_train_batch_size=4,  # Change from 8 to 4
  per_device_eval_batch_size=4,
  ```
- Use gradient accumulation:
  ```python
  gradient_accumulation_steps=2,
  ```
- Use Colab Pro for more GPU memory

#### 2. **Module Not Found Error**

**Error**: `ModuleNotFoundError: No module named 'afinn'`

**Solution**: Run the installation cell again:
```bash
!pip install afinn transformers datasets evaluate accelerate shap openpyxl safetensors scikit-learn
```

#### 3. **File Not Found Error**

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solutions**:
- Verify Google Drive is mounted: Check for `/content/drive` folder
- Check file paths are correct
- Ensure datasets are uploaded to correct locations
- Re-mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)
  ```

#### 4. **Model Loading Error**

**Error**: `Error loading model weights`

**Solutions**:
- Verify model file exists in Google Drive
- Check if model is `.zip` (use `from_zip=True`) or `.bin` (use `from_zip=False`)
- Re-train the model if file is corrupted

#### 5. **Label Mismatch Error**

**Error**: `ValueError: y_true and y_pred have different number of labels`

**Solution**: Ensure datasets have consistent label formats and mapping functions are correctly applied

#### 6. **SHAP Too Slow**

**Issue**: SHAP analysis taking too long

**Solutions**:
- Reduce sample size from 50 to 20:
  ```python
  run_shap_analysis(..., sample_size=20)
  ```
- Use only specific models instead of all three
- Run SHAP analysis separately for each model

#### 7. **Disconnected from Colab**

**Issue**: Colab disconnects during long training

**Solutions**:
- Use Colab Pro for longer runtimes
- Keep the browser tab active
- Use this script to prevent disconnection:
  ```javascript
  function ClickConnect(){
    console.log("Clicking connect");
    document.querySelector("colab-connect-button").click()
  }
  setInterval(ClickConnect, 60000)
  ```
  Paste in browser console (F12)

---

## ⏱️ Estimated Time Requirements

| Task | Duration | Notes |
|------|----------|-------|
| Environment Setup | 10-15 min | First time only |
| Dataset Upload | 5-10 min | Depends on internet speed |
| Training (per model) | 3-6 hours | T4 GPU on Colab |
| In-Domain Evaluation | 15-20 min | All 3 models |
| Cross-Dataset Evaluation | 30-45 min | 6 cross-evaluations |
| SHAP Analysis | 1-2 hours | 50 samples per model |
| **Total (all models)** | **12-20 hours** | Can be parallelized |

---

## 🎯 Quick Start Checklist

- [ ] Google Colab account created
- [ ] Google Drive folders created
- [ ] Datasets uploaded to Google Drive
- [ ] GPU enabled in Colab
- [ ] Google Drive mounted
- [ ] Dependencies installed
- [ ] Training notebook selected
- [ ] Dataset paths updated
- [ ] Training started (Run all cells)
- [ ] Model saved to Google Drive
- [ ] Evaluation notebook opened
- [ ] Model paths updated
- [ ] Evaluation completed
- [ ] SHAP analysis notebook opened
- [ ] SHAP analysis completed
- [ ] Results downloaded

---

## 📞 Support

For issues or questions:
- Check the **Troubleshooting** section above
- Review the main **README.md** for architecture details
- Open an issue on GitHub
- Contact: mansoor7tariq@gmail.com

---

## 🔄 Workflow Summary

```
┌─────────────────────┐
│  1. Setup Google    │
│     Drive & Colab   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Install         │
│     Dependencies    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Train Models    │
│     (3 notebooks)   │
│     [3-6h each]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Evaluate Models │
│     In-Domain +     │
│     Cross-Dataset   │
│     [45-60 min]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. SHAP Analysis   │
│     Interpretability│
│     [1-2 hours]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. Download &      │
│     Analyze Results │
└─────────────────────┘
```

---

## ✅ Success Indicators

You've successfully completed the setup when you have:

1. ✅ Three trained models saved in Google Drive
2. ✅ Confusion matrices generated for each model
3. ✅ Classification reports showing accuracy > 70%
4. ✅ Excel files with test predictions
5. ✅ Cross-dataset evaluation completed (6 files)
6. ✅ SHAP analysis Excel files with token attributions
7. ✅ All results downloadable and reviewable

