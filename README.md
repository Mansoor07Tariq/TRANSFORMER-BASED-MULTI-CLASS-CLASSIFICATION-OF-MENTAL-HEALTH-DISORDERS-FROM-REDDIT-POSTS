# Transformer-Based Multi-Class Classification of Mental Health Disorders from Reddit Posts

A comprehensive deep learning framework for classifying mental health disorders using RoBERTa-large with multi-task learning, emotion-aware features, and supervised contrastive learning.

**📁 Access Datasets, Weights & Results**: [Google Drive Folder](https://drive.google.com/drive/folders/1F7e-BxpRG240z-PROqJWcSeC7YIfi0pI?usp=sharing)

---

## 📋 Overview

This repository contains the implementation of a novel multi-task learning approach for mental health disorder classification from social media text. The model combines:

- **RoBERTa-large** as the base encoder
- **Multi-task learning**: Classification + Emotion intensity regression
- **Supervised Contrastive Learning (SCL)**: Improved class separation
- **AFINN-based emotion scoring**: Domain-specific feature engineering
- **Class weighting**: Handling imbalanced mental health data

---

## 📂 Repository Structure

### Training Notebooks

1. **`RMHD_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`**
   - Trains on RMHD Combined dataset (8 mental health categories)
   - Categories: Depression, Anxiety, Bipolar, ADHD, PTSD, OCD, Autism, Eating Disorders

2. **`RMHDkaggle_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`**
   - Trains on RMHD Kaggle dataset (5 categories)
   - Categories: Depression, Anxiety, Bipolar, Suicidewatch, Normal

3. **`swmh_roberta_large_constructive_learning_affinnemotion_scl_v1.ipynb`**
   - Trains on SWMH dataset (5 Reddit subreddits)
   - Categories: self.depression, self.Anxiety, self.bipolar, self.SuicideWatch, self.offmychest

### Analysis Notebooks

4. **`SHAP_ANALYSIS.ipynb`**
   - Token-level interpretability using SHAP values
   - Analyzes model decisions for correct and incorrect predictions
   - Identifies key words influencing mental health classifications
   - Generates comprehensive Excel reports with top influential tokens

5. **`Evaluation code.ipynb`**
   - In-domain evaluation: Tests each model on its respective test set
   - Cross-dataset evaluation: Tests generalization across different datasets
   - Label mapping system for cross-dataset compatibility
   - Generates detailed classification reports and Excel outputs

---

## 🏗️ Model Architecture

### RobertaSWMHClassifier

```
Input Text → RoBERTa-large Encoder → Dropout → Three Parallel Heads:
├── Classification Head (2-layer MLP with GELU + LayerNorm) → Class Logits
├── Regression Head (Linear layer) → Emotion Intensity Score
└── Projection Head (2-layer MLP) → Contrastive Learning Embeddings
```

### Multi-Task Loss Function

```
Total Loss = CE_Loss + 0.3 × MSE_Loss + 0.1 × SCL_Loss
```

- **CE_Loss**: Weighted Cross-Entropy for classification (handles class imbalance)
- **MSE_Loss**: Mean Squared Error for emotion intensity regression
- **SCL_Loss**: Supervised Contrastive Loss for better class separation

---

## 🔧 Key Features

### 1. Emotion Intensity Scoring
Uses AFINN lexicon to compute emotion intensity:
- Identifies negative sentiment words in text
- Normalizes scores from [-5, 0] to [1, 0]
- Computes mean intensity as auxiliary regression target

### 2. Class Weighting
- Computes inverse frequency weights for each class
- Addresses severe class imbalance in mental health data
- Applied to Cross-Entropy loss during training

### 3. Supervised Contrastive Learning
- Projects CLS embeddings to 64-dimensional space
- Pulls same-class samples together in embedding space
- Pushes different-class samples apart
- Temperature-scaled similarity (τ = 0.07)

### 4. Text Preprocessing
- URL removal
- Special character filtering (keeps only alphanumeric and basic punctuation)
- Whitespace normalization
- Lowercase conversion

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `roberta-large` |
| Learning Rate | 1e-6 |
| Batch Size | 8 |
| Max Epochs | 15 |
| Early Stopping Patience | 1 |
| Weight Decay | 0.05 |
| LR Scheduler | Cosine with 10% warmup |
| Label Smoothing | 0.1 |
| Max Sequence Length | 512 |
| Precision | FP16 (mixed precision) |

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1-Score**: Balanced F1 across all classes
- **Weighted F1-Score**: Class-size weighted F1
- **Per-Class Metrics**: Precision, Recall, F1 for each disorder
- **Confusion Matrix**: Visualizes class-wise performance
- **Macro Recall**: Average recall across classes

---

## 🔍 SHAP Interpretability

The SHAP analysis provides:
- Token-level attribution scores for each prediction
- Top 20 most influential words per sample
- Separate analysis for correct vs. incorrect predictions
- Excel reports with three sheets:
  - **ALL**: Complete analysis
  - **CORRECT**: Only correct predictions
  - **INCORRECT**: Only misclassifications

---

## 🔄 Cross-Dataset Evaluation

Tests model generalization with 6 cross-evaluations:

| Model | Test Dataset | Purpose |
|-------|-------------|---------|
| SWMH | RMHD | Reddit → Clinical dataset |
| SWMH | RMHD Kaggle | Reddit → Kaggle competition |
| RMHD | SWMH | Clinical → Reddit |
| RMHD | RMHD Kaggle | Clinical 8-class → 5-class |
| RMHD Kaggle | SWMH | Competition → Reddit |
| RMHD Kaggle | RMHD | Competition → Clinical |

Label mapping handles different taxonomies across datasets.

---

## 🚀 Usage

### Clone the Repository

```bash
git clone https://github.com/Mansoor07Tariq/TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS.git
cd TRANSFORMER-BASED-MULTI-CLASS-CLASSIFICATION-OF-MENTAL-HEALTH-DISORDERS-FROM-REDDIT-POSTS
```

### Training a Model

```python
# 1. Mount Google Drive and load data
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install afinn transformers datasets evaluate accelerate shap openpyxl safetensors scikit-learn

# 3. Run all cells in the training notebook
# Model will be saved to Google Drive automatically
```

### Running SHAP Analysis

```python
# 1. Set paths to trained model weights and test data
RMHD_weights = 'path/to/model.zip'
test_path = 'path/to/test.csv'

# 2. Run evaluation and SHAP analysis
model, predictions_file, id2label = evaluate_model(
    csv_path=test_path,
    weights_path=RMHD_weights,
    num_labels=8,
    from_zip=True
)

run_shap_analysis(
    model=model,
    test_csv_path=predictions_file,
    label2id={v: k for k, v in id2label.items()},
    sample_size=50
)
```

### Cross-Dataset Evaluation

```python
# 1. Load and map labels between datasets
df_test = pd.read_csv(test_path)
df_mapped = map_labels(df_test, source="RMHD", target="SWMH")
df_mapped.to_csv("mapped_test.csv", index=False)

# 2. Evaluate with mapped labels
evaluate_model(
    csv_path="mapped_test.csv",
    weights_path=model_weights,
    num_labels=5,
    from_zip=True
)
```

---

## 📦 Dependencies

```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.20.0
evaluate>=0.4.0
afinn>=0.1
shap>=0.41.0
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
safetensors>=0.3.0
```

---

## 📝 Datasets

### RMHD Combined (8 classes)
- 8 mental health disorder categories
- Train/Val/Test splits pre-defined
- Balanced across disorders

### RMHD Kaggle (5 classes)
- Kaggle competition format
- 5 major mental health conditions
- Includes "Normal" class

### SWMH (5 Reddit subreddits)
- Real Reddit posts from mental health communities
- 5 subreddit-based categories
- Natural language with slang and abbreviations

---

## 🎯 Key Results

- **Multi-task learning** improves performance over single-task baselines
- **Emotion scoring** provides meaningful auxiliary signal
- **Contrastive learning** enhances class separation
- **SHAP analysis** reveals interpretable decision patterns
- **Cross-dataset evaluation** demonstrates model generalization capabilities

---

## 📄 Model Outputs

Each training run produces:
- ✅ Best model checkpoint (saved to Google Drive)
- ✅ Training logs and metrics (accuracy, F1, loss curves)
- ✅ Confusion matrix visualization
- ✅ Classification report (precision, recall, F1 per class)
- ✅ Test predictions Excel file
- ✅ Updated datasets with emotion scores

SHAP analysis produces:
- ✅ Token-level attribution visualizations
- ✅ Excel files with top influential tokens
- ✅ Separate sheets for correct/incorrect predictions

Evaluation produces:
- ✅ Accuracy and F1 metrics
- ✅ Classification reports
- ✅ Excel files with predictions and true labels
- ✅ Cross-dataset performance metrics

---

## 🔬 Research Contributions

1. **Multi-task framework**: Combines classification with emotion regression
2. **Emotion-aware features**: AFINN-based intensity scoring for mental health text
3. **Contrastive learning**: Supervised contrastive loss for better representations
4. **Comprehensive evaluation**: Both in-domain and cross-dataset testing
5. **Interpretability**: SHAP-based token attribution analysis
6. **Practical deployment**: Handles class imbalance and dataset heterogeneity

---

## 👤 Author

**Mansoor Tariq**
- GitHub: [@Mansoor07Tariq](https://github.com/Mansoor07Tariq)
