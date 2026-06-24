# Credit Card Fraud Detection: Imbalanced Data & Anomaly Detection

An end-to-end machine learning and anomaly detection project to identify fraudulent credit card transactions. 

Fraud detection presents a classic **highly imbalanced classification** problem (only **0.173%** of all transactions are fraudulent). This project implements, compares, and evaluates multiple strategies for handling imbalanced data (Synthetic Minority Over-sampling Technique (SMOTE), class weighting, and unsupervised anomaly detection) across various supervised and unsupervised models.

---

## 🚀 Key Features

* **Anonymized Real Dataset**: Loaded 284,807 transactions with PCA-anonymized features (`V1-V28`), `Time`, and `Amount`.
* **Addressing Extreme Imbalance**:
  * **SMOTE (Oversampling)**: Synthetically generates minority class instances during training.
  * **Cost-Sensitive Class Weighting**: Penalizes minority class misclassifications heavily.
* **Supervised Classification Comparison**:
  * Baseline Logistic Regression (no balancing)
  * Class-Weighted Logistic Regression
  * Class-Weighted Random Forest
  * Logistic Regression trained with SMOTE
  * Class-Weighted Hist Gradient Boosting (LightGBM-like)
* **Unsupervised Anomaly Detection**:
  * **Isolation Forest**: Trained in a semi-supervised fashion (on legitimate transactions only) to detect anomalous transactions.
* **Robust Evaluation Metrics**: Prioritizes **Precision-Recall AUC (PR-AUC)**, **F1-Score**, and **Recall** over misleading accuracy scores.
* **Production-Ready Pipeline**: Modular codebase utilizing Scikit-learn pipelines to scale features robustly and avoid data leakage.
* **Interactive Inference CLI**: A manual prediction helper to evaluate transaction inputs in real-time.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml   # GitHub Actions automated training CI workflow
├── data/
│   └── creditcard.csv        # Anonymized transactions dataset (ignored by git)
├── src/
│   ├── download_data.py      # Automated data download (with synthetic fallback)
│   ├── train.py              # ML modeling, SMOTE, anomaly training script
│   ├── visualize.py          # Script generating Precision-Recall and ROC curves
│   └── predict_manual.py     # Interactive manual inference script
├── plots/                    # Evaluation charts (generated dynamically)
│   ├── class_distribution.png
│   ├── precision_recall_curves.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── metrics_comparison.png
├── models/
│   ├── best_model.joblib     # Serialized best model pipeline
│   ├── test_predictions.csv  # Test set prediction logs
│   └── metrics_summary.txt   # Metrics comparison file
├── .gitignore                # Project git exclusion rules
├── requirements.txt          # Python dependencies
├── run_all.py                # Unified runner (downloads data, trains models, creates plots)
├── push_to_github.bat        # Double-clickable Windows batch script to easily upload code
└── README.md                 # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository and navigate to the project directory:
```bash
git clone <your-repo-url>
cd credit_card_fraud_detection
```

### 2. Install dependencies:
It is recommended to run this inside a virtual environment.
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### 1. Run the Entire Pipeline (One-Step Run):
To download the dataset, train/evaluate all models, and generate the validation plots with a single command:
```bash
python run_all.py
```

---

### 2. Alternative Step-by-Step Run:

#### Step A. Acquire Data:
```bash
python src/download_data.py
```

#### Step B. Train Models:
```bash
python src/train.py
```

#### Step C. Generate Visualizations:
```bash
python src/visualize.py
```

---

### 3. Run Manual Inference (Interactive Tester):
Evaluate custom transactions interactively:
```bash
python src/predict_manual.py
```

---

## 📊 Model Performance & Comparison

Models evaluated on a stratified 80/20 train/test split. Precision-Recall AUC (PR-AUC) and F1-Score are the primary optimization metrics.

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Class Weighted)** | **0.9186** | **0.8061** | **0.8587** | **0.9749** | **0.8649** |
| **LR (Baseline)** | 0.8289 | 0.6429 | 0.7241 | 0.9427 | 0.7388 |
| **Gradient Boosting (Class Weighted)** | 0.3398 | 0.8980 | 0.4930 | 0.9734 | 0.7449 |
| **Isolation Forest (Anomaly)** | 0.2313 | 0.3163 | 0.2672 | 0.6575 | 0.1611 |
| **LR (Class Weighted)** | 0.0609 | 0.9184 | 0.1142 | 0.9634 | 0.7175 |
| **LR (SMOTE)** | 0.0589 | 0.9184 | 0.1106 | 0.9629 | 0.7251 |

### 🏆 Key Observations
* **Supervised Win**: **Class-Weighted Random Forest** achieved the best overall F1-Score (**0.8587**) and PR-AUC (**0.8649**), maintaining high precision (91.8%) and detecting 80.6% of frauds.
* **Sensitivity Trade-off**: Logistic Regression with Class Weights or SMOTE oversampling maximises recall (**91.8%**), catching almost all fraud. However, they suffer from high False Positives, dropping precision to ~6%.
* **Anomaly Detection Limitations**: Unsupervised Isolation Forest (F1: **26.7%**) acts as a baseline when labels are unavailable, but underperforms compared to supervised models that utilize historical fraud labels.

---

## 📈 Visualizations

### 1. Class Imbalance
Visualizing the massive class imbalance: only **492 out of 284,807 transactions** in the dataset are fraudulent.

![Class Distribution](plots/class_distribution.png)

### 2. Precision-Recall Curves
Precision-Recall curves are the gold standard for evaluating imbalanced classification. The Random Forest model dominates the upper-right quadrant.

![Precision-Recall Curves](plots/precision_recall_curves.png)

### 3. Receiver Operating Characteristic (ROC) Curves
ROC curves can be overly optimistic for highly imbalanced datasets, which is why PR curves are prioritized.

![ROC Curves](plots/roc_curves.png)

### 4. Confusion Matrices Comparison
Comparing the actual confusion matrices. Note how Random Forest keeps false positives very low while detecting the majority of frauds, whereas Class Weighted / SMOTE models have higher false alarm rates.

![Confusion Matrices](plots/confusion_matrices.png)

### 5. Model Metric Comparison
![Metrics Comparison](plots/metrics_comparison.png)

---

## 🛠️ CI/CD & GitHub Pages Deployment

### 1. GitHub Actions (Continuous Integration)
This repository includes a pre-configured GitHub Actions workflow: [.github/workflows/ml_pipeline.yml](.github/workflows/ml_pipeline.yml).
- **Trigger**: Every push or pull request to the `main` or `master` branch.
- **Workflow**: Sets up Python, installs dependencies, fetches the dataset, trains all models, selects the best pipeline, and exports generated plots and model binaries as downloadable artifacts.

### 2. GitHub Pages (Dashboard Hosting)
To view the visual project dashboard live on the web:
1. Navigate to your repository settings on GitHub.
2. Select **Pages** from the left-hand navigation.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Choose the `main` branch and select `/ (root)` folder, then click **Save**.
5. Your dashboard will be live at: `https://<your-username>.github.io/credit_card_fraud_detection/`

