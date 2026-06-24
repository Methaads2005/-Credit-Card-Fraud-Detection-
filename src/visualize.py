import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def plot_class_distribution(df: pd.DataFrame, output_dir: str):
    """Plots the distribution of classes (Normal vs Fraud)."""
    print("Plotting class distribution...")
    plt.figure(figsize=(8, 6))
    
    class_counts = df['Class'].value_counts()
    total = len(df)
    
    barplot = sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='coolwarm', legend=False)
    
    for i, p in enumerate(barplot.patches):
        val = class_counts.values[i]
        pct = (val / total) * 100
        barplot.annotate(f"{val:,}\n({pct:.3f}%)",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 10),
                         textcoords='offset points',
                         fontsize=12, fontweight='bold')
                         
    plt.title("Transaction Class Distribution (Imbalanced Data)")
    plt.xlabel("Class (0: Legitimate, 1: Fraudulent)")
    plt.ylabel("Transaction Count")
    plt.xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()

def plot_precision_recall_curves(test_preds_df: pd.DataFrame, output_dir: str):
    """Plots Precision-Recall curves for all models on the same chart."""
    print("Plotting Precision-Recall curves...")
    plt.figure(figsize=(10, 8))
    
    y_test = test_preds_df['y_test']
    
    # Models are prefixes in column names ending with '_prob'
    prob_cols = [col for col in test_preds_df.columns if col.endswith('_prob')]
    
    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    for col, color in zip(prob_cols, colors):
        model_name = col.replace('_prob', '')
        y_prob = test_preds_df[col]
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        # Calculate Average Precision (AP)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, y_prob)
        
        plt.plot(recall, precision, color=color, label=f"{model_name} (AP = {ap:.4f})", linewidth=2)
        
    plt.title("Precision-Recall Curves (Focus on Fraud Class)")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=150)
    plt.close()

def plot_roc_curves(test_preds_df: pd.DataFrame, output_dir: str):
    """Plots ROC curves for all models."""
    print("Plotting ROC curves...")
    plt.figure(figsize=(10, 8))
    
    y_test = test_preds_df['y_test']
    prob_cols = [col for col in test_preds_df.columns if col.endswith('_prob')]
    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    
    for col, color in zip(prob_cols, colors):
        model_name = col.replace('_prob', '')
        y_prob = test_preds_df[col]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, label=f"{model_name} (AUC = {roc_auc:.4f})", linewidth=2)
        
    # Plot random chance line
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()

def plot_confusion_matrices(test_preds_df: pd.DataFrame, output_dir: str):
    """Plots confusion matrices of key models in a subplot grid."""
    print("Plotting confusion matrices...")
    
    # Select 4 representative models
    models_to_plot = [
        'LR (Baseline)',
        'Random Forest (Class Weighted)',
        'Gradient Boosting (Class Weighted)',
        'Isolation Forest (Anomaly Detection)'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    y_test = test_preds_df['y_test']
    
    for i, name in enumerate(models_to_plot):
        pred_col = f"{name}_pred"
        if pred_col not in test_preds_df.columns:
            continue
            
        y_pred = test_preds_df[pred_col]
        cm = confusion_matrix(y_test, y_pred)
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
                    
        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
        axes[i].set_xticklabels(['Legit (0)', 'Fraud (1)'])
        axes[i].set_yticklabels(['Legit (0)', 'Fraud (1)'])
        
    plt.suptitle("Confusion Matrices Comparison", fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()

def plot_metrics_comparison(metrics_path: str, output_dir: str):
    """Plots model metrics comparison (grouped bar chart for Precision, Recall, F1)."""
    print("Plotting metrics comparison...")
    if not os.path.exists(metrics_path):
        return
        
    rows = []
    with open(metrics_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # skip header
            if line.strip() == "" or "Best Model" in line:
                break
            parts = line.strip().split(',')
            rows.append({
                'Model': parts[0],
                'Precision': float(parts[1]),
                'Recall': float(parts[2]),
                'F1-Score': float(parts[3])
            })
            
    df = pd.DataFrame(rows)
    
    # Melt dataframe for grouped bar plotting
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Value')
                        
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted, palette='viridis')
    
    plt.title("Model Metrics Comparison (Precision, Recall, F1-Score)")
    plt.ylabel("Score Value")
    plt.xlabel("Models")
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    preds_csv_path = os.path.join(base_dir, "models", "test_predictions.csv")
    metrics_path = os.path.join(base_dir, "models", "metrics_summary.txt")
    plots_dir = os.path.join(base_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Class Distribution
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        plot_class_distribution(df, plots_dir)
        
    # 2. Model performance comparisons
    if os.path.exists(preds_csv_path):
        test_preds_df = pd.read_csv(preds_csv_path)
        plot_precision_recall_curves(test_preds_df, plots_dir)
        plot_roc_curves(test_preds_df, plots_dir)
        plot_confusion_matrices(test_preds_df, plots_dir)
        
    # 3. Bar comparisons
    if os.path.exists(metrics_path):
        plot_metrics_comparison(metrics_path, plots_dir)
        
    print(f"All visualizations successfully created and saved in: {plots_dir}/")

if __name__ == "__main__":
    main()
