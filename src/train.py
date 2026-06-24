import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(file_path: str):
    """Loads dataset from CSV."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def get_preprocessor(numeric_features: list) -> ColumnTransformer:
    """Builds a Scikit-Learn preprocessor to scale Time and Amount."""
    logger.info("Building preprocessing pipeline...")
    # Scale Time and Amount using RobustScaler to handle outliers
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), numeric_features)
        ],
        remainder='passthrough' # Keep V1-V28 features untouched
    )
    return preprocessor

def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """Calculates all key metrics for imbalanced classification."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please run src/download_data.py first.")
        return
        
    df = load_data(data_path)
    
    # Feature columns and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Train-test split (80-20) with stratification
    logger.info("Splitting dataset into train and test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Features to scale: Time and Amount
    scale_features = ['Time', 'Amount']
    preprocessor = get_preprocessor(scale_features)
    
    # Preprocess training and testing features
    logger.info("Fitting and transforming features...")
    # Get preprocessed column representations
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    
    # Anomaly rate in training data for Anomaly Detection (contamination rate)
    anomaly_rate = y_train.sum() / len(y_train)
    logger.info(f"Training Fraud Rate: {anomaly_rate * 100:.4f}%")
    
    results = {}
    fitted_models = {}
    
    # Dict to collect predictions on test set for visualization
    test_preds = {'y_test': y_test.values}
    
    # ----------------------------------------------------
    # Model 1: Baseline Logistic Regression (Imbalanced)
    # ----------------------------------------------------
    logger.info("Training Baseline Logistic Regression...")
    clf_lr = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr.fit(X_train_pre, y_train)
    
    y_pred = clf_lr.predict(X_test_pre)
    y_prob = clf_lr.predict_proba(X_test_pre)[:, 1]
    
    results['LR (Baseline)'] = evaluate_model(y_test, y_pred, y_prob)
    fitted_models['LR (Baseline)'] = clf_lr
    test_preds['LR (Baseline)_pred'] = y_pred
    test_preds['LR (Baseline)_prob'] = y_prob
    
    # ----------------------------------------------------
    # Model 2: Logistic Regression (Class Weighted)
    # ----------------------------------------------------
    logger.info("Training Class-Weighted Logistic Regression...")
    clf_lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_lr_weighted.fit(X_train_pre, y_train)
    
    y_pred = clf_lr_weighted.predict(X_test_pre)
    y_prob = clf_lr_weighted.predict_proba(X_test_pre)[:, 1]
    
    results['LR (Class Weighted)'] = evaluate_model(y_test, y_pred, y_prob)
    fitted_models['LR (Class Weighted)'] = clf_lr_weighted
    test_preds['LR (Class Weighted)_pred'] = y_pred
    test_preds['LR (Class Weighted)_prob'] = y_prob
    
    # ----------------------------------------------------
    # Model 3: Random Forest (Class Weighted)
    # ----------------------------------------------------
    logger.info("Training Class-Weighted Random Forest...")
    clf_rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train_pre, y_train)
    
    y_pred = clf_rf.predict(X_test_pre)
    y_prob = clf_rf.predict_proba(X_test_pre)[:, 1]
    
    results['Random Forest (Class Weighted)'] = evaluate_model(y_test, y_pred, y_prob)
    fitted_models['Random Forest (Class Weighted)'] = clf_rf
    test_preds['Random Forest (Class Weighted)_pred'] = y_pred
    test_preds['Random Forest (Class Weighted)_prob'] = y_prob
    
    # ----------------------------------------------------
    # Model 4: Logistic Regression + SMOTE (Oversampling)
    # ----------------------------------------------------
    logger.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)
    logger.info(f"Resampled Class Distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    
    logger.info("Training Logistic Regression on SMOTE data...")
    clf_lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr_smote.fit(X_train_res, y_train_res)
    
    y_pred = clf_lr_smote.predict(X_test_pre)
    y_prob = clf_lr_smote.predict_proba(X_test_pre)[:, 1]
    
    results['LR (SMOTE)'] = evaluate_model(y_test, y_pred, y_prob)
    fitted_models['LR (SMOTE)'] = clf_lr_smote
    test_preds['LR (SMOTE)_pred'] = y_pred
    test_preds['LR (SMOTE)_prob'] = y_prob
    
    # ----------------------------------------------------
    # Model 5: Hist Gradient Boosting (Class Weighted)
    # ----------------------------------------------------
    logger.info("Training Class-Weighted Hist Gradient Boosting...")
    clf_hgb = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
    clf_hgb.fit(X_train_pre, y_train)
    
    y_pred = clf_hgb.predict(X_test_pre)
    y_prob = clf_hgb.predict_proba(X_test_pre)[:, 1]
    
    results['Gradient Boosting (Class Weighted)'] = evaluate_model(y_test, y_pred, y_prob)
    fitted_models['Gradient Boosting (Class Weighted)'] = clf_hgb
    test_preds['Gradient Boosting (Class Weighted)_pred'] = y_pred
    test_preds['Gradient Boosting (Class Weighted)_prob'] = y_prob

    # ----------------------------------------------------
    # Model 6: Anomaly Detection - Isolation Forest (Unsupervised)
    # ----------------------------------------------------
    logger.info("Training Isolation Forest (Semi-supervised, on normal data only)...")
    # Fit only on normal transactions
    X_train_normal = X_train_pre[y_train == 0]
    
    # Set contamination to training fraud rate (or slightly higher for buffer)
    clf_iforest = IsolationForest(
        contamination=max(anomaly_rate, 0.001), 
        random_state=42, 
        n_jobs=-1
    )
    clf_iforest.fit(X_train_normal)
    
    # Predict on test set
    iforest_preds = clf_iforest.predict(X_test_pre)
    y_pred_iforest = np.where(iforest_preds == -1, 1, 0)
    
    # Outlier scores (lower score = more anomalous)
    scores = clf_iforest.score_samples(X_test_pre)
    # Invert scores so that higher score = more anomalous, suitable for Average Precision
    y_prob_iforest = -scores 
    
    results['Isolation Forest (Anomaly Detection)'] = evaluate_model(y_test, y_pred_iforest, y_prob_iforest)
    fitted_models['Isolation Forest (Anomaly Detection)'] = clf_iforest
    test_preds['Isolation Forest (Anomaly Detection)_pred'] = y_pred_iforest
    test_preds['Isolation Forest (Anomaly Detection)_prob'] = y_prob_iforest
    
    # ----------------------------------------------------
    # Model Comparison & Best Model Serialization
    # ----------------------------------------------------
    logger.info("Printing Model Evaluation Metrics:")
    for name, metrics in results.items():
        logger.info(f"{name}: F1={metrics['F1-Score']:.4f}, Recall={metrics['Recall']:.4f}, Precision={metrics['Precision']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}")
        
    # Select best model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
    best_model = fitted_models[best_model_name]
    logger.info(f"Best model based on F1-Score: {best_model_name} (F1={results[best_model_name]['F1-Score']:.4f})")
    
    # Save the pipeline (including preprocessor and model)
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    # Save best pipeline
    model_save_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    logger.info(f"Saved best model pipeline to {model_save_path}")
    
    # Save test predictions for visualizations
    preds_df = pd.DataFrame(test_preds)
    preds_csv_path = os.path.join(models_dir, "test_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    logger.info(f"Saved test predictions to {preds_csv_path}")
    
    # Save all metrics to a text file for visualizations
    metrics_path = os.path.join(models_dir, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write("Model,Precision,Recall,F1-Score,ROC-AUC,PR-AUC,TN,FP,FN,TP\n")
        for name, m in results.items():
            f.write(f"{name},{m['Precision']:.4f},{m['Recall']:.4f},{m['F1-Score']:.4f},{m['ROC-AUC']:.4f},{m['PR-AUC']:.4f},{m['TN']},{m['FP']},{m['FN']},{m['TP']}\n")
        f.write(f"\nBest Model: {best_model_name}\n")
    logger.info(f"Metrics summary saved to {metrics_path}")

if __name__ == "__main__":
    main()
