import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def download_dataset(url: str, dest_path: str, timeout: int = 10) -> bool:
    """Attempts to download the dataset from a URL."""
    print(f"Attempting to download dataset from {url}...")
    try:
        # Set a user-agent to avoid HTTP 403 Forbidden
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, 'wb') as out_file:
            # Download in chunks to monitor progress or allow aborting
            chunk_size = 1024 * 1024 # 1MB
            bytes_downloaded = 0
            start_time = time.time()
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_downloaded += len(chunk)
                mb = bytes_downloaded / (1024 * 1024)
                
                # Check for download speed / time limits to avoid hanging
                elapsed = time.time() - start_time
                if elapsed > 15:  # If it takes >15s, abort and use synthetic
                    print("\nDownload is taking too long. Aborting to use synthetic data generator...")
                    return False
                    
                print(f"Downloaded {mb:.1f} MB...", end='\r')
            print(f"\nDownload complete! File saved to {dest_path}")
            return True
    except Exception as e:
        print(f"\nDownload failed or timed out: {e}")
        return False

def generate_synthetic_data(dest_path: str, n_samples: int = 150000) -> None:
    """Generates a high-quality synthetic credit card fraud dataset."""
    print(f"Generating synthetic Credit Card Fraud dataset ({n_samples} samples)...")
    np.random.seed(42)
    
    # 1. Generate 28 PCA-like features V1-V28 using make_classification
    # Anomaly rate: 0.17% (255 fraud cases out of 150,000)
    X_pca, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=22,
        n_redundant=6,
        weights=[0.9983, 0.0017],
        random_state=42
    )
    
    # Create column names
    pca_cols = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(X_pca, columns=pca_cols)
    
    # 2. Add 'Time' feature (uniform distribution representing seconds over 2 days: 0 to 172800)
    df['Time'] = np.random.uniform(0, 172800, size=n_samples).astype(int)
    
    # 3. Add 'Amount' feature (log-normal/exponential-like distributions)
    # Fraud transactions generally have slightly different amounts
    amount_legit = np.random.exponential(scale=80, size=n_samples)
    amount_fraud = np.random.exponential(scale=150, size=n_samples)
    
    # Blend them based on target variable y
    df['Amount'] = np.where(y == 0, amount_legit, amount_fraud)
    df['Amount'] = np.round(df['Amount'], 2)
    
    # 4. Add the target variable 'Class'
    df['Class'] = y
    
    # Reorder columns: Time, V1-V28, Amount, Class (matching original Kaggle format)
    cols = ['Time'] + pca_cols + ['Amount', 'Class']
    df = df[cols]
    
    # Save to destination
    df.to_csv(dest_path, index=False)
    print(f"Synthetic dataset successfully created and saved to {dest_path}!")
    print(f"Class Distribution: {df['Class'].value_counts().to_dict()}")
    print(f"Fraud Rate: {(df['Class'].sum() / len(df)) * 100:.3f}%")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    dest_path = os.path.join(data_dir, "creditcard.csv")
    
    # A list of mirrors for creditcard.csv (anonymized 284k Kaggle dataset)
    # The first one is a direct raw download link from a small repository hosting it
    mirrors = [
        "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        "https://raw.githubusercontent.com/davidsbatista/machine-learning-notebooks/master/creditcard.csv"
    ]
    
    download_success = False
    for url in mirrors:
        # Attempt download with a 10s timeout
        if download_dataset(url, dest_path, timeout=10):
            download_success = True
            break
            
    if not download_success:
        print("\nUsing fallback synthetic generator...")
        # Clean up any partial download
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass
        generate_synthetic_data(dest_path)

if __name__ == "__main__":
    main()
