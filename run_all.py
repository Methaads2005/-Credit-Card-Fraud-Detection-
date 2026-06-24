import os
import sys

# Add src directory to python module search path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

try:
    import download_data
    import train
    import visualize
except ImportError as e:
    print(f"Import Error: Could not load source files. {e}")
    sys.exit(1)

def main():
    print("="*60)
    print("   Running Credit Card Fraud Detection Pipeline")
    print("="*60)
    
    # Step 1: Data Ingestion
    print("\n[Step 1/3] Fetching transaction dataset...")
    download_data.main()
    
    # Step 2: Training & Evaluation
    print("\n[Step 2/3] Training classifiers & anomaly detection models...")
    train.main()
    
    # Step 3: Visualization
    print("\n[Step 3/3] Generating performance charts...")
    visualize.main()
    
    print("\n" + "="*60)
    print("   Pipeline Execution Completed Successfully!")
    print("   Plots directory:  C:\\indhrametha\\credit_card_fraud_detection\\plots")
    print("   Models directory: C:\\indhrametha\\credit_card_fraud_detection\\models")
    print("="*60)

if __name__ == "__main__":
    main()
