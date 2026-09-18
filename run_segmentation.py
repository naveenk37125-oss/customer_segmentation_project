import os
import pandas as pd
from src.preprocess import generate_synthetic_data, calculate_rfm, transform_features
from src.model import find_optimal_k, train_kmeans
from src.visualize import plot_elbow, plot_clusters

def main():
    print("=" * 60)
    print("  E-COMMERCE CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preparation
    data_file = "data/online_retail.csv"
    if os.path.exists(data_file):
        print(f"[1/5] Loading transactional data from {data_file}...")
        df = pd.read_csv(data_file)
    else:
        print("[1/5] Dataset not found. Generating synthetic transactional dataset...")
        os.makedirs("data", exist_ok=True)
        df = generate_synthetic_data()
        df.to_csv(data_file, index=False)
        print(f"      Synthetic data created and saved to {data_file}")
        
    # Step 2: RFM Calculation
    print("[2/5] Calculating Recency, Frequency, and Monetary (RFM) metrics...")
    rfm = calculate_rfm(df)
    print(f"      Engineered RFM metrics for {len(rfm)} unique customers.")
    
    # Step 3: Feature Scaling
    print("[3/5] Applying Log Transformation & Standard Scaling...")
    rfm_scaled, scaler = transform_features(rfm)
    
    # Step 4: Model Optimization & Training
    print("[4/5] Computing Elbow Curve & Training K-Means Model (k=4)...")
    k_range, inertias, silhouettes = find_optimal_k(rfm_scaled)
    plot_elbow(k_range, inertias, output_path="elbow_curve.png")
    
    kmeans_model, labels = train_kmeans(rfm_scaled, n_clusters=4)
    rfm['Cluster'] = labels
    
    # Step 5: Save Results & Visualizations
    print("[5/5] Saving outputs...")
    output_csv = "segmented_customers.csv"
    rfm.to_csv(output_csv, index=False)
    plot_clusters(rfm, output_path="customer_clusters.png")
    
    print("-" * 60)
    print("SUCCESS: Pipeline completed successfully!")
    print(f" -> Segmented Data Saved: {output_csv}")
    print(f" -> Elbow Chart Saved:    elbow_curve.png")
    print(f" -> Clusters Plot Saved:  customer_clusters.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
