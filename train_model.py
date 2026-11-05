"""
Customer Personality Segmentation - Model Training Pipeline
Dataset: Customer Personality Analysis (Kaggle)
Algorithm: K-Means Clustering + Principal Component Analysis (PCA)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load dataset, handle missing values and perform domain feature engineering."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Handle missing values
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # 2. Feature Engineering
    # Calculate Age
    reference_year = 2026
    df['Age'] = reference_year - df['Year_Birth']
    
    # Remove outliers
    df = df[(df['Age'] > 18) & (df['Age'] < 95)]
    df = df[df['Income'] < 200000]
    
    # Total Spending across all product categories
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spent'] = df[spend_cols].sum(axis=1)
    
    # Total Children & Family Size
    df['Children'] = df['Kidhome'] + df['Teenhome']
    marital_mapping = {'Married': 2, 'Together': 2, 'Single': 1, 'Divorced': 1, 'Widow': 1, 'Absurd': 1, 'YOLO': 1}
    df['Family_Size'] = df['Marital_Status'].map(marital_mapping).fillna(1) + df['Children']
    df['Is_Parent'] = (df['Children'] > 0).astype(int)
    
    # Total Purchases
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    
    print(f"Preprocessing completed. Remaining records: {len(df)}")
    return df

def train_clustering_pipeline(df: pd.DataFrame, n_clusters: int = 4):
    """Scale features, perform PCA dimensionality reduction, and fit K-Means."""
    features = [
        'Income', 'Age', 'Total_Spent', 'Children',
        'MntWines', 'MntMeatProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
        'NumWebVisitsMonth'
    ]
    
    X = df[features]
    
    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow and Silhouette evaluation
    print("\n--- Evaluating Cluster Quality ---")
    silhouette_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = round(score, 4)
        print(f"k = {k} | Silhouette Score: {score:.4f} | Inertia: {km.inertia_:.2f}")
    
    # Fit Final KMeans model
    print(f"\nTraining optimal K-Means model with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # Segment Persona Profiling
    cluster_names = {
        0: "Stars / Luxury Connoisseurs",
        1: "Frugal / Budget Seekers",
        2: "Prudent Middle-Class",
        3: "Emerging Tech Professionals"
    }
    df['Segment_Name'] = df['Cluster'].map(cluster_names)
    
    print("\n--- Cluster Summary Metrics ---")
    summary = df.groupby('Cluster').agg({
        'Income': 'mean',
        'Age': 'mean',
        'Total_Spent': 'mean',
        'Children': 'mean',
        'MntWines': 'mean',
        'MntMeatProducts': 'mean',
        'NumDealsPurchases': 'mean',
        'NumWebPurchases': 'mean',
        'NumCatalogPurchases': 'mean',
        'NumStorePurchases': 'mean',
        'Response': 'mean'
    }).round(2)
    print(summary)
    
    # Export artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(kmeans, 'models/kmeans_model.joblib')
    joblib.dump(pca, 'models/pca_model.joblib')
    
    metadata = {
        'features': features,
        'n_clusters': n_clusters,
        'cluster_names': cluster_names,
        'silhouette_scores': silhouette_scores,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist()
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\nModels and metadata successfully saved to 'models/' directory.")
    return df, scaler, kmeans, pca

if __name__ == "__main__":
    data_path = 'customer_segmentation_data.csv'
    if os.path.exists(data_path):
        processed_df = load_and_preprocess(data_path)
        train_clustering_pipeline(processed_df, n_clusters=4)
    else:
        print(f"Error: {data_path} not found.")
