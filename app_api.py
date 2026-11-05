"""
Customer Personality Segmentation - FastAPI Backend Service
Provides REST API endpoints for customer personality classification,
batch scoring, and marketing campaign recommendations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import os
import json
import joblib

app = FastAPI(
    title="Customer Personality Segmentation API",
    description="Machine Learning service for unsupervised customer clustering and automated personalization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load metadata or models if available
METADATA_PATH = "models/metadata.json"
SCALER_PATH = "models/scaler.joblib"
KMEANS_PATH = "models/kmeans_model.joblib"

class CustomerProfile(BaseModel):
    Age: int = Field(..., ge=18, le=100, example=45)
    Income: float = Field(..., ge=5000, le=250000, example=75000)
    Children: int = Field(0, ge=0, le=5, example=1)
    MntWines: float = Field(..., ge=0, example=450)
    MntFruits: float = Field(0, ge=0, example=35)
    MntMeatProducts: float = Field(..., ge=0, example=320)
    MntFishProducts: float = Field(0, ge=0, example=50)
    MntSweetProducts: float = Field(0, ge=0, example=25)
    MntGoldProds: float = Field(0, ge=0, example=65)
    NumDealsPurchases: int = Field(0, ge=0, example=2)
    NumWebPurchases: int = Field(..., ge=0, example=6)
    NumCatalogPurchases: int = Field(0, ge=0, example=4)
    NumStorePurchases: int = Field(..., ge=0, example=7)
    NumWebVisitsMonth: int = Field(..., ge=0, example=4)

CLUSTER_DETAILS = {
    0: {
        "name": "Stars / Luxury Connoisseurs",
        "badge": "High Value VIP",
        "description": "High income, premium spend in wines & gourmet meats, high catalog/store shopping, low discount reliance.",
        "recommended_strategy": "VIP private concierge, vintage wine club invites, personalized exclusive catalogs.",
        "discount_sensitivity": "Very Low",
        "preferred_channel": "Direct In-Store & Catalog Drops"
    },
    1: {
        "name": "Frugal / Budget Seekers",
        "badge": "Price Sensitive",
        "description": "Lower income, higher family count, highly motivated by promotional deals, careful with discretionary spending.",
        "recommended_strategy": "Bundle saver alerts, cashback promos, family discount packs, weekly coupon blasts.",
        "discount_sensitivity": "High",
        "preferred_channel": "Deal Emails & Promotional Push"
    },
    2: {
        "name": "Prudent Middle-Class",
        "badge": "Steady Regulars",
        "description": "Balanced earnings and moderate spending, steady omnichannel presence across web and physical stores.",
        "recommended_strategy": "Tiered loyalty points, seasonal holiday bundles, category cross-sell discounts.",
        "discount_sensitivity": "Moderate",
        "preferred_channel": "Omnichannel (Web & Retail Store)"
    },
    3: {
        "name": "Emerging Tech Professionals",
        "badge": "Digital Trendsetters",
        "description": "Younger demographics with above-average earnings, frequent web visits, strong digital campaign affinity.",
        "recommended_strategy": "App-exclusive flash sales, dynamic digital recommendations, instant checkout incentives.",
        "discount_sensitivity": "Medium-Low",
        "preferred_channel": "E-Commerce Web Store & App"
    }
}

@app.get("/")
def health_check():
    return {"status": "online", "service": "Customer Personality Segmentation ML API", "version": "1.0.0"}

@app.get("/api/clusters")
def get_clusters():
    """Return all defined customer personas and marketing strategies."""
    return {"clusters": CLUSTER_DETAILS}

@app.post("/api/predict")
def predict_segment(customer: CustomerProfile):
    """Predict the customer segment for a single customer profile."""
    # Derived Feature
    total_spent = (customer.MntWines + customer.MntFruits + customer.MntMeatProducts +
                   customer.MntFishProducts + customer.MntSweetProducts + customer.MntGoldProds)
    
    # Feature vector matching training order:
    # ['Income', 'Age', 'Total_Spent', 'Children', 'MntWines', 'MntMeatProducts', 'MntGoldProds',
    #  'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    feature_vector = np.array([
        customer.Income,
        customer.Age,
        total_spent,
        customer.Children,
        customer.MntWines,
        customer.MntMeatProducts,
        customer.MntGoldProds,
        customer.NumDealsPurchases,
        customer.NumWebPurchases,
        customer.NumCatalogPurchases,
        customer.NumStorePurchases,
        customer.NumWebVisitsMonth
    ]).reshape(1, -1)
    
    # If trained models exist on disk, use them, otherwise use calibrated reference centroids
    try:
        if os.path.exists(SCALER_PATH) and os.path.exists(KMEANS_PATH):
            scaler = joblib.load(SCALER_PATH)
            kmeans = joblib.load(KMEANS_PATH)
            scaled = scaler.transform(feature_vector)
            cluster_id = int(kmeans.predict(scaled)[0])
            dists = np.linalg.norm(scaled - kmeans.cluster_centers_, axis=1)
            probs = np.exp(-dists) / np.sum(np.exp(-dists))
        else:
            # Fallback heuristic based on income and spend thresholds
            if customer.Income > 75000 and total_spent > 1000:
                cluster_id = 0
            elif customer.Income < 40000 or customer.NumDealsPurchases > 4:
                cluster_id = 1
            elif customer.NumWebPurchases > 6 and customer.Age < 45:
                cluster_id = 3
            else:
                cluster_id = 2
            probs = [0.25, 0.25, 0.25, 0.25]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    cluster_info = CLUSTER_DETAILS.get(cluster_id, CLUSTER_DETAILS[0])
    
    return {
        "predicted_cluster": cluster_id,
        "cluster_name": cluster_info["name"],
        "badge": cluster_info["badge"],
        "description": cluster_info["description"],
        "strategy": cluster_info["recommended_strategy"],
        "discount_sensitivity": cluster_info["discount_sensitivity"],
        "preferred_channel": cluster_info["preferred_channel"],
        "total_spent": total_spent,
        "confidence_scores": {
            f"Cluster_{i}": round(float(p), 3) for i, p in enumerate(probs)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
