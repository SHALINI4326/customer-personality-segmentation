# Customer Personality Segmentation using Machine Learning & Interactive Web App

An end-to-end Machine Learning data science and full-stack project for **Customer Personality Analysis and Unsupervised Segmentation**. This project extracts actionable behavioral personas from customer transaction and demographic records, enabling hyper-personalized marketing campaigns and revenue optimization.

---

## 🚀 Key Features

1. **Exploratory Data Analysis (EDA)**: Multi-category spend patterns (Wines, Fruits, Meat, Fish, Sweets, Gold), demographic distributions (Age, Income, Education, Family composition), and purchasing channel breakdowns.
2. **Feature Engineering**:
   - `Age`: Extracted from `Year_Birth` with outlier bounds.
   - `Total_Spent`: Sum of all six product categories.
   - `Children`: Sum of `Kidhome` and `Teenhome`.
   - `Total_Purchases`: Aggregation of web, catalog, and store transactions.
   - `Family_Size`: Marital status plus children count.
3. **Unsupervised ML Engine**:
   - Feature normalization via `StandardScaler`.
   - Optimal cluster detection using the **Elbow Method (Inertia)** and **Silhouette Analysis**.
   - **K-Means Clustering** ($k=4$) for distinct persona generation.
   - **Principal Component Analysis (PCA)** dimensionality reduction for 2D latent space projection.
4. **Interactive Modern Web App Frontend**:
   - Single-page responsive dashboard powered by Tailwind CSS and Chart.js.
   - **Interactive Live Customer Classifier**: Real-time Euclidean distance & centroid classification from user input or pre-loaded profiles.
   - **Marketing Campaign Simulator**: Test promotional response rates and calculate estimated ROI by segment.
   - **Cluster Deep Dives**: Multi-axis radar charts, behavioral traits, and tailored communication strategies.
5. **Production Backend API**: Built with FastAPI and Pydantic for high-performance inference.

---

## 👥 The 4 Customer Segments

| Segment | Persona Name | Key Demographics & Spending | Preferred Channel | Strategic Marketing Action |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **Stars / Luxury Connoisseurs** | High Income ($80k+), heavy wine & meat spend ($1,200+ total), few children | Physical Stores & Luxury Catalogs | Exclusive VIP concierge, wine tasting events, high-tier loyalty perks |
| **Cluster 1** | **Frugal / Budget Seekers** | Lower Income (<$35k), multiple children, high deal purchases, low spend | Coupon Blasts & Web Bargains | Bulk-family bundles, clearance flash sales, price-match guarantees |
| **Cluster 2** | **Prudent Middle-Class** | Average Income ($54k), moderate spend, balanced category consumption | Omnichannel (Store & Web) | Tiered loyalty cashback, holiday cross-sell promotions |
| **Cluster 3** | **Emerging Tech Professionals** | High Income ($67k), younger demographic, heavy web visits & online orders | E-Commerce Web Store & App | Digital app-exclusive drops, social media influencer campaigns, instant checkout rewards |

---

## 🛠️ Project Structure

```text
customer_personality_segmentation/
├── customer_segmentation_data.csv    # Benchmark dataset
├── train_model.py                    # Scikit-Learn training, PCA, and model export script
├── app_api.py                        # FastAPI REST inference service
├── customer_segmentation_notebook.ipynb # Comprehensive Jupyter Notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── frontend/
    └── index.html                    # Interactive web app
```

---

## ⚡ Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone <repo_url>
cd customer_personality_segmentation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Clustering Models
```bash
python train_model.py
```
This will preprocess the dataset, evaluate Silhouette scores for $k=2..6$, train the optimal K-Means model, fit PCA, and persist artifacts inside `models/`.

### 3. Launch FastAPI Backend
```bash
python app_api.py
```
API Documentation will be live at `http://localhost:8000/docs`.

### 4. Open Interactive Web App
Open `frontend/index.html` in any modern web browser or preview directly in Canvas!

### 5. screenshot
<img width="1920" height="1080" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/38f81d1d-c340-4f5d-9aab-04f13d1c07f5" />

### 6. visit site 
https://shalini4326.github.io/customer-personality-segmentation/
