#  Customer Segmentation Analysis

This project involves performing **customer segmentation** using unsupervised machine learning techniques. The goal is to cluster customers based on their demographics, purchasing behavior, and promotional engagement to aid targeted marketing strategies.

## 🔍 Problem Statement

Understanding customer personas is vital for personalized marketing. By clustering customers into meaningful segments using unsupervised learning, businesses can:

- Improve customer retention
- Design targeted promotions
- Boost product recommendations

## 💾 Data Description

The dataset includes features such as:

- `Age`
- `Income`
- `Spending Score`
- `Family Size`
- `Education Level`
- `Engagement with Promotions`

Preprocessing steps included:

- Handling missing values
- Scaling numerical variables using Min-Max Scaler
- Encoding categorical features

## 🧪 Models Used

The following clustering models were implemented and evaluated:

- **K-Means Clustering**
- **Hierarchical Clustering (Agglomerative)**
- **DBSCAN** *(Best performance – Silhouette Score ≈ 0.43)*
- **Gaussian Mixture Models (GMM)**

### 📊 Model Evaluation Metrics:
- Silhouette Score
- Davies-Bouldin Score
- Cluster Visualization using PCA/TSNE

## 🚀 Streamlit App

A Streamlit app is included for easy interaction with the pre-trained DBSCAN model. Users can:

- Upload customer data or manually input values
- Automatically preprocess and assign a cluster
- Handle noise/outlier detection (DBSCAN)

To run the app:

```bash
cd app
streamlit run app.py
