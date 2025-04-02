# SmartShopping/agents/customer_agent.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import sqlite3
from utils.plot_utils import plot_clusters

def customer_agent(users):
    # Create numerical features from list columns
    users['browse_count'] = users['Browsing_History'].apply(len)
    users['purchase_count'] = users['Purchase_History'].apply(len)
    
    # Create binary features for common categories
    categories = ['Books', 'Fashion', 'Fitness', 'Electronics', 'Beauty']
    for cat in categories:
        users[f'browsed_{cat.lower()}'] = users['Browsing_History'].apply(
            lambda x: 1 if cat in x else 0)
        users[f'purchased_{cat.lower()}'] = users['Purchase_History'].apply(
            lambda x: 1 if cat in x else 0)
    
    # Select numerical features for clustering
    numerical_features = [
        'Age', 'Avg_Order_Value', 'browse_count', 'purchase_count',
        *[f'browsed_{cat.lower()}' for cat in categories],
        *[f'purchased_{cat.lower()}' for cat in categories]
    ]
    
    X = users[numerical_features].fillna(0)
    
    # Scale only the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster users
    kmeans = KMeans(n_clusters=3, random_state=42)
    users["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Store in database
    conn = sqlite3.connect("ecommerce.db")
    users[["Customer_ID", "cluster"]].to_sql("user_clusters", conn, if_exists="replace", index=False)
    conn.close()
    
    # Generate visualization (using first two features for plotting)
    cluster_img = plot_clusters(X_scaled[:, :2], users["cluster"])
    
    return users, cluster_img