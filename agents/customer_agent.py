# SmartShopping/agents/customer_agent.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import sqlite3
from utils.plot_utils import plot_clusters
from sklearn.decomposition import PCA
import plotly.express as px


def cluster_users(users, features, n_clusters=12):
    """
    Performs PCA and clustering visualization with interactive plot
    
    Parameters:
    - users: DataFrame with original user data
    - features: Processed features for clustering
    - n_clusters: Number of clusters to create
    """
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)
    
    # Cluster in PCA space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(principal_components)
    
    # Create visualization DataFrame
    viz_df = pd.DataFrame({
        'Customer_ID': users['Customer_ID'],
        'PC1': principal_components[:, 0],
        'PC2': principal_components[:, 1],
        'Cluster': clusters,
        'Age': users['Age'],
        'Gender': users['Gender'],
        'Location': users['Location'],
        'Holiday': users['Holiday'],
        'Season': users['Season'],
        'Customer_Segment': users['Customer_Segment'],
        'Avg_Order_Value': users['Avg_Order_Value'],
        'Top_Category': users['Purchase_History'].apply(
            lambda x: max(set(x), key=x.count) if x else 'None')
    })
    
    # Interactive 3D plot
    fig = px.scatter_3d(
        viz_df,
        x='PC1',
        y='PC2',
        z='Avg_Order_Value',
        color='Cluster',
        hover_data=['Customer_ID', 'Age', 'Top_Category','Holiday', 'Season', 'Gender', 'Location'],
        title=f'Customer Segments (PCA + K-Means, {n_clusters} clusters)',
        labels={'PC1': 'Shopping Frequency',
               'PC2': 'Category Preference',
               'Avg_Order_Value': 'Spending Level'},
        height=800
    )
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    fig.add_trace(
        px.scatter_3d(
            pd.DataFrame({
                'PC1': centers[:, 0],
                'PC2': centers[:, 1],
                'Avg_Order_Value': [viz_df['Avg_Order_Value'].mean()]*n_clusters,
                'Cluster': range(n_clusters)
            }),
            x='PC1',
            y='PC2',
            z='Avg_Order_Value',
            color='Cluster',
            symbol_sequence=['x']
        ).data[0]
    )
    
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    # Return cluster assignments
    return viz_df[['Customer_ID', 'Cluster']], fig

def customer_agent(users, products, n_clusters=12):
    """
    Analyze customer data and cluster users based on their behavior.            "
    """
    # Create numerical features from list columns
    users['browse_count'] = users['Browsing_History'].apply(len)
    users['purchase_count'] = users['Purchase_History'].apply(len)
    
    # Create binary features for common categories
    categories = products['Category'].unique()
    subcategories = products['Subcategory'].unique()

    # Create binary features for subcategories
    for subcat in subcategories:
        users[f'purchased_{subcat.lower()}'] = users['Purchase_History'].apply(
            lambda x: 1 if subcat in x else 0)

    for cat in categories:
        users[f'browsed_{cat.lower()}'] = users['Browsing_History'].apply(
            lambda x: 1 if cat in x else 0)
        
    
    # Select numerical features for clustering
    numerical_features = [
        'Age', 'Avg_Order_Value', 'browse_count', 'purchase_count',
        *[f'browsed_{cat.lower()}' for cat in categories],
        *[f'purchased_{subcat.lower()}' for subcat in subcategories]
    ]
    
    X = users[numerical_features].fillna(0)
    
    # Scale only the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    # Cluster users
    clusters, fig=cluster_users(users, X_scaled, n_clusters=n_clusters)
    users["cluster"] = clusters["Cluster"]
    
    # Store in database
    conn = sqlite3.connect("ecommerce.db")
    users[["Customer_ID", "cluster"]].to_sql("user_clusters", conn, if_exists="replace", index=False)
    conn.close()
    
    return users, fig