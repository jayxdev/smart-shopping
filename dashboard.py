import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="SmartShopping Customer Analysis Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    users = pd.read_csv("data/customer_data_collection.csv")
    products = pd.read_csv("data/product_recommendation_data.csv")
    
    # Convert list columns to strings for hashing
    users['Browsing_History'] = users['Browsing_History'].apply(str)
    users['Purchase_History'] = users['Purchase_History'].apply(str)
    products['Similar_Product_List'] = products['Similar_Product_List'].apply(str)
    
    return users, products

users, products = load_data()

# Function to convert string back to list when needed
def safe_convert_to_list(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

# Function to prepare features for clustering
@st.cache_data
def prepare_features(users):
    users_copy = users.copy()
    users_copy['Browsing_History'] = users_copy['Browsing_History'].apply(safe_convert_to_list)
    users_copy['Purchase_History'] = users_copy['Purchase_History'].apply(safe_convert_to_list)
    
    # Prepare features for clustering
    features = users_copy[['Age', 'Avg_Order_Value']].copy()
    
    # Vectorize categorical data
    users_copy['Browsing_History_Str'] = users_copy['Browsing_History'].apply(lambda x: ' '.join(x))
    users_copy['Purchase_History_Str'] = users_copy['Purchase_History'].apply(lambda x: ' '.join(x))
    
    vectorizer_browse = CountVectorizer(max_features=10)
    browse_matrix = vectorizer_browse.fit_transform(users_copy['Browsing_History_Str']).toarray()
    browse_df = pd.DataFrame(browse_matrix, columns=[f"Browse_{i}" for i in range(browse_matrix.shape[1])])
    
    vectorizer_purchase = CountVectorizer(max_features=10)
    purchase_matrix = vectorizer_purchase.fit_transform(users_copy['Purchase_History_Str']).toarray()
    purchase_df = pd.DataFrame(purchase_matrix, columns=[f"Purchase_{i}" for i in range(purchase_matrix.shape[1])])
    
    # Combine all features
    features = pd.concat([features, browse_df, purchase_df], axis=1)
    features = features.fillna(0)  # Handle NaN values
    
    return features

# Function to create clusters dynamically with PCA
@st.cache_data
def create_clusters(users, features, n_clusters=4):
    """
    Performs PCA and clustering visualization with interactive 3D plot
    
    Parameters:
    - users: DataFrame with original user data
    - features: Processed features for clustering
    - n_clusters: Number of clusters to create
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
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
        'Avg_Order_Value': users['Avg_Order_Value'],
        'Top_Category': users['Purchase_History'].apply(
            lambda x: max(set(safe_convert_to_list(x)), key=safe_convert_to_list(x).count) if safe_convert_to_list(x) else 'None')
    })
    
    return viz_df, kmeans  # Return kmeans object for cluster centers

# Cluster profiling function
def profile_clusters(users, cluster_assignments):
    clustered_users = users.merge(cluster_assignments[['Customer_ID', 'Cluster']], on='Customer_ID', how='left')
    clustered_users['Purchase_History'] = clustered_users['Purchase_History'].apply(safe_convert_to_list)
    clustered_users['Browsing_History'] = clustered_users['Browsing_History'].apply(safe_convert_to_list)
    
    profiles = []
    category_affinity = {}
    
    for cluster in sorted(clustered_users['Cluster'].unique()):
        cluster_data = clustered_users[clustered_users['Cluster'] == cluster]
        
        purchased_items = [item for sublist in cluster_data['Purchase_History'] for item in sublist]
        purchased_series = pd.Series(purchased_items)
        browsed_items = [item for sublist in cluster_data['Browsing_History'] for item in sublist]
        browsed_series = pd.Series(browsed_items)
        
        profile = {
            'Cluster': cluster,
            'Size': f"{len(cluster_data)} ({len(cluster_data)/len(users)*100:.1f}%)",
            'Avg Age': f"{cluster_data['Age'].mean():.1f}",
            'Avg Order Value': f"${cluster_data['Avg_Order_Value'].mean():.2f}",
            'Top Purchased Category': purchased_series.mode()[0] if not purchased_series.empty else "None",
            'Top Browsed Category': browsed_series.mode()[0] if not browsed_series.empty else "None",
            'Avg Purchase Frequency': f"{cluster_data['Purchase_History'].apply(len).mean():.1f}",
            'Conversion Rate': f"{len(purchased_items)/len(browsed_items)*100:.1f}%" if browsed_items else "N/A",
            'Premium Customer': cluster_data['Avg_Order_Value'].mean() > users['Avg_Order_Value'].mean()
        }
        profiles.append(profile)
        
        category_affinity[cluster] = {
            'purchased': dict(purchased_series.value_counts(normalize=True).items()),
            'browsed': dict(browsed_series.value_counts(normalize=True).items())
        }
    
    return pd.DataFrame(profiles), category_affinity

# Strategy generation function
def generate_strategies(cluster_profiles, clustered_users):
    cluster_profiles['Avg Order Value'] = cluster_profiles['Avg Order Value'].replace(r'[\\$,]', '', regex=True).astype(float)
    
    strategies = []
    for _, profile in cluster_profiles.iterrows():
        strategy = "Premium upsell opportunities" if profile['Avg Order Value'] > clustered_users['Avg_Order_Value'].mean() else "Value bundle recommendations"
        strategies.append({
            'Cluster': profile['Cluster'],
            'Targeting Strategy': strategy,
            'Recommended Campaign': f"Focus on {profile['Top Purchased Category']} products",
            'Channel': "Email" if float(profile['Avg Age']) > 35 else "Social Media"
        })
    
    return pd.DataFrame(strategies)

# Dashboard layout
st.title("SmartShopping Customer Analysis Dashboard")
st.markdown("### Pre-Clustering EDA and Customer Segmentation Insights")

# Sidebar for navigation and cluster control
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Cluster Profiles", "Targeting Strategies"])

# Dynamic cluster number selection
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=24, value=6, step=2)
features = prepare_features(users)
cluster_assignments, kmeans_model = create_clusters(users, features, n_clusters)
profiles, category_affinity = profile_clusters(users, cluster_assignments)
strategies_df = generate_strategies(profiles, users.merge(cluster_assignments[['Customer_ID', 'Cluster']], on='Customer_ID'))

# Overview Section
if section == "Overview":
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Users Data Shape:**", users.shape)
        st.dataframe(users.head())
    with col2:
        st.write("**Products Data Shape:**", products.shape)
        st.dataframe(products.head())

# Cluster Profiles Section
elif section == "Cluster Profiles":
    st.header("Cluster Profiles")
    st.dataframe(profiles)
    
    fig = px.scatter_3d(
        cluster_assignments,
        x='PC1',
        y='PC2',
        z='Avg_Order_Value',
        color='Cluster',
        hover_data=['Customer_ID', 'Age', 'Top_Category'],
        title=f'Customer Segments (PCA + K-Means, {n_clusters} clusters)',
        labels={'PC1': 'Shopping Frequency',
               'PC2': 'Category Preference',
               'Avg_Order_Value': 'Spending Level'},
        height=800,
    )
    
    # Add cluster centers
    centers = kmeans_model.cluster_centers_
    fig.add_trace(
        px.scatter_3d(
            pd.DataFrame({
                'PC1': centers[:, 0],
                'PC2': centers[:, 1],
                'Avg_Order_Value': [cluster_assignments['Avg_Order_Value'].mean()]*n_clusters,
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

    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster selection for detailed analysis
    cluster = st.selectbox("Select Cluster for Detailed Analysis", sorted(profiles['Cluster'].unique()))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if category_affinity[cluster]['purchased']:
            fig_purchased = px.pie(
                names=list(category_affinity[cluster]['purchased'].keys()),
                values=list(category_affinity[cluster]['purchased'].values()),
                title=f"Cluster {cluster} Purchase Distribution"
            )
            st.plotly_chart(fig_purchased)
    
    with col2:
        if category_affinity[cluster]['browsed']:
            fig_browsed = px.pie(
                names=list(category_affinity[cluster]['browsed'].keys()),
                values=list(category_affinity[cluster]['browsed'].values()),
                title=f"Cluster {cluster} Browsing Distribution"
            )
            st.plotly_chart(fig_browsed)

# Targeting Strategies Section
elif section == "Targeting Strategies":
    st.header("Targeting Strategies")
    st.dataframe(strategies_df)
    
    # Bar chart for dominant categories
    fig_bar = px.bar(
        pd.DataFrame({
            'Cluster': list(category_affinity.keys()),
            'Top Categories': [
                sorted(data['purchased'], key=lambda x: x[1], reverse=True)[0][0] 
                if data['purchased'] else "None"
                for data in category_affinity.values()
            ]
        }),
        x='Cluster',
        y='Top Categories',
        color='Cluster',
        title='Dominant Categories by Cluster',
        labels={'Top Categories': 'Most Purchased Category'}
    )
    st.plotly_chart(fig_bar)

# Run the app
if __name__ == "__main__":
    st.sidebar.markdown("Built with Streamlit")