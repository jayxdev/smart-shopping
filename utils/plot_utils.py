# SmartShopping/utils/plot_utils.py
import matplotlib.pyplot as plt
import io

def plot_clusters(X_scaled, clusters):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", alpha=0.6)
    plt.title("Customer Segmentation Clusters")
    plt.xlabel("Standardized Age")
    plt.ylabel("Standardized Avg Order Value")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=120)
    img.seek(0)
    plt.close()
    return img