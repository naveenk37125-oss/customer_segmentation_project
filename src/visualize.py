import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(k_range, inertias, output_path="elbow_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_clusters(rfm, output_path="customer_clusters.png"):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=rfm['Cluster'].nunique())
    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette=palette,
        style='Cluster',
        s=100,
        alpha=0.8
    )
    plt.title('Customer Segments (Recency vs Monetary)', fontsize=14, fontweight='bold')
    plt.xlabel('Recency (Days)', fontsize=12)
    plt.ylabel('Monetary Value ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
