from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# 加载数据集
iris = datasets.load_iris()
wine = datasets.load_wine()

# 降维以便于可视化
pca_iris = PCA(n_components=2)
pca_wine = PCA(n_components=2)
iris_2d = pca_iris.fit_transform(iris.data)
wine_2d = pca_wine.fit_transform(wine.data)

# 初始化聚类算法
kmeans = KMeans(n_clusters=3)
agg_clust = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 应用聚类算法到Iris数据集
iris_kmeans_labels = kmeans.fit_predict(iris.data)
iris_agg_labels = agg_clust.fit_predict(iris.data)
iris_dbscan_labels = dbscan.fit_predict(iris.data)
dbscan = DBSCAN(eps=30, min_samples=8)
# 应用聚类算法到Wine数据集
wine_kmeans_labels = kmeans.fit_predict(wine.data)
wine_agg_labels = agg_clust.fit_predict(wine.data)
wine_dbscan_labels = dbscan.fit_predict(wine.data)

# 可视化函数
def plot_clusters(data_2d, labels, algorithm_name, dataset_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(f'{dataset_name} - {algorithm_name} Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

# Iris数据集可视化
plot_clusters(iris_2d, iris_kmeans_labels, 'KMeans', 'Iris')
plot_clusters(iris_2d, iris_agg_labels, 'Agglomerative', 'Iris')
plot_clusters(iris_2d, iris_dbscan_labels, 'DBSCAN', 'Iris')

# Wine数据集可视化
plot_clusters(wine_2d, wine_kmeans_labels, 'KMeans', 'Wine')
plot_clusters(wine_2d, wine_agg_labels, 'Agglomerative', 'Wine')
plot_clusters(wine_2d, wine_dbscan_labels, 'DBSCAN', 'Wine')
