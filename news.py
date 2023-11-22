from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import time
import pandas as pd
selected_file_path='20 newsgroups/sci.space.txt'
# 读取文件内容
with open(selected_file_path, 'r', encoding='utf-8', errors='ignore') as file:
    newsgroup_data = file.read()

# 将文本数据分割为文档列表（假设每个文档由两个换行符分隔）
documents = newsgroup_data.split('\n\n')

# 使用TF-IDF转换器进行文本预处理（减少特征数量以节省内存）
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # 减少 max_features
X = vectorizer.fit_transform(documents)

# 初始化聚类算法
kmeans = KMeans(n_clusters=3)
agg_clust = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 应用K-means
start_time = time.time()
kmeans_labels = kmeans.fit_predict(X)
kmeans_time = time.time() - start_time
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# 应用层次聚类
start_time = time.time()
agg_labels = agg_clust.fit_predict(X.toarray())  # AgglomerativeClustering 不支持稀疏矩阵
agg_time = time.time() - start_time
agg_silhouette = silhouette_score(X.toarray(), agg_labels)

# 应用DBSCAN
start_time = time.time()
dbscan_labels = dbscan.fit_predict(X)
dbscan_time = time.time() - start_time
dbscan_silhouette = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A"

# 汇总结果
clustering_results = pd.DataFrame({
    'Algorithm': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Time (seconds)': [kmeans_time, agg_time, dbscan_time],
    'Silhouette Score': [kmeans_silhouette, agg_silhouette, dbscan_silhouette]
})

print(clustering_results)
