from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
import DBSCAN
import numpy as np


centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]
X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.7
#minimum neighbouring points set to 3
minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] )

dbscan = DBSCAN.DBScan()
clusters = dbscan.fitEjemplo(eps, minPts, data)
idx , cluster = list(zip(*clusters))
cluster_df = pd.DataFrame(clusters, columns = ["idx", "cluster"])

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')
#plt.show()
plt.savefig('Imagenes/makeblobs.png')