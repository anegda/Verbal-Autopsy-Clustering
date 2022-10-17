import pandas as pd
pd.options.mode.chained_assignment = None
import DBSCAN
import numpy as np
import preproceso


f="datasets/train.csv"
df = pd.read_csv(f)
df = df.head(100)
df, diccionario = preproceso.topicosTrain(df, 2)
#DE AQUÍ PARA ABAJO PASA ALGO RARO Y NO DEBERÍA

dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(0.01, 3, df)
idx , cluster = list(zip(*clusters))
print(sorted(list(idx)))
resultados = pd.DataFrame()
resultados["Indice"] = idx
resultados["Cluster"] = cluster
resultados.to_csv('Prueba.csv')
cluster_df = pd.DataFrame(clusters, columns = ["idx", "cluster"])


#INSERTAR DICCIONARIO
fTest = "datasets/test.csv"
dfTest = pd.read_csv(fTest)
dfTest = preproceso.topicosTest(dfTest, diccionario)
print(dfTest.head())

"""
#EJEMPLO!!
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]
X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.6
#minimum neighbouring points set to 3
minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] )

dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(eps, minPts, data)
idx , cluster = list(zip(*clusters))
cluster_df = pd.DataFrame(clusters, columns = ["idx", "cluster"])

plt.figure(figsize=(10,7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0], X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1], s=10, label=f"Cluster{clust}")

plt.legend([f"Cluster {clust}" for clust in np.unique(cluster)], loc ="lower right")
plt.title('Clustered Data')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
"""