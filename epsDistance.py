import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
pd.options.mode.chained_assignment = None
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns

import preproceso

sns.set()

f = "datasets/train.csv"
df = pd.read_csv(f)
# df = df.head(100)
numTopics = 2
df, diccionario = preproceso.topicosTrain(df, numTopics)

topics = df["Topicos"]
X = np.zeros(shape=(df.shape[0], 2))

idx = 0
for fila in topics:
    X[idx] = np.array(fila)
    idx += 1

neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()
