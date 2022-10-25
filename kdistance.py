import sys
import numpy as np
import pandas as pd
from nltk.metrics import distance
from sklearn.datasets import make_blobs
pd.options.mode.chained_assignment = None
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns

import preproceso

sns.set()

f = "datasets/train.csv"
df = pd.read_csv(f)
#df = df.head(1000)
numTopics = 20
minPoints = 2*numTopics
distMedias = []
df, diccionario = preproceso.topicosTrain(df, numTopics)
#print(df)

topics = df["Topicos"]
topicDocs = np.zeros(shape=(df.shape[0], 20))

idx = 0
for fila in topics:
    topicDocs[idx] = np.array(fila)
    idx += 1

for td in topicDocs:
    dists = []

    for k in range(0, minPoints, 1):
        aux = topicDocs[k]
        dst = np.linalg.norm(aux - td)
        dists.append(dst)

    distMedia = sum(dists) / len(dists)
    distMedias.append(distMedia)

y = sorted(distMedias)
print(y)
# plotting
plt.title("Line graph")
plt.xlabel("nPoints closer than k-distance")
plt.ylabel("k-distance")
plt.plot(y, color ="green")
plt.show()



