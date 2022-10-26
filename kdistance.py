import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
import seaborn as sns

import preproceso
import DBSCAN

sns.set()

f = "datasets/train.csv"
df = pd.read_csv(f)
df = df.head(500)
numTopics = 20
minPoints = (numTopics+1)
distMedias = []
df, diccionario = preproceso.topicosTrain(df, numTopics)
#print(df)

topics = df["Topicos"]
topicDocs = np.zeros(shape=(df.shape[0], 20))

for i in range(len(df)):
    topicDocs[i] = np.array(topics.iloc[i])

for i in range(len(topicDocs)):
    dists = []
    td = topicDocs[i]
    for j in range(len(topicDocs)):
        if i!=j:
            aux = topicDocs[j]
            dst = np.linalg.norm(aux - td)
            dists.append(dst)

    dists = sorted(dists)
    dists = dists[:minPoints]
    print(len(dists))
    distMedia = sum(dists) / len(dists)
    distMedias.append(distMedia)

y = sorted(distMedias)

# plotting
plt.title("Line graph")
plt.xlabel("nPoints closer than k-distance")
plt.ylabel("k-distance")
plt.plot(y, color ="green")
plt.savefig('Imagenes/kdistance1000.png')



