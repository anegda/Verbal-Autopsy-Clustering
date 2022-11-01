import pandas as pd
import evaluacion
pd.options.mode.chained_assignment = None
import DBSCAN
from matplotlib import pyplot as plt
import numpy as np
import preproceso

f="datasets/train.csv"
df = pd.read_csv(f)
df, diccionario = preproceso.topicosTrain(df, 26)
df.to_csv("Resultados/ResultadosPreproceso.csv")

dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(0.1, 70, df)
clusters = sorted(clusters, key=lambda x: x[0])
print(clusters)
referencias = evaluacion.etiqueta_significativa(clusters, df["Chapter"])
idx , cluster = list(zip(*clusters))
evaluacion.evaluar(referencias, clusters, df["Chapter"])

"""
distMedias = []
topics = df["Topicos"]
topicDocs = np.zeros(shape=(df.shape[0], 26))

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
    dists = dists[:(52)]
    distMedia = sum(dists) / len(dists)
    distMedias.append(distMedia)

y = sorted(distMedias)

# plotting
plt.title("Line graph")
plt.xlabel("nPoints closer than k-distance")
plt.ylabel("k-distance")
plt.plot(y, color ="green")
plt.savefig('Imagenes/kdistance.png')
"""