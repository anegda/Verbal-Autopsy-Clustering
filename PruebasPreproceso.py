import pandas as pd
pd.options.mode.chained_assignment = None
import preproceso
import numpy as np
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt

"""
f="datasets/train.csv"
df = pd.read_csv(f)
f2 ="Resultados/lda_tuning_results+0.6.csv"
parDF = pd.read_csv(f2)
for i in range(len(parDF)):
    numTopics = parDF.iloc[i]["Topics"]
    alfa = parDF.iloc[i]["Alpha"]
    beta = parDF.iloc[i]["Beta"]
    print(alfa, beta, numTopics)
    preproceso.topicosTrain(df, numTopics, alfa, beta)
"""
f = "datasets/train.csv"
df = pd.read_csv(f)
distMedias = []
df, diccionario = preproceso.topicosTrain(df, 26, 0.2, 0.9)
#print(df)

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
    dists = dists[:(27)]
    distMedia = sum(dists) / len(dists)
    distMedias.append(distMedia)

y = sorted(distMedias)

# plotting
plt.title("Line graph")
plt.xlabel("nPoints closer than k-distance")
plt.ylabel("k-distance")
plt.plot(y, color ="green")
plt.savefig('Imagenes/kdistance.png')