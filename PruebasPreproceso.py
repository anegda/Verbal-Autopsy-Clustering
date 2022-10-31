import pandas as pd
import evaluacion
pd.options.mode.chained_assignment = None
import DBSCAN
import numpy as np
import preproceso

f="datasets/train.csv"
df = pd.read_csv(f)
df, diccionario = preproceso.topicosTrain(df, 12)
nmprArray = [13,18,24,29,34,39,44,50]
epsilon = np.arange(0.08, 0.20, 0.01)
for i in nmprArray:
    for j in epsilon:
        print("nmpr: ",str(i),"epsilon: ", str(j))
        dbscan = DBSCAN.DBScan()
        clusters = dbscan.fit(j, i, df)
        clusters = sorted(clusters, key=lambda x: x[0])
        print(clusters)
        referencias = evaluacion.etiqueta_significativa(clusters, df["Chapter"])
        idx , cluster = list(zip(*clusters))
        evaluacion.evaluar(referencias, clusters, df["Chapter"])