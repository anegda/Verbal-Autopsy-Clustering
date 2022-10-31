import pandas as pd
import evaluacion
pd.options.mode.chained_assignment = None
import DBSCAN
import numpy as np
import preproceso

f="datasets/train.csv"
df = pd.read_csv(f)
df, diccionario = preproceso.topicosTrain(df, 12)

dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(0.125, 13, df)
clusters = sorted(clusters, key=lambda x: x[0])
print(clusters)
referencias = evaluacion.etiqueta_significativa(clusters, df["Chapter"])
idx , cluster = list(zip(*clusters))
evaluacion.evaluar(referencias, clusters, df["Chapter"])