import pandas as pd
import evaluacion
pd.options.mode.chained_assignment = None
import DBSCAN
import numpy as np
import preproceso

f="datasets/train.csv"
df = pd.read_csv(f)
df, diccionario = preproceso.topicosTrain(df, 20)
df.to_csv('Resultados/ResultadosPreproceso.csv')
df = df.head(500)
"""
dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(0.01, 40, df)
clusters = sorted(clusters, key=lambda x: x[0])
print(clusters)
referencias = evaluacion.etiqueta_significativa(clusters, df["Chapter"])
idx , cluster = list(zip(*clusters))

evaluacion.evaluar(referencias, clusters, df["Chapter"])

resultados = pd.DataFrame()
newid = []
chapters = []
for i in idx:
    newid.append(df.iloc[i]["newid"])
    chapters.append(df.iloc[i]["Chapter"])
resultados["Indice"] = idx
resultados["newid"] =  np.array(newid)
resultados["Cluster"] = cluster
resultados["Chapter"] = np.array(chapters)
resultados.to_csv('Resultados/ResultadosTrain.csv')
cluster_df = pd.DataFrame(clusters, columns = ["idx", "cluster"])
"""
"""
#INSERTAR DICCIONARIO
fTest = "datasets/test.csv"
dfTest = pd.read_csv(fTest)
dfTest = preproceso.topicosTest(dfTest, diccionario)

indicesTest = []
clustersTest = []
newidTest = []
for i in range(len(dfTest)):
    cluster = dbscan.predict(dfTest.iloc[i])
    indicesTest.append(i)
    newidTest.append(dfTest.iloc[i]["newid"])
    clustersTest.append(cluster)

resultadosTest = pd.DataFrame()
resultadosTest["Indice"] = np.array(indicesTest)
resultadosTest["newid"] = np.array(newidTest)
resultadosTest["Cluster"] = np.array(clustersTest)
resultadosTest.to_csv('Resultados/ResultadosTest.csv')
"""