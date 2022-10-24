import pandas as pd
import preproceso

fTrain = "datasets/train.csv"
dfTrain = pd.read_csv(fTrain)
fTest = "datasets/train.csv"
dfTest = pd.read_csv(fTest)

print("--- TAMAÑO DE LOS CONJUNTOS TRAIN y TEST ---")
print("Train:" , len(dfTrain.df.axes[0]))
print("Test:" , len(dfTest.df.axes[0]))

print("--- ATRIBUTOS DEL DATASET ---")
print(dfTrain.head())
print(dfTest.head())

print("--- AGRUPACIÓN POR ENFERMEDAD DEL DATASET ---")
print(dfTrain.groupby('gs_text34').size())

dfTrain["Chapter"]=dfTrain["gs_text34"].apply(preproceso.diseaseToChapter)
print(dfTrain.groupby('Chapter').size())



