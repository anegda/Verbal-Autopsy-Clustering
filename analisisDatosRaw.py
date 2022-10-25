import pandas as pd
import preproceso

fTrain = "datasets/train.csv"
dfTrain = pd.read_csv(fTrain)
fTest = "datasets/test.csv"
dfTest = pd.read_csv(fTest)

print("\n--- TAMAÑO DE LOS CONJUNTOS TRAIN y TEST ---")
print("Train:" , len(dfTrain.axes[0]))
print("Test:" , len(dfTest.axes[0]))

print("\n--- EXISTEN INSTANCIAS REPETIDAS? ---")
print(dfTrain.duplicated())
print(dfTest.duplicated())
dfTrain = dfTrain.drop_duplicates()
dfTest = dfTest.drop_duplicates()
print("Train:" , len(dfTrain.axes[0]))
print("Test:" , len(dfTest.axes[0]))

print("\n--- MÁS INFORMACIÓN SOBRE EL DATASET ---")
dfTrain.info(verbose=True,null_counts=True)

print("\n--- ATRIBUTOS DEL DATASET ---")
print(dfTrain.head())
print(dfTest.head())

print("\n--- AGRUPACIÓN POR ENFERMEDAD DEL DATASET ---")
print(dfTrain.groupby('gs_text34').size())

dfTrain["Chapter"]=dfTrain["gs_text34"].apply(preproceso.diseaseToChapter)
print(dfTrain.groupby('Chapter').size())




