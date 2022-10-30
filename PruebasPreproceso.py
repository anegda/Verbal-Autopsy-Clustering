import pandas as pd
pd.options.mode.chained_assignment = None
import preproceso

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