import gensim.corpora
import pandas as pd
import evaluacion
import DBSCAN
import kdistance
import numpy as np
import preproceso
import pickle
pd.options.mode.chained_assignment = None

def entrenarModelo():
    f="datasets/train.csv"
    df = pd.read_csv(f)
    # df = df.head(100)
    print("### Definición de parámetros del modelo LDA:")
    print("Introduzca el número de tópicos (mejor --> 26): ")
    numTopics = int(input())
    df, diccionario = preproceso.topicosTrain(df, numTopics)

    df.to_csv('Resultados/ResultadosPreproceso.csv')
    print("El resultado del preproceso ha sido almacenado en Resultados/ResultadosPreproceso.csv")
    #df = df.head(2000)

    print("### Definición de parámetros del DBSCAN")
    print("Introduzca el valor de épsilon (mejor --> 0.1: ")
    epsilon = float(input())
    print("Introduzca el valor de NMPR (mejor --> 100): ")
    nmpr = int(input())
    dbscan = DBSCAN.DBScan()
    clusters = dbscan.fit(epsilon, nmpr, df)
    file = open("./modelos/dbscan.sav", "wb")
    pickle.dump(dbscan, file)
    file.close()

    clusters = sorted(clusters, key=lambda x: x[0])
    # print(clusters)
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


def clasificarInstancias():
    #INSERTAR DICCIONARIO

    fTest = "datasets/test.csv"
    dfTest = pd.read_csv(fTest)
    # dfTest = dfTest.head(100)
    dicc = gensim.corpora.Dictionary.load("modelos/dicc")
    dfTest = preproceso.topicosTest(dfTest, dicc)

    file = open("./modelos/dbscan.sav", "rb")
    dbscan = pickle.load(file)
    file.close()

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
    print("Los resultados del test han sido almacenados en Resultados/ResultadosTest.csv")

def main():
    print('''BIENVENIDO AL AGRUPADOR DE DOCUMENTOS MÉDICOS
    
        Previamente hay que tener instaladas las siguientes librerías:
            - pandas
            - numpy
            - matplotlib
            - sklearn
            - seaborn
            - scikitplot
            - nltk
    
        Pulse el número según lo que que desee ejecutar:
            (1) Estimar el valor de epsilon dados unos ciertos parámetros:
            (2) Entrenar el modelo 
            (3) Clasificar instancias nuevas del modelo (debe estar entrenado)
            (4) Salir
    
        By Ane García, Urko García, Marcos Merino\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Ha elegido estimar el valor de epsilon dados unos ciertos parámetros")
        kdistance.kdistance()
        main()

    elif int(eleccion) == 2:
        print("Ha elegido entrenar el modelo ")
        entrenarModelo()
        main()

    elif int(eleccion) == 3:
        print("Ha elegido clasificar instancias nuevas del modelo")
        clasificarInstancias()
        main()

    elif int(eleccion) == 4:
        print("SALIENDO...")
        return

    else:
        print("Seleccion incorrecta\n\n")
        main()

if __name__ == "__main__":
    main()