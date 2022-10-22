import numpy as np
# Calculating accuracy score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score
import scikitplot.metrics as skplt

def etiqueta_significativa(clusters, y_train):
    # Initializing
    reference_labels = {}
    dic = dict(clusters)
    clustersTot = [i for i in dic.values()]
    numCluster = max(clustersTot)
    for i in range(numCluster+1):
        capCount = {}
        for id in dic.keys():
            if dic[id] == i:
                if y_train[id] in capCount:
                    capCount[y_train[id]]+=1
                else:
                    capCount[y_train[id]]=1

        recuento = [i for i in capCount.values()]
        maximo = max(recuento)
        capitulos = [i for i in capCount.keys()]
        etiqueta = capitulos[recuento.index(maximo)]
        reference_labels[i] = etiqueta


    return reference_labels

def evaluar(referencias, clusters ,y_train):
    idx, cluster = list(zip(*clusters))
    cluster = np.asarray(cluster)
    labels = []
    for i in range(len(cluster)):
        labels.append(referencias[cluster[i]])

    print("La accuracy es:", accuracy_score(labels, y_train))
    print("La precision es:", precision_score(labels, y_train, average='weighted'))
    print("El f1 score es:", f1_score(labels, y_train, average='weighted'))

    print(list(y_train))
    print(labels)
    error = {"aciertos":0, "errores":0}
    for y, label in zip(list(y_train), labels):
        if y-label == 0:
            error["aciertos"] = error["aciertos"]+1
        else:
            error["errores"] = error["errores"] + 1
    print(error)
    errorTotal = error["errores"]/(error["errores"]+error["aciertos"])
    print("El error es de: " + str(errorTotal))

    skplt.plot_confusion_matrix(labels, y_train)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.show()