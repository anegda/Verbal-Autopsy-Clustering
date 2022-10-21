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

    skplt.plot_confusion_matrix(y_train, labels)
    plt.show()