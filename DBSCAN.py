import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import random

class DBScan:
    def __init__(self):
        # inicializacion de las variables locales
        self.clusters = []  #clusers(instanciaId, clusterId)
        self.epsilon = 0
        self.nmpr = 0
        self.df = []

    def fit(self, epsilon, nmpr, df):
        """
        input:  epsilon --> radio
                nmpr --> minimo numero de instancias por cluster
                df --> dataframe que almacena instancias de train
        return: la lista de clusters con el id cada instancia que contiene por cada cluster   
        """

        self.epsilon = epsilon
        self.nmpr = nmpr
        self.df = df
        clusterId = 1
        current_stack = set()
        unvisited = list(df.index)
        primer_pto = True

        while len(unvisited) != 0:  # recorrer todos los puntos no visitados

            # añadir un punto aleatorio no visitado para explorarlo
            current_stack.add(random.choice(unvisited))

            # ciclo hasta completar el cluster
            while len(current_stack) > 0:

                curr_idx = current_stack.pop()  # instancia a examinar y expandir

                # obtener los vecinos del punto actual y el tipo de punto actual (core, edge, outlier)
                neigh_indexes, tipo = obtenerVecinos(epsilon, nmpr, df, curr_idx)

                if tipo == 1 and primer_pto:  # si el primer punto es un border point
                    # marcar el punto actual y a sus vecinos como outliers
                    self.clusters.append((curr_idx, 0))
                    self.clusters.extend(list(zip(neigh_indexes, [0 for _ in range(len(neigh_indexes))])))
                    # marcar el punto actual y a sus vecinos como visitados
                    unvisited.remove(curr_idx)
                    unvisited = [e for e in unvisited if e not in neigh_indexes]
                    continue

                unvisited.remove(curr_idx)  # marcar el punto actual como visitado
                neigh_indexes = set(neigh_indexes) & set(unvisited)  # look at only unvisited points   duda

                if tipo == 0:  # si es core
                    # marcar que se ha obtenido el primer punto válido del cluster
                    # a partir de ahora se podrán evaluar los border points como puntos válidos
                    primer_pto = False
                    self.clusters.append((curr_idx, clusterId))  # añadir al cluster actual
                    current_stack.update(neigh_indexes)  # añadir los vecinos a la pila para posteriormente expandirlos

                elif tipo == 1:  # si es core
                    self.clusters.append((curr_idx, clusterId))  # añadir al cluster actual
                    continue

                elif tipo == 2:  # si es outlier
                    self.clusters.append((curr_idx, 0))  # añadir al cluster de los outilers (id=0)
                    continue

            if not primer_pto:  # si al menos hay un cluster creado
                # cuando se haya completado el cluster y no queden puntos al alcance, se definirá otro cluster
                clusterId += 1

        return self.clusters

    def predict(self, x):
        """
        input: x --> nueva instancia a clasificar
        return: el id del cluster al que pertenece x       
        """

        # Hacerle lo que haga falta

        vecinos = []
        nInstanciasCl = {}  # dict(clusterId,numInstancias)
        cl = dict(self.clusters)  # copiar el puntero

        for b in self.df:  # obtener la lista de vecinos de x
            dist = np.linalg.norm(x - b)
            if dist <= self.epsilon:
                vecinos.append(b)

        for vecino in vecinos:  # recuento de vecinos en clusters
            if cl[vecino] not in nInstanciasCl:
                nInstanciasCl[cl[vecino]] = 1
            else:
                nInstanciasCl[cl[vecino]] += 1

        recuento = nInstanciasCl.values()
        maximo = max(recuento)
        keys = nInstanciasCl.keys()

        return keys(recuento.index(maximo))  # keys(recuento(maximo).index)   duda


def obtenerVecinos(epsilon, nmpr, df, index):
    a = df.iloc[index]
    a = a.to_numpy()
    vecinos = pd.DataFrame()
    for i in range(len(df)):
        b = df.iloc[i]
        b = b.to_numpy()
        dist = np.linalg.norm(a - b)
        if dist <= epsilon:
            vecinos = vecinos.append(df.iloc[i])

    if len(vecinos) >= nmpr:  # core
        return vecinos.index, 0

    elif (len(vecinos) < nmpr) and len(vecinos) > 0:  # edge
        return vecinos.index, 1

    elif len(vecinos) == 0:  # outlier
        return vecinos.index, 2
