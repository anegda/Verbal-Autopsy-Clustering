import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import random

class DBScan:
    def __init__(self):
        # inicializacion de las variables locales
        self.clusters = []  #clusers(instanciaId, clusterId)
        self.tipos = [] #(instanciaId, tipoInstancia) => 0 core, 1 border, 2 outlier
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

        while len(unvisited) != 0:  # recorrer todos los puntos no visitados
            # marcamos el primer punto de cada cluster para el caso crítico con los border points
            primer_pto = True

            # añadir un punto aleatorio no visitado para explorarlo
            current_stack.add(random.choice(unvisited))

            # ciclo hasta completar el cluster
            while len(current_stack) > 0:

                curr_idx = current_stack.pop()  # instancia a examinar y expandir

                # obtener los vecinos del punto actual y el tipo de punto actual (core, edge, outlier)
                neigh_indexes, tipo = obtenerVecinos(epsilon, nmpr, df, curr_idx)
                neigh_indexes = set(neigh_indexes) & set(unvisited)  # look at only unvisited points   duda

                if tipo == 1 and primer_pto:  # si el primer punto es un border point
                    # marcar el punto actual y a sus vecinos como outliers
                    self.clusters.append((curr_idx, 0))
                    self.tipos.append((curr_idx, 1))
                    self.clusters.extend(list(zip(neigh_indexes, [0 for _ in range(len(neigh_indexes))])))
                    self.tipos.extend(list(zip(neigh_indexes, [1 for _ in range(len(neigh_indexes))])))
                    # marcar el punto actual y a sus vecinos como visitados
                    unvisited.remove(curr_idx)
                    unvisited = [e for e in unvisited if e not in neigh_indexes]
                    continue

                unvisited.remove(curr_idx)  # marcar el punto actual como visitado

                if tipo == 0:  # si es core
                    # marcar que se ha obtenido el primer punto válido del cluster
                    # a partir de ahora se podrán evaluar los border points como puntos válidos
                    primer_pto = False
                    self.clusters.append((curr_idx, clusterId))  # añadir al cluster actual
                    self.tipos.append((curr_idx, tipo))     # indicamos que este punto es un core point (útil para el predict)
                    current_stack.update(neigh_indexes)  # añadir los vecinos a la pila para posteriormente expandirlos

                elif tipo == 1:  # si es core
                    self.clusters.append((curr_idx, clusterId))  # añadir al cluster actual
                    self.tipos.append((curr_idx, tipo))
                    continue

                elif tipo == 2:  # si es outlier
                    self.clusters.append((curr_idx, 0))  # añadir al cluster de los outilers (id=0)
                    self.tipos.append((curr_idx, tipo))
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
        vecinos = pd.DataFrame()
        nInstanciasCl = {}  # dict(clusterId,numInstancias)
        cl = dict(self.clusters)  # copiar el puntero
        tipos = dict(self.tipos)
        x = x["Topicos"]

        for i in range(len(self.df)):  # obtener la lista de vecinos de x
            b = self.df.iloc[i]
            b = b["Topicos"]

            dist = distanciaEuclidea(x, b)
            if dist <= self.epsilon and tipos[i]==0:
                vecinos = vecinos.append(self.df.iloc[i])
        for vecino in vecinos.index:  # recuento de vecinos en clusters
            if cl[vecino] not in nInstanciasCl:
                nInstanciasCl[cl[vecino]] = 1
            else:
                nInstanciasCl[cl[vecino]] += 1

        if(len(nInstanciasCl.values())==0):
            return 0
        else:
            recuento = [i for i in nInstanciasCl.values()]
            maximo = max(recuento)
            keys = [i for i in nInstanciasCl.keys()]

            return keys[recuento.index(maximo)]  # keys(recuento(maximo).index)   duda


def obtenerVecinos(epsilon, nmpr, df, index):
    a = df.iloc[index]
    a = a["Topicos"]

    vecinos = pd.DataFrame()
    for i in range(len(df)):
        b = df.iloc[i]
        b = b["Topicos"]
        dist = distanciaEuclidea(a, b)
        if dist <= epsilon and i!=index:
            vecinos = vecinos.append(df.iloc[i])
    if len(vecinos) >= nmpr:  # core
        return vecinos.index, 0

    elif (len(vecinos) < nmpr) and len(vecinos) > 0:  # edge
        return vecinos.index, 1

    elif len(vecinos) == 0:  # outlier
        return vecinos.index, 2

def distanciaEuclidea(a, b):
    #a = a.to_numpy()  # PARA EL EJEMPLO
    a = np.array(a)
    #b = b.to_numpy() # PARA EL EJEMPLO
    b = np.array(b)
    return np.linalg.norm(a - b)

def distanciaTopico(a, b):
    # Si el tópico más común coincide se calcula esa distancia.
    # Sino se suma la distancia de cada tópico más común con su correspondiente en la otra instancia
    iA = a.index(max(a))
    iB = b.index(max(b))
    if (iA == iB):
        return abs(a[iA]-b[iB])
    else:
        return abs(a[iA]-b[iA]) + abs(a[iB]-b[iB])