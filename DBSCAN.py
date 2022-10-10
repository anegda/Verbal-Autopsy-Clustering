import pandas as pd
import numpy as np
import random
class DBScan:
    def __init__(self):
        self.clusters= []
        self.epsilon = 0
        self.nmpr = 0
        self.df = []

    def fit(self, epsilon, nmpr, df):
        self.epsilon=epsilon
        self.nmpr=nmpr
        self.df = df
        C = 1
        current_stack = set()
        unvisited = list(df.index)

        while (len(unvisited)!=0):
            primer_pto = True
            current_stack.add(random.choice(unvisited))

            while len(current_stack) != 0:
                curr_idx = current_stack.pop()
                vecinos_idx, tipo = obtenerVecinos(epsilon,nmpr,df, curr_idx)

                if(tipo==1 & primer_pto):    #si es un border point
                    self.clusters.append((curr_idx, 0))
                    self.clusters.extend(list(zip(vecinos_idx, [0 for _ in range(len(vecinos_idx))])))

                    # label as visited
                    unvisited.remove(curr_idx)
                    unvisited = [e for e in unvisited if e not in vecinos_idx]

                    continue
                unvisited.remove(curr_idx)  # remove point from unvisited list

                neigh_indexes = set(neigh_indexes) & set(unvisited)  # look at only unvisited points

                if(tipo==0):    #Si es core
                    first_point = False

                    self.clusters.append((curr_idx, C))  # assign to a cluster
                    current_stack.update(neigh_indexes)  # add neighbours to a stack

                elif(tipo==1):
                    self.clusters((curr_idx, C))
                    continue

                elif(tipo==2):
                    self.clusters((curr_idx,0))
                    continue

            if not first_point:
                C+=1
        return self.clusters

    def predict(self, x):
        #Hacerle lo que haga falta
        vecinos = []
        for b in self.df:
            dist = np.linalg.norm(x - b)
            if (dist <= self.epsilon):
                vecinos.append(b)
        count = {}
        cl = dict(self.clusters)
        for vecino in vecinos:
            if cl[vecino] not in count:
                count[cl[vecino]] = 1
            else:
                count[cl[vecino]]+=1

        valores = count.values()
        keys = count.keys()
        maximo = max(valores)
        return keys(valores.index(maximo))



def obtenerVecinos(epsilon, nmpr, df, index):
    a = df.iloc[index]
    a = a.to_numpy()
    vecinos = []
    for b in df:
        dist = np.linalg.norm(a-b)
        if (dist <=epsilon):
            vecinos.append(b)

    if len(vecinos) >= nmpr:
        return (vecinos.index, 0)

    elif (len(vecinos) < nmpr) and len(vecinos) > 0:
        return (vecinos.index, 1)

    elif len(vecinos) == 0:
        return (vecinos.index, 2)
