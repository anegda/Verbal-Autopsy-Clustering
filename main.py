import csv
import numpy as np
import pandas as pd
import DBSCAN
from sklearn.datasets import make_blobs

f="datasets/train.csv"
ml_dataset = pd.read_csv(f)

centers = [(0, 4), (5, 5) , (8,2)]
cluster_std = [1.2, 1, 1.1]
X, y= make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
#radius of the circle defined as 0.6
eps = 0.6
#minimum neighbouring points set to 3
minPts = 3

data = pd.DataFrame(X, columns = ["X", "Y"] )

dbscan = DBSCAN.DBScan()
clusters = dbscan.fit(eps, minPts, data)