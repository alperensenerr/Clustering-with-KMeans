from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

veriler = pd.read_csv('musteriler.csv')

data = veriler.iloc[:,2:].values

sonuclar =[]
for i in range(1,11):
    km = KMeans(n_clusters=i, init = 'k-means++',random_state=123)
    km.fit(data)
    sonuclar.append(km.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

#we see from graph that most likely to be the optimum k is 4
