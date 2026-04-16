from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ElbowMethod:

    def __init__(self, X):
        self.X = X
        self.inertias = []

    def run(self, k_range=range(1, 11)):

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.X)
            self.inertias.append(kmeans.inertia_)

        self.plot(k_range)

    def plot(self, k_range):

        plt.plot(list(k_range), self.inertias, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class SilhouetteMethod:

    def __init__(self, X):
        self.X = X
        self.scores = []

    def run(self, k_range=range(2, 11)):

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(self.X)

            score = silhouette_score(self.X, labels)
            self.scores.append(score)

            print(f"k={k}, silhouette={score:.4f}")