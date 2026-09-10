from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


"""
    Implementación de un algoritmo de clustering DBSCAN en Python para entrenar un modelo de clasificación no supervisada.
    
    Implementado a partir de la solucion de Kaggle:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
 
"""

FEATURES = ['battery_power', 'ram']
TARGET = 'price_range'
NEIGHBORS = 5
EPS = 0.12
MIN_SAMPLES = 5


def load_dataset(dataset_path):
    # Carga el conjunto de datos desde un archivo CSV y devuelve las características y el objetivo.
    dataset = pd.read_csv(dataset_path)
    features = dataset[FEATURES].to_numpy()
    target = dataset[TARGET].to_numpy()
    return features, target


def normalize_features(features):
    # Normaliza las características para que tengan media 0 y desviación estándar 1.
    return (features - features.mean(axis=0)) / features.std(axis=0)


def calculate_k_distances(features, k):
    # Calcula las distancias al k-ésimo vecino más cercano para cada punto de datos.
    neighbors = NearestNeighbors(n_neighbors=k)
    distances, _ = neighbors.fit(features).kneighbors(features)
    return np.sort(distances[:, k - 1])


def plot_k_distances(k_distances):
    # Grafica la curva de distancias para ayudar a determinar el valor de eps.
    plt.plot(k_distances)
    plt.title('Curva de distancias para determinar eps')
    plt.xlabel('Puntos de datos ordenados por distancia al k-ésimo vecino')
    plt.ylabel('Distancia al k-ésimo vecino más cercano')
    plt.show()


def train_dbscan(features, eps, min_samples):
    ## Entrena el modelo DBSCAN y devuelve los clusters asignados a cada punto.
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(features)


def calculate_silhouette_score(features, clusters):
    ## Calcula el Silhouette Score para los clusters encontrados.
    non_noise = clusters != -1
    non_noise_clusters = np.unique(clusters[non_noise])

    if len(non_noise_clusters) < 2:
        return -1.0

    return silhouette_score(features[non_noise], clusters[non_noise])


def print_results(features, target, clusters):
    cluster_count = len(set(clusters)) - (1 if -1 in clusters else 0)
    noise_count = np.count_nonzero(clusters == -1)
    silhouette = calculate_silhouette_score(features, clusters)
    ari = adjusted_rand_score(target, clusters)

    print(f'Clusters encontrados: {cluster_count}, Puntos de ruido: {noise_count}')
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Adjusted Rand Index (ARI): {ari:.4f}')
    print(pd.crosstab(clusters, target))

    examples = pd.DataFrame(features[:10], columns=FEATURES)
    examples['Cluster asignado'] = clusters[:10]
    examples['price_range real'] = target[:10]
    print(examples)


def plot_clusters(features, clusters):
    plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Battery Power (normalizado)')
    plt.ylabel('RAM (normalizado)')
    plt.title('Clusters encontrados por DBSCAN')
    plt.colorbar(label='Cluster')
    plt.show()


def main():
    pd.set_option('display.max_columns', None)
    dataset_path = Path(__file__).resolve().parent.parent / 'train.csv'

    features, target = load_dataset(dataset_path)
    normalized_features = normalize_features(features)

    k_distances = calculate_k_distances(normalized_features, NEIGHBORS)
    plot_k_distances(k_distances)

    clusters = train_dbscan(normalized_features, EPS, MIN_SAMPLES)
    print_results(features, target, clusters)
    plot_clusters(normalized_features, clusters)


if __name__ == '__main__':
    main()