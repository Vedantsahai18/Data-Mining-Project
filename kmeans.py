import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def parse(csv_file, x_axis_col, y_axis_col, csv_folder="csv"):
    dataset = pd.read_csv(csv_folder + "//" + csv_file)
    X = dataset.iloc[:, [x_axis_col, y_axis_col]].values
    return X


def elbow(X, cluster_limit=11):
    wcss = []
    for i in range(1, cluster_limit):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, cluster_limit), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join('static', 'elbow.png'))
    plt.close()


def kmeans(X, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    y_kmeans = kmeans.fit_predict(X)

    c = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=c[i], label='Cluster ' + str(i))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters')
    plt.xlabel('X AXIS')
    plt.ylabel('Y AXIS')
    plt.legend()
    plt.savefig(os.path.join('static','kmeans_clusters.png'))
    plt.close()