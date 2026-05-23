import numpy as np

from functools import cached_property
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import pairwise_distances


class InternalClusterScore:
    """
    Internal validation scores for clustering.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    labels : array-like of shape (n_samples,)
        Predicted cluster labels.

    Notes
    -----
    InternalClusterScore stores copies of X and labels at initialization.

    It uses cached properties, so each score is computed once and then reused.
    If you want to evaluate different labels, create a new InternalClusterScore
    object.

    Silhouette scores are intentionally not included here because they are
    provided by the separate sil-score PyPI package:
    https://pypi.org/project/sil-score/
    """

    def __init__(self, X, labels):
        self.X = np.asarray(X).copy()
        self.labels = np.asarray(labels).copy()

        self._validate_inputs()

    def _validate_inputs(self):
        if self.X.ndim != 2:
            raise ValueError("X must be two-dimensional.")

        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional.")

        if len(self.X) != len(self.labels):
            raise ValueError("X and labels must have the same number of samples.")

        if len(self.X) == 0:
            raise ValueError("X and labels must not be empty.")

        if len(np.unique(self.labels)) < 2:
            raise ValueError("At least two clusters are required.")

        if len(np.unique(self.labels)) >= len(self.labels):
            raise ValueError("Number of clusters must be smaller than number of samples.")

    @cached_property
    def n_samples(self):
        """Number of samples."""
        return int(self.X.shape[0])

    @cached_property
    def n_features(self):
        """Number of features."""
        return int(self.X.shape[1])

    @cached_property
    def labels_unique(self):
        """Unique cluster labels."""
        return np.unique(self.labels)

    @cached_property
    def n_clusters(self):
        """Number of clusters."""
        return int(len(self.labels_unique))

    @cached_property
    def cluster_sizes(self):
        """Number of samples in each cluster."""
        labels, counts = np.unique(self.labels, return_counts=True)

        return {
            label.item() if hasattr(label, "item") else label: int(count)
            for label, count in zip(labels, counts)
        }

    @cached_property
    def centroids(self):
        """
        Cluster centroids.

        Rows correspond to clusters in labels_unique order.
        """
        centroids = []

        for label in self.labels_unique:
            centroids.append(np.mean(self.X[self.labels == label], axis=0))

        return np.asarray(centroids)

    @cached_property
    def calinski_harabasz(self):
        """Calinski-Harabasz score. Higher is better."""
        return float(calinski_harabasz_score(self.X, self.labels))

    @cached_property
    def ch(self):
        """Alias for calinski_harabasz."""
        return self.calinski_harabasz

    @cached_property
    def davies_bouldin(self):
        """Davies-Bouldin score. Lower is better."""
        return float(davies_bouldin_score(self.X, self.labels))

    @cached_property
    def db(self):
        """Alias for davies_bouldin."""
        return self.davies_bouldin

    @cached_property
    def inertia(self):
        """
        Within-cluster sum of squared distances to cluster centroids.

        Lower is better, but inertia always decreases as the number of clusters
        increases.
        """
        total = 0.0

        for label, centroid in zip(self.labels_unique, self.centroids):
            points = self.X[self.labels == label]
            total += np.sum((points - centroid) ** 2)

        return float(total)

    @cached_property
    def within_cluster_dispersion(self):
        """
        Alias for inertia.

        This is the within-cluster sum of squared distances.
        """
        return self.inertia

    @cached_property
    def dunn_index(self):
        """
        Dunn Index.

        Dunn Index = minimum inter-cluster distance / maximum intra-cluster
        diameter.

        Higher is better.

        Notes
        -----
        This implementation uses pairwise distances and may be slow for large
        datasets.
        """
        distances = pairwise_distances(self.X)

        max_intra_distance = 0.0

        for label in self.labels_unique:
            indices = np.where(self.labels == label)[0]

            if len(indices) <= 1:
                intra_distance = 0.0
            else:
                cluster_distances = distances[np.ix_(indices, indices)]
                intra_distance = np.max(cluster_distances)

            max_intra_distance = max(max_intra_distance, intra_distance)

        min_inter_distance = np.inf

        for i, label_i in enumerate(self.labels_unique):
            indices_i = np.where(self.labels == label_i)[0]

            for label_j in self.labels_unique[i + 1:]:
                indices_j = np.where(self.labels == label_j)[0]

                inter_distances = distances[np.ix_(indices_i, indices_j)]
                inter_distance = np.min(inter_distances)

                min_inter_distance = min(min_inter_distance, inter_distance)

        if max_intra_distance == 0:
            return float(np.inf)

        return float(min_inter_distance / max_intra_distance)

    @cached_property
    def xie_beni(self):
        """
        Xie-Beni index.

        Lower is better.

        Formula
        -------
        Xie-Beni = inertia / (n_samples * minimum squared centroid distance)
        """
        centroid_distances = pairwise_distances(self.centroids) ** 2

        np.fill_diagonal(centroid_distances, np.inf)

        min_centroid_distance = np.min(centroid_distances)

        if min_centroid_distance == 0:
            return float(np.inf)

        return float(self.inertia / (self.n_samples * min_centroid_distance))

    @cached_property
    def scores(self):
        """
        Return all aggregate internal validation scores as a dictionary.
        """
        return {
            "calinski_harabasz": self.calinski_harabasz,
            "ch": self.ch,
            "davies_bouldin": self.davies_bouldin,
            "db": self.db,
            "inertia": self.inertia,
            "within_cluster_dispersion": self.within_cluster_dispersion,
            "dunn_index": self.dunn_index,
            "xie_beni": self.xie_beni,
        }

    def to_dict(self):
        """Return all aggregate internal validation scores as a dictionary."""
        return dict(self.scores)
