# intclustval

A lightweight Python package for internal clustering validation metrics.

`intclustval` provides a simple `InternalClusterScore` class for evaluating clustering quality using internal validation metrics.

Internal clustering validation metrics use only the input data and predicted cluster labels. They do not require ground-truth labels.

## Related packages

This package is part of a small clustering-validation ecosystem:

| Package | Purpose |
|---|---|
| [`intclustval`](https://github.com/semoglou/intclustval) | Internal clustering validation metrics |
| [`extclustval`](https://github.com/semoglou/extclustval) | External clustering validation metrics using ground-truth labels |
| [`sil-score`](https://pypi.org/project/sil-score/) | Exact and approximate silhouette scoring |

Silhouette scores are intentionally not included in `intclustval`, because they are provided by the separate [`sil-score`](https://pypi.org/project/sil-score/) package.

This keeps `intclustval` focused on other internal validation metrics such as Calinski-Harabasz, Davies-Bouldin, inertia, Dunn Index, and Xie-Beni.

## Metrics included

### Internal clustering validation metrics

| Attribute | Metric | Better direction |
|---|---|---|
| `calinski_harabasz` | Calinski-Harabasz score | Higher is better |
| `davies_bouldin` | Davies-Bouldin score | Lower is better |
| `inertia` | Within-cluster sum of squared distances | Lower is better for fixed number of clusters |
| `dunn_index` | Dunn Index | Higher is better |
| `xie_beni` | Xie-Beni index | Lower is better |

### Aliases

| Attribute | Alias for |
|---|---|
| `ch` | `calinski_harabasz` |
| `db` | `davies_bouldin` |
| `within_cluster_dispersion` | `inertia` |

### Metadata

| Attribute | Description |
|---|---|
| `n_samples` | Number of samples |
| `n_features` | Number of features |
| `n_clusters` | Number of clusters |
| `labels_unique` | Unique cluster labels |
| `cluster_sizes` | Number of samples in each cluster |
| `centroids` | Cluster centroids |

## Installation

You can install `intclustval` from [PyPI](https://pypi.org/project/intclustval/):

```python
pip install intclustval
```

## Quick start

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from intclustval import InternalClusterScore

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42,
)

labels = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
).fit_predict(X)

score = InternalClusterScore(X, labels)

print(score.calinski_harabasz)
print(score.davies_bouldin)
print(score.inertia)
print(score.dunn_index)
print(score.xie_beni)
```

Example output:

```text
5196.295097418395
0.21231599538998425
566.8595511244131
0.9484430301054112
0.018180444255623783
```

You can also access all aggregate scores as a dictionary:

```python
scores = score.to_dict()
print(scores)
```

Example:

```python
{
    "calinski_harabasz": 5196.295097418395,
    "ch": 5196.295097418395,
    "davies_bouldin": 0.21231599538998425,
    "db": 0.21231599538998425,
    "inertia": 566.8595511244131,
    "within_cluster_dispersion": 566.8595511244131,
    "dunn_index": 0.9484430301054112,
    "xie_beni": 0.018180444255623783,
}
```

## Using silhouette scores

Silhouette scores are available in the separate [`sil-score`](https://pypi.org/project/sil-score/) package.

Install it with:

```bash
pip install sil-score
```

Then use:

```python
from sil_score import (
    sil_samples,
    micro_sil_score,
    macro_sil_score,
)

sample_scores = sil_samples(X, labels)
micro_score = micro_sil_score(X, labels)
macro_score = macro_sil_score(X, labels)

print(micro_score)
print(macro_score)
```

The [`sil-score`](https://github.com/semoglou/sil-score) package also supports approximate silhouette scoring through its `approximation` argument.

## Metric definitions

### Calinski-Harabasz score

The Calinski-Harabasz score measures the ratio of between-cluster dispersion to within-cluster dispersion.

A higher value usually indicates better-defined clusters. It is useful for comparing different clustering solutions on the same dataset.

### Davies-Bouldin score

The Davies-Bouldin score measures average similarity between each cluster and its most similar other cluster.

A lower value indicates better clustering, because it means clusters are more compact and more separated from each other.

### Inertia

Inertia is the within-cluster sum of squared distances from each sample to its assigned cluster centroid.

Lower inertia means samples are closer to their cluster centers. However, inertia always decreases as the number of clusters increases, so it should mainly be used to compare solutions with different values of `k` on the same dataset.

### Dunn Index

The Dunn Index compares the minimum distance between different clusters to the maximum diameter within any cluster.

A higher Dunn Index indicates better clustering, with clusters that are compact and well separated.

This implementation uses pairwise distances, so it may be slower for large datasets.

### Xie-Beni index

The Xie-Beni index compares total within-cluster compactness to the minimum squared distance between cluster centroids.

A lower value indicates better clustering, because it means compact clusters with well-separated centers.

## Notes

Internal clustering validation metrics do not use ground-truth labels. They evaluate clustering structure using only:

```python
X
labels
```

For external clustering validation with ground-truth labels, use [`extclustval`](https://github.com/semoglou/extclustval).

For silhouette-specific scoring, use [`sil-score`](https://pypi.org/project/sil-score/).

## Cached properties

`InternalClusterScore` uses cached properties.

This means each metric is computed once and then stored.

```python
score = InternalClusterScore(X, labels)

score.inertia  # computed once
score.inertia  # reused from cache
```

If you want to evaluate different labels, create a new `InternalClusterScore` object:

```python
score = InternalClusterScore(X, labels)

new_score = InternalClusterScore(X, new_labels)
```

Do not modify `score.X` or `score.labels` after creating the object.

## Requirements

```text
numpy
scipy
scikit-learn
```

## License

This project is licensed under the [MIT](https://github.com/semoglou/intclustval/blob/main/LICENSE) License.
