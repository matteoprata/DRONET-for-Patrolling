import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# Example with randomly generated 3D data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X = np.column_stack((X, np.random.rand(300)))  # Adding a third dimension

# Range of k values to test
k_values = range(1, 11)

# Plot the data points in 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Perform k-means clustering for k=4 (for example)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Assign different colors to clusters
colors = ['r', 'g', 'b', 'y']
for i in range(4):
    cluster_points = X[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i}', c=colors[i])

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Data Points')
ax.legend()

# Plot the elbow curve
ax = fig.add_subplot(122)
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    inertia_values.append(kmeans.inertia_)

ax.plot(k_values, inertia_values, marker='o')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal k')

plt.show()
