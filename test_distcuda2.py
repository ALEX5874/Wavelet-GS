import numpy as np
from sklearn.neighbors import NearestNeighbors

# Step 1: Create a random matrix X of size (10000, 3)
n = 200000
X = np.random.rand(n, 3)  # Replace with actual data if needed

# Step 2: Define the number of nearest neighbors (k)
k = 100  # For example, select 5 nearest neighbors

# Step 3: Use NearestNeighbors from scikit-learn to compute distances
knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 because the nearest neighbor is the point itself
knn.fit(X)

# Step 4: Compute the k-nearest neighbors
distances, indices = knn.kneighbors(X)
print(distances.shape)
print(indices.shape)

# Step 5: Initialize an adjacency matrix of size (10000, 10000)
adj_matrix = np.zeros((n, n))

# Step 6: Fill the adjacency matrix with 1s for the k-nearest neighbors
for i in range(n):
    # indices[i] contains the k+1 nearest neighbors (including the point itself)
    # Skip the first neighbor, which is the point itself
    for j in indices[i][1:]:  # Skip the first neighbor (itself)
        adj_matrix[i, j] = 1

# Step 7: Output the adjacency matrix
print("Adjacency Matrix:")
print(adj_matrix)

# python test_distcuda2.py