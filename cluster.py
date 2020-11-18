import numpy as np


def calculate_distance(point1, point2):
    """
    calculate the distance between two data points
    """
    return np.linalg.norm(point1 - point2)


def calculate_centroids_change(centroids1, centroids2):
    """
    calculate the mean of centroids change
    input: two dictionary
    """
    # save change of each centroid in a list
    changes = []
    for i in range(len(centroids1)):
        change = calculate_distance(centroids1[i], centroids2[i])
        changes.append(change)
    return sum(changes)/len(changes)


class KMeans:
    """
    K Means algorithm
    """
    def __init__(self, k=2):
        # dataset
        self.X = None
        # number of clusters
        self.k = k
        # centroids
        self.centroids = {i: [] for i in range(self.k)}
        # clusters, store data points index belongs to each cluster
        self.clusters = {i: [] for i in range(self.k)}
        # save the historical centroids  TODO: use queue instead of list
        self.centroids_history = []

    def initiate_centroids(self, X):
        """
        randomly initiate centroids of each cluster
        """
        centroids_idx = list(np.random.choice(np.arange(0, len(X)), self.k, replace=False))
        for i in range(self.k):
            self.centroids[i] = X[centroids_idx[i]]
        self.centroids_history.append(self.centroids)

    def fit(self, X, max_iter=10000, min_centroids_change=0.001):
        self.X = X
        # randomly initiate cluster centroids
        self.initiate_centroids(self.X)
        iteration = 0
        while iteration < max_iter:
            # assign each data point to cluster
            self.assign_points(self.X)
            # recalculate the centroid of each cluster
            self.calculate_centroids(self.X)
            # justify if meet stopping criteria
            # calculate the centroids distance change
            centroids_change = calculate_centroids_change(self.centroids_history[-1], self.centroids_history[-2])
            if centroids_change < min_centroids_change:
                break
            iteration += 1

    def assign_points(self, indices):
        """
        assign data points to closet cluster
        """
        self.clusters = {i: [] for i in range(self.k)}
        labels = []
        for i in range(len(indices)):
            min_distance = float('inf')
            min_idx = 0
            for cluster_idx in range(self.k):
                distance = calculate_distance(self.X[i], self.centroids[cluster_idx])
                if distance < min_distance:
                    min_distance = distance
                    min_idx = cluster_idx
            labels.append(min_idx)
            self.clusters[min_idx].append(i)
        return labels

    def calculate_centroids(self, X):
        for i in self.clusters:
            summation = np.zeros(len(X[0]))
            for data_idx in self.clusters[i]:
                summation = np.sum([summation, X[data_idx]], axis=0)
            centroid = summation / len(self.clusters[i])
            self.centroids[i] = centroid
        self.centroids_history.append(self.centroids)


def icd(X, centroids):
    """
    calculate intra-cluster distance for one cluster
    """
    distances = []
    for i in range(len(X)):
        distance = calculate_distance(X[i], centroids)
        distances.append(distance)
    return sum(distances)/len(distances)


class BisectKMeans:
    """
    bisect k-means algorithm, will call the KMeans class and Node class for implementing binary tree
    """
    def __init__(self, X, max_n_clusters=8, cluster_threshold=10, intra_cluster_distance=0.1):
        self.X = X
        self.n_clusters = 1
        self.max_n_clusters = max_n_clusters
        self.cluster_threshold = cluster_threshold
        self.intra_cluster_distance = intra_cluster_distance
        self.centroids = []
        self.clusters = []
        self.clusters_label = {}
        self.root = None

    def fit(self):
        cluster = KMeans(k=1)
        cluster.fit(self.X)
        root = Node(cluster.clusters[0], cluster.centroids[0])
        root.original_indices = np.array(range(len(self.X)))
        self.root = root
        # print(self.root.indices)
        self.bisect(root)
        self.iterate_leaves(root)
        self.dendrogram()

    def iterate_leaves(self, root):
        # print('root: ', root.indices)
        if not root:
            return
        if not root.left and not root.right:
            self.centroids.append(root.centroid)
            self.clusters.append(root.original_indices)
            return
        # if root.left:
            # print('left: ', root.left.indices)
        # if root.right:
            # print('right: ', root.right.indices)
        if root.left:
            self.iterate_leaves(root.left)
        if root.right:
            self.iterate_leaves(root.right)

    def bisect(self, root):
        if not root:
            return
        # check if meet stopping criteria
        if self.stop_criteria(root.indices, root.centroid):
            return
        cluster = KMeans()
        cluster.fit(self.X[root.indices])
        # ensure the smaller cluster to be on the left
        if len(cluster.clusters[0]) < len(cluster.clusters[0]):
            left = 0
            right = 1
        else:
            left = 1
            right = 0
        self.n_clusters += 1
        root.right = Node(cluster.clusters[right], cluster.centroids[right])
        root.right.original_indices = root.original_indices[root.right.indices]
        # print(root.right.indices)
        root.left = Node(cluster.clusters[left], cluster.centroids[left])
        root.left.original_indices = root.original_indices[root.left.indices]
        # print(root.left.indices)
        largest_leaf = find_largest_leaf(self.root)
        self.bisect(largest_leaf)

    def stop_criteria(self, indices, centroids):
        """
        there are three criteria. stop bisecting if any of them satisfied
        """
        # smaller than specified cluster size
        if len(indices) < self.cluster_threshold:
            return True
        # total number of clusters
        if self.n_clusters >= self.max_n_clusters:
            return True
        # smaller than specified intra-cluster distance
        if icd(self.X[indices], centroids) < self.intra_cluster_distance:
            return True
        return False

    def dendrogram(self):
        level_order_traverse(self.root)


def find_largest_leaf(root):
    """find the leaf node with the largest cluster size"""
    # Base case
    if not root:
        return
    if not root.left and not root.right:
        return root

    left_largest = find_largest_leaf(root.left)
    right_largest = find_largest_leaf(root.right)
    if len(left_largest.indices) > len(right_largest.indices):
        return left_largest
    else:
        return right_largest


def height(node):
    """
    compute the height of a binary tree
    """
    if not node:
        return 0
    else:
        # Compute the height of each subtree
        left_height = height(node.left)
        right_height = height(node.right)

        # Use the larger one
        if left_height > right_height:
            return left_height + 1
        else:
            return right_height + 1


def iterate_given_level(root, level):
    """iterate nodes at a given level"""
    # print('\n')
    if root is None:
        return
    if level == 1:
        print(len(root.indices), end=" ")
    elif level > 1 :
        iterate_given_level(root.left, level-1)
        iterate_given_level(root.right, level-1)


def level_order_traverse(root):
    """level order traversal of a binary tree"""
    h = height(root)
    for i in range(1, h+1):
        print()
        iterate_given_level(root, i)


class Node:
    """
    binary tree node
    """
    def __init__(self, indices, centroid):
        self.indices = indices
        self.centroid = centroid
        self.left = None
        self.right = None
        self.original_indices = None



