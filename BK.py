import argparse
import os

import data_loader as dat
from cluster import KMeans, BisectKMeans

"""
command example for running:
python BK.py -data data1.txt -k 8 -s 4 -d 0.05 -o output.txt

k-means algorithm is implemented as a class in cluster module
bisect algorithm is implemented as a class in cluster module, it will call KMeans class
icd function is implemented as a static function in cluster module
"""


def bisecting():
    # load data
    train = dat.get_dat_sets(data)
    # create a bisect k-means model
    model = BisectKMeans(train, max_n_clusters=k, cluster_threshold=s, intra_cluster_distance=d)
    # train the model
    model.fit()
    # make output
    inverted_index_list = cluster_inverted_index(model.clusters)
    print_clusters(output, inverted_index_list)


def print_clusters(output_file_name, list_for_output):
    """
    write the output to file
    """
    # check if output file already exist, delete it if exist
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
    file = open(output_file_name, 'a')
    for i in list_for_output:
        file.write(str(i))
        file.write('\n')
    file.close()


def cluster_inverted_index(clusters):
    inverted_index = []
    # number of clusters
    n_clusters = len(clusters)
    # compute number of data points
    n_data_points = 0
    for cluster in clusters:
        n_data_points += cluster.shape[0]
    for i in range(n_data_points):
        for j in range(n_clusters):
            if i in clusters[j]:
                inverted_index.append(j)
                break
    return inverted_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', help='data file')
    parser.add_argument('-k', help='cluster number', default=8, type=int)
    parser.add_argument('-s', help='cluster size', default=10, type=int)
    parser.add_argument('-d', help='the maximum intra-cluster distance allow within a cluster', default=0.05, type=float)
    parser.add_argument('-o', help='output file name')

    args = parser.parse_args()

    # parameters
    data = args.data
    k = args.k
    s = args.s
    d = args.d
    output = args.o

    # run bisecting
    bisecting()