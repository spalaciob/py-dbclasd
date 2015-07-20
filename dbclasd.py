#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python implementation of the DBCLASD algorithm: a non-parametric clustering algorithm

@copyright: Deutsches Forschungszentrum fuer Kuenstliche Intelligenz GmbH or its licensors, as applicable (2015)
@author: Sebastian Palacio
"""

import sys
import argparse
import traceback
import numpy as np

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def cluster_area(pts, nnfinder):
    """
    Approximate the area of a set of candidate points belonging to a cluster.
    :param pts:
    :return: area, grid_length
    """
    nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts)
    grid_length = max(nnfinder.kneighbors(pts)[0][:, 1])  # Max NN distance of pts
    # TODO: adjust the offset to fit in one cell the points with the largest distance (=grid_length)
    grid_x_lims = np.arange(np.ceil((pts[:, 0].max() - pts[:, 0].min()) / grid_length) + 1) * grid_length + pts[:, 0].min()
    grid_y_lims = np.arange(np.ceil((pts[:, 1].max() - pts[:, 1].min()) / grid_length) + 1) * grid_length + pts[:, 1].min()
    grid = np.histogram2d(pts[:, 0], pts[:, 1], bins=[grid_x_lims, grid_y_lims])[0]

    # # Plot grid (Take it out afterwards)
    # plt.scatter(pts[:, 0], pts[:, 1], c='b')
    # for xbin in grid_x_lims:
    #     plt.plot([xbin, xbin], [grid_y_lims[0], grid_y_lims[-1]], 'g--')
    # for ybin in grid_y_lims:
    #     plt.plot([grid_x_lims[0], grid_x_lims[-1]], [ybin, ybin], 'r--')

    return (grid >= 1).sum() * grid_length, grid_length



def retrieve_neighborhood(cluster, nnfinder):
    """
    Part of the dbclasd algorithm
    :param cluster: cluster
    :param nnfinder: distances from all points
    :param pt_idx: point
    :return:list of points
    """
    # Calculate radius r > sqrt(A/PI * (1 - 1/N^(1/N)))
    area, grid_length = cluster_area(cluster, nnfinder)
    N = float(len(cluster))
    r = np.sqrt((area/np.pi) * (1 - 1/N**(1/N)))
    # The paper doesn't define how large should the radius be (it just defines a lower bound). It makes sense to keep
    # it below the grid length, otherwise, an epsilon is added to comply with the strict lower bound definition
    if r > grid_length:
        r += (grid_length - r) / 2.0
    else:
        r += 0.0001
    query_nn_dists, query_nn_idxs = nnfinder.kneighbors([cluster[0]])
    return query_nn_idxs[query_nn_dists <= r]


def expand_cluster(cluster):
    """

    :param cluster:
    :return:
    """
    pass


def update_candidates(cluster, pts):
    """

    :param cluster:
    :param pts:
    :return:
    """
    for c_pt in cluster:
        if pt not in


def dbclasd(pts):
    """
    Implementation of the DBCLASD clustering algorithm
    :param pts: input points as 2D-array where the first axis is the number of points and the second, the dimensions
    of each point
    :return: clusters (not sure yet)
    """
    assigned_cands = np.zeros(len(pts)) - 1
    candidates = []  # Candidates
    unsuccessful_cands = []
    proccessed_pts = []
    nnfinder = NearestNeighbors(30, algorithm='ball_tree', p=2).fit(pts)
    for pt_idx, pt in enumerate(pts):
        if pt_idx not in assigned_cands:
            new_clust_dists, new_clust_idxs = nnfinder.kneighbors([pt])  # It includes pt_idx already. shape = (1, size)
            new_clust_idxs = new_clust_idxs.flatten()
            new_clust_dists = new_clust_dists.flatten()
            for clust_pt_idx in new_clust_idxs:
                local_answer_idxs = retrieve_neighborhood(pts[new_clust_idxs], nnfinder)
                answer_idxs = new_clust_dists[local_answer_idxs]

                # Update candidates
                for c_idx in answer_idxs:
                    if c_idx not in proccessed_pts:
                        candidates.append(c_idx)
                        proccessed_pts.append(c_idx)
            # Expand Cluster
            change = True
            while change:
                change = False
                while len(candidates) > 0:
                    new_clust_idxs.append(candidates.pop(0))
                    




def main(opts):
    """Main loop"""
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', metavar='FILE', dest='infile', required=False, default=None, help='Input file')

    opts = parser.parse_args(sys.argv[1:])

    try:
        main(opts)
    except:
        print 'Unhandled error!'
        traceback.print_exc()
        sys.exit(-1)

    print 'All Done.'