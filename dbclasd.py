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


def cluster_area(pts):
    """
    Approximate the area of a set of candidate points belonging to a cluster.
    :param pts:
    :return:
    """
    clust_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=1).fit(pts)
    grid_length = max(clust_nnfinder.kneighbors(pts)[0][:, 1])  # Max NN distance of pts
    # TODO: adjust the offset to fit in one cell the points with the largest distance (=grid_length)
    grid_x_lims = np.arange(np.ceil((pts[:, 0].max() - pts[:, 0].min()) / grid_length)) * grid_length + pts[:, 0].min()
    grid_y_lims = np.arange(np.ceil((pts[:, 1].max() - pts[:, 1].min()) / grid_length)) * grid_length + pts[:, 1].min()
    grid = np.histogram2d(pts[:, 0], pts[:, 1], bins=[grid_x_lims, grid_y_lims])[0]
    return (grid >= 1).sum() * grid_length



def retrieve_neighborhood(cluster, dists, pt_idx):
    """
    Part of the dbclasd algorithm
    :param cluster: cluster
    :param dists: distances from all points
    :param pt_idx: point
    :return:list of points
    """
    # Calculate radius r > sqrt(A/PI * (1 - 1/N^(1/N)))
    area = cluster_area(cluster[pt_idx])


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
    pass


def dbclasd(pts):
    """
    Implementation of the DBCLASD clustering algorithm
    :param pts: input points as 2D-array where the first axis is the number of points and the second, the dimensions
    of each point
    :return: clusters (not sure yet)
    """
    assigned_cands = np.zeros(len(pts)) - 1
    cands = []  # Candidates
    unsuccessful_cands = []
    proccessed_pts = []
    nnfinder = NearestNeighbors(30, algorithm='ball_tree', p=1).fit(pts)
    for pt_idx, pt in enumerate(pts):
        if pt_idx not in assigned_cands:
            new_clust, new_clust_dists = nnfinder.kneighbors([pt])  # It includes pt_idx already. shape = (1, size)
            for clust_pt_idx in new_clust:
                answers = retrieve_neighborhood(new_clust, new_clust_dists, clust_pt_idx)



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