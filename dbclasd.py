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
from scipy.stats import chisquare as sci_chisquare
from matplotlib import pyplot as plt


def color_marker_generator(n, cm='jet'):
    """
    Returns a generator with as many colors as possible, combined with shapes for plotting with matplotlib. Using
    standard colormaps, this generator can yield up to 6144 unique color/shape combinations.
    :param n: how many different values have to be generated
    :param cm: color map (from matplotlib.cm) to be used as color range.
    :return: a color and a shape. It first iterates over the colors, then the shapes.
    """
    colors = plt.cm.get_cmap('jet')
    shapes = ',.ov^v<>12348sp*hH+xdD|_'
    n = min(n, colors.N * len(shapes))
    mark_iter = np.round(np.linspace(0, colors.N * len(shapes), n)).astype('int')
    for i in range(n):
        yield colors(mark_iter[i] / len(shapes)), shapes[mark_iter[i] % len(shapes)]


def cluster_area(pts, nnfinder):
    """
    Approximate the area of a set of candidate points belonging to a cluster.
    :param pts:
    :return: area, grid_length
    """
    nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts)
    grid_length = max(nnfinder.kneighbors(pts))
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


def retrieve_neighborhood_area(allpts, cluster_idxs, nnfinder):
    """
    Part of the dbclasd algorithm
    :param cluster: cluster
    :param nnfinder: distances from all points
    :param pt_idx: point
    :return:list of points
    """
    cluster = allpts[cluster_idxs]
    # Calculate radius r > sqrt(A/PI * (1 - 1/N^(1/N)))
    area, grid_length = cluster_area(cluster, nnfinder)
    N = float(len(cluster))
    lower_bound = np.sqrt((area/np.pi) * (1 - 1/N**(1/N)))
    r = grid_length
    try:
        assert r > lower_bound, 'Radius does not meet the lower bound criteria.'
    except AssertionError:
        r = lower_bound * 1.01
    # The paper doesn't define how large should the radius be (it just defines a lower bound). It makes sense to keep
    # it below the grid length, otherwise, an epsilon is added to comply with the strict lower bound definition
    if r > grid_length:
        r += (grid_length - r) / 2.0
    else:
        r += 0.0001
    query_nn_dists, query_nn_idxs = nnfinder.kneighbors([cluster[0]])
    return query_nn_idxs[query_nn_dists <= r]


def retrieve_neighborhood_simple(allpts, cluster_idxs, nnfinder, pt_idx):
    """
    Return neighboring candidates to the first point in cluster, which are within a given radius r. r is the defined
    as the largest distance between all possible nearest neighbors within cluster.
    :param cluster:
    :param nnfinder:
    :param pt_idx:
    :return: None
    """
    cluster = allpts[cluster_idxs]
    two_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(cluster)
    r = two_nnfinder.kneighborgs(cluster)[0][:, 1].max()  # Max NN distance of pts
    query_nn_dists, query_nn_idxs = nnfinder.kneighbors([allpts[pt_idx]])
    return query_nn_idxs[query_nn_dists[1:] <= r]  # Discard the input point itself


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
    two_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts)
    for pt_idx, pt in enumerate(pts):
        if pt_idx not in assigned_cands:
            new_clust_idxs = nnfinder.kneighbors([pt])[1].flatten()  # It includes pt_idx already. shape = (1, size)
            new_clust_dists = two_nnfinder.kneighbors(pts[new_clust_idxs])[0][:, 1]  # All 1-NNs distances in cluster

            # Retrieve Neighborhood
            two_clust_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts[new_clust_idxs])
            r = two_clust_nnfinder.kneighbors(pts[new_clust_idxs])[0][:, 1].max()  # Max NN distance of cluster points
            for clust_pt_idx in new_clust_idxs:
                query_nn_dists, query_nn_idxs = nnfinder.kneighbors([pts[clust_pt_idx]])
                answer_idxs = query_nn_idxs[query_nn_dists <= r][1:]  # Discard the input point itself
                # answer_idxs = retrieve_neighborhood_simple(pts, new_clust_idxs, nnfinder, clust_pt_idx)

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
                    new_candidate = candidates.pop(0)
                    chisq_0, p_0 = sci_chisquare(new_clust_dists)
                    new_clust_idxs = np.r_[new_clust_idxs, new_candidate]
                    new_clust_dists = two_nnfinder.kneighbors(pts[new_clust_idxs])[0][:, 1]
                    chisq_1, p_1 = sci_chisquare(new_clust_dists)
                    print int(p_0*10000), int(p_1*10000)
                    if int(p_0*10000)-1 <= int(p_1*10000) <= int(p_0*10000)+1:
                        # Retrieve and update answers once more
                        for clust_pt_idx in new_clust_idxs:
                            query_nn_dists, query_nn_idxs = nnfinder.kneighbors([pts[clust_pt_idx]])
                            answer_idxs = query_nn_idxs[query_nn_dists[1:] <= r]  # Discard the input point itself

                        for c_idx in answer_idxs:
                            if c_idx not in proccessed_pts:
                                candidates.append(c_idx)
                                proccessed_pts.append(c_idx)
                        change = True
                    else:
                        print 'distribution change!'
                        unsuccessful_cands.append(new_candidate)
                candidates = unsuccessful_cands[:]
                unsuccessful_cands = []
            assigned_cands[new_clust_idxs] = pt_idx
    return assigned_cands


def load_data(fpath):
    dim = len(open(fpath).readline().split())
    syn_data = np.array(open(fpath).read().split(), dtype='float')
    syn_data = syn_data.reshape(syn_data.size/dim, dim)
    if len(syn_data) > 10000:
        idxs = np.arange(len(syn_data))
        np.random.shuffle(idxs)
        syn_data = syn_data[idxs[:10000]]
        print 'subsampling to 10k points'
    return syn_data


def main(opts):
    """Main loop"""
    pts = load_data(opts.infile)
    labels = dbclasd(pts)
    colors = color_marker_generator(np.unique(labels).size)
    print 'plotting...'
    for lbl in np.unique(labels):
        col, shp = colors.next()
        cluster = pts[labels==lbl]
        plt.scatter(cluster[:, 0], cluster[:, 1], c=col, marker=shp)
    plt.show()



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