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
from scipy.stats import poisson


def color_marker_generator(n, cm='jet'):
    """
    Returns a generator with as many colors as possible, combined with shapes for plotting with matplotlib. Using
    standard colormaps, this generator can yield up to 6144 unique color/shape combinations.
    :param n: how many different values have to be generated
    :param cm: color map (from matplotlib.cm) to be used as color range.
    :return: a color and a shape. It first iterates over the colors, then the shapes.
    """
    colors = plt.cm.get_cmap(cm)
    shapes = ',.ov^v<>12348sp*hH+xdD|_'
    n = min(n, colors.N * len(shapes))
    mark_iter = np.round(np.linspace(0, colors.N * len(shapes), n)).astype('int')
    for i in range(n):
        yield colors(mark_iter[i] / len(shapes)), shapes[mark_iter[i] % len(shapes)]


def is_distribution_stable(pts, pt, nnfinder):
    """
    Perform a chi-square test on pts and compare the results to pts with pt added to it.
    If the second test stays below the first one, the distribution hasn't changed.
    """
    add_pt = True
    pt_idx = None

    area, gl, whole_grid = cluster_area(pts, nnfinder)
    grid = whole_grid[whole_grid >= 1]  # Assume they are connected (they are, at least diagonally)
    grid_bins = np.unique(grid)
    lambda_hat = grid.mean()
    p_est = poisson.pmf(grid_bins, lambda_hat)
    p_est[-1] = 1-p_est[:-1].sum()  # This last probability has to be expressed as P(x >= p_n) instead of P(x == p_n)
    chisq, p = sci_chisquare([grid[grid == i].sum() for i in grid_bins]/grid.sum(), f_exp=p_est, ddof=grid_bins.size-2)

    if add_pt:
        area2, gl2, whole_grid2 = cluster_area(np.vstack((pts, pt)), nnfinder)
    else:
        area2, gl2, whole_grid2 = cluster_area(np.r_[pts[:pt_idx], pts[pt_idx+1:]], nnfinder)
    grid2 = whole_grid2[whole_grid2 >= 1]
    grid2_bins = np.unique(grid2)
    lambda_hat2 = grid2.mean()
    p_est2 = poisson.pmf(grid2_bins, lambda_hat2)
    p_est2[-1] = 1-p_est2[:-1].sum()  # Same as before
    chisq_2, p_2 = sci_chisquare([grid2[grid2 == i].sum() for i in grid2_bins]/grid2.sum(), f_exp=p_est2, ddof=grid2_bins.size-2)

    # print '%f (%.3f) ¿>=? %f (%.3f)' % (chisq, p, chisq_2, p_2)
    
    return p >= p_2


def cluster_area(pts, nnfinder):
    """
    Approximate the area of a set of candidate points belonging to a cluster. This is only necessary to compute the
    lower bound for the radius of "retrieve_heighborhood".
    :param pts: NxM 2d-array with N points of M dimensions.
    :return: area, grid_length (both floating point values)
    """
    nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts)
    grid_length = nnfinder.kneighbors(pts)[0].max()
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

    return (grid >= 1).sum() * grid_length, grid_length, grid


def retrieve_neighborhood_area(allpts, cluster_idxs, nnfinder, pt_idx=None):
    """
    Retrieve candidates to expand a given cluster. The radius is set to the largest 1-NN distance within all cluster
    points. The area is only computed to guarantee that the condition has been met.
    :param allpts: NxM 2d-array with N points of M dimensions. These are all points in the dataset.
    :param cluster_idxs: the indices whthin allpts of the (partial) cluster to be evaluated/expanded.
    :param nnfinder: instance of scipy's NearestNeighbor fit with all points in the dataset.
    :param pt_idx: index within allpts of the reference point to be checked for neighboring candidates to expand
    the cluster. If None, the first index of cluster_idxs is going to be used instead.
    :return: list of point indices that are within a radius r from the query point, including the query point.
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

    if pt_idx is None:
        pt_idx = cluster_idxs[0]

    query_nn_dists, query_nn_idxs = nnfinder.kneighbors([allpts[pt_idx]])
    return query_nn_idxs[query_nn_dists <= r]


def retrieve_neighborhood_simple(allpts, cluster_idxs, nnfinder, pt_idx=None):
    """
    Retrieve candidates to expand a given cluster. The radius is set to the largest 1-NN distance within all cluster
    points. This method omits the lower bound check and follows directly the definition for the radius.
    :param allpts: NxM 2d-array with N points of M dimensions. These are all points in the dataset.
    :param cluster_idxs: the indices whthin allpts of the (partial) cluster to be evaluated/expanded.
    :param nnfinder: instance of scipy's NearestNeighbor fit with all points in the dataset.
    :param pt_idx: index within allpts of the reference point to be checked for neighboring candidates to expand
    the cluster. If None, the first index of cluster_idxs is going to be used instead.
    :return: list of point indices that are within a radius r from the query point, including the query point.
    """
    cluster = allpts[cluster_idxs]
    two_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(cluster)
    r = two_nnfinder.kneighborgs(cluster)[0][:, 1].max()  # Max NN distance of pts
    if pt_idx is None:
        pt_idx = cluster_idxs[0]
    query_nn_dists, query_nn_idxs = nnfinder.kneighbors([allpts[pt_idx]])
    return query_nn_idxs[query_nn_dists <= r]  # Discard the input point itself


def dbclasd(pts):
    """
    Implementation of the DBCLASD clustering algorithm.
    :param pts: input points as 2D-array where the first axis is the number of points and the second, the dimensions
    of each point
    :return: 1d-array with labels for each point in pts.
    """
    assigned_cands = np.zeros(len(pts)) - 1
    candidates = []  # Candidates
    unsuccessful_cands = []
    proccessed_pts = []
    nnfinder = NearestNeighbors(30, algorithm='ball_tree', p=2).fit(pts)
    two_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts)
    for pt_idx, pt in enumerate(pts):
        if assigned_cands[pt_idx] == -1:
            candidates = []
            new_clust_idxs = nnfinder.kneighbors([pt])[1].flatten()  # It includes pt_idx already. shape = (1, size)
            new_clust_dists = two_nnfinder.kneighbors(pts[new_clust_idxs])[0][:, 1]  # All 1-NNs distances in cluster

            # Retrieve Neighborhood
            two_clust_nnfinder = NearestNeighbors(2, algorithm='ball_tree', p=2).fit(pts[new_clust_idxs])
            r = two_clust_nnfinder.kneighbors(pts[new_clust_idxs])[0][:, 1].max()  # Max NN distance of cluster points
            for clust_pt_idx in new_clust_idxs:
                query_nn_dists, query_nn_idxs = nnfinder.kneighbors([pts[clust_pt_idx]], n_neighbors=len(pts))
                answer_idxs = query_nn_idxs[query_nn_dists <= r][1:]  # Discard the input point itself

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
                    # Evaluate only points that weren't chosen by the first k-nn expansion
                    # if new_candidate in new_clust_idxs:
                    #     continue

                    if is_distribution_stable(pts[new_clust_idxs], pts[new_candidate], nnfinder):
                        # Insert into the cluster
                        new_clust_idxs = np.r_[new_clust_idxs, new_candidate]
                        new_clust_dists = np.r_[new_clust_dists, two_nnfinder.kneighbors(pts[new_candidate])[0][:, 1]]
                        # Retrieve
                        answer_idxs = []
                        for clust_pt_idx in new_clust_idxs:
                            query_nn_dists, query_nn_idxs = nnfinder.kneighbors([pts[clust_pt_idx]], n_neighbors=len(pts))
                            answer_idxs = query_nn_idxs[query_nn_dists <= r][1:] # Discard the input point itself
                        # Update
                        for c_idx in answer_idxs:
                            if c_idx not in proccessed_pts:
                                candidates.append(c_idx)
                                proccessed_pts.append(c_idx)
                        change = True
                    else:
                        unsuccessful_cands.append(new_candidate)
                candidates = unsuccessful_cands[:]
                unsuccessful_cands = []
            assigned_cands[new_clust_idxs] = pt_idx
    return assigned_cands


def load_data(fpath):
    """
    Loads a dataset from a text file. It expects the data to be 2-dimensional and have a format like so:

    12345   98765\n
    56789   01234\n
    ...

    There can be an arbitrary number of spaces between the two numbers in a line and a new line at the end of each line.
    In case the dataset is too big (> 10000 points), a random sample of 10000 points is returned instead.
    :param fpath: Path in the file system where the text file lies.
    :return: a 2d array with at most 10000 points loaded.
    """
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
    """Main loop loads the data, performs the clustering on that data and plots it (assuming is 2D)"""
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
