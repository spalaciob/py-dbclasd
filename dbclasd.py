#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Python implementation of the DBCLASD algorithm: a non-parametric clustering algorithm

@copyright: 2015 - Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.
@author: Sebastian Palacio
"""

import sys
import argparse
import traceback
import numpy as np

from sklearn.neighbors import NearestNeighbors


def retrieve_neighborhood(cluster, pt):
    """
    Part of the dbclasd algorithm
    :param cluster: cluster
    :param pt: point
    :return:list of points
    """
    area = 0


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
            new_clust = np.r_[pt_idx, nnfinder.kneighbors([pt])[1].flatten()]



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