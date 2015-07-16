#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Lorem ipsum dolor sit amet

@copyright: Copyright 2014 Deutsches Forschungszentrum fuer Kuenstliche
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
    area =


def gtcheck(pts, gt):
    if len(pts) != len(gt):
        return False
    for p in pts:
        if not ((p == gt).sum(axis=-1) == p.size).any():
            return False

    return True


def eval_noise(pts, noise):
    gt = []
    af_tolerance = 0
    kl_tolerance = 0
    af_end = False
    kl_end = False
    for i in range(len(noise)):
        noise_data = np.vstack((X, XN[:i]))
        af = AffinityPropagation(preference=-50).fit(noise_data)
        af_cb = np.round(af.cluster_centers_)
        kl, kl_cb = kless_clustering(noise_data)
        kl_cb = np.round(kl_cb)

        if i == 0:
            gt = af_cb

        if not af_end and gtcheck(af_cb, gt):
            af_tolerance += 1
        else:
            af_end = True
            print af_cb, '\nAF FAILED'
            break
        if not kl_end and gtcheck(kl_cb, gt):
            kl_tolerance += 1
        else:
            kl_end = True
            break

        print 'AF: %d - KL: %d' % (af_tolerance, kl_tolerance)

        if af_end and kl_end:
            break
    print 'AF: %d - KL: %d' % (af_tolerance, kl_tolerance)
    return af_tolerance, kl_tolerance

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