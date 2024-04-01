#!/usr/bin/env python
import os
import sys
import math
import numpy as np
import time
sys.path.append("/home/bcov/sc/random/npose")
import npose_util as nu
# import pyRMSD.RMSDCalculator
# import pyRMSD.RMSDCalculator as fast_rmsd_calc
import argparse
import atomic_depth
import io

## This function avoids getting the target input every iteration because they are the same and takes up time.

## Sasa calculation staff (Original from bcov, buwei modified)
# I/O
# target_pdb = "<TO ENTER>"
# scaffold_pdb = "<TO ENTER>"

# Parameters 
sasa_probe_size = 4
sasa_resl = 1.0

# Functions
def get_scaffold_surface(scaffold_npose):
    # This gets a sort of shell around the scaffold
    radii = np.zeros(len(scaffold_npose), dtype=np.float)
    radii[:] = 2

    surf = atomic_depth.AtomicDepth(scaffold_npose[:,:3].reshape(-1), radii, sasa_probe_size, sasa_resl, True, 1)

    vertex_pts = surf.get_surface_vertex_bases().reshape(-1, 3)
    vertex_padded = np.ones((len(vertex_pts), 4))
    vertex_padded[:,:3] = vertex_pts

    # we don't need the entire sasa surface, so cluster it down
    cluster_centers, _ = nu.cluster_points(vertex_padded, sasa_probe_size, find_centers=True)
    #ORIGINAL:
#     return vertex_padded[cluster_centers]
    # debug version:
    try:
        rtrn = vertex_padded[cluster_centers]
    except:
        rtrn = 0
        # print("Scaff npose", scaffold_npose)
        # print("Cluster centers", cluster_centers)
        # print("Vertex padded", vertex_padded)
        # print("V pad shape", vertex_padded.shape)
        raise ValueError
    return rtrn


def get_target_surface(target_pdb):
    target_npose = nu.npose_from_file(target_pdb)
    target_clashgrid = nu.clashgrid_from_points(target_npose, 1.9, 0.33)
    radii = np.zeros(len(target_npose), dtype=np.float)
    radii[:] = 2
    target_surf = atomic_depth.AtomicDepth(target_npose[:,:3].reshape(-1), radii, sasa_probe_size, sasa_resl, True, 1)
    target_face_centers = target_surf.get_surface_face_centers().reshape(-1, 3)
    target_face_areas = target_surf.get_surface_face_areas()
    close_to_target = nu.clashgrid_from_points( target_face_centers, sasa_probe_size, 1)
    return target_npose, target_face_centers, target_face_areas, close_to_target


def calculate_raw_sasa(scaffold_npose, target_face_centers, target_face_areas, close_to_target):
    scaffold_surface = get_scaffold_surface(scaffold_npose)
    surf_close_to_target = close_to_target.arr[tuple(close_to_target.floats_to_indices(scaffold_surface).T)]
    if surf_close_to_target.sum() != 0:
        scaffold_sasa_clashgrid = nu.clashgrid_from_points( scaffold_surface[surf_close_to_target], sasa_probe_size, 1)
        indices = scaffold_sasa_clashgrid.floats_to_indices(target_face_centers)
        used_target_faces = scaffold_sasa_clashgrid.arr[tuple(indices.T)]
        sasa = np.sum(target_face_areas[used_target_faces]) * 2 
    else:
        # no contact between target and scaffold surface!
        sasa = 0
    return sasa

def sasa_score(scaffold_npose, target_face_centers, target_face_areas, close_to_target):
    # Implementation
    # We already give it as npose file
#     scaffold_npose = nu.npose_from_file(scaffold_pdb)
    start_time = time.time()
    # here we give this directly as arguments to the function to reduce computing time
#     target_npose, target_face_centers, target_face_areas, close_to_target = get_target_surface(target_pdb)
#     print("Time cost for processing the target: ", time.time() - start_time)
    sasa = calculate_raw_sasa(scaffold_npose, target_face_centers, target_face_areas, close_to_target)
    tot_time = time.time() - start_time
#     print("raw sasa value: %s"%sasa)
    return sasa, tot_time
