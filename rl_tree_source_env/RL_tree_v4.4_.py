#!/usr/bin/env python
# coding: utf-8

import sys
import pymesh

sys.path.append('/home/ilutz/BINDERS_RL/2_RL/tree_source/')

from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign,exp
from math import pi as mPI
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import os
import glob
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import itertools
import operator
from functools import reduce
import pickle
from sklearn.cluster import KMeans
import scipy
from scipy import signal as scipysignal
from sklearn.cluster import DBSCAN
from scipy.stats import norm
import random
import statistics
import time
import timeit
from mpl_toolkits.mplot3d import Axes3D
import math
import localization as lx
import gzip

import npose_util as nu
import motif_stuff2

import subprocess
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

import math
from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import getpy
import xbin
import h5py

import voxel_array
import npose_util as nu

from importlib import reload
reload(nu)

from sasa_util_eff import sasa_score
from sasa_util_eff import get_target_surface

zero_ih = nu.npose_from_file('/home/ilutz/BINDERS_RL/2_RL/zero_ih_long.pdb')



tg_res = []
with open('../../1_rifgen/resid.txt','r') as res:
    for line in res:
        tg_res.append(int(line.split()[0]))

tg_atoms = []
restricted = []
with open('../tg.pdb','r') as tg:
    for line in tg:
        splt = line.split()
        if len(splt) > 9 and splt[0] == 'ATOM':
            if int(splt[5]) in tg_res:
                tg_atoms.append([float(splt[6]),float(splt[7]),float(splt[8])])
                restricted.append(line)

if len(glob.glob('../tg_restricted.pdb')) == 0:
    with open('../tg_restricted.pdb','w') as newf:
        for j in restricted:
            newf.write(j)

target_npose, target_face_centers, target_face_areas, close_to_target = get_target_surface('../tg_restricted.pdb')

B_SASA_TG = sum(target_face_areas) + 385



def load_full_rif_rots(rif_rots_pdb):
    # not actually full rif rots -- just full relative to npose
    # only grab sidechain heavy atoms for clash checking
    # glycines and alanines will have empty lists, already checked
    full_rr_dict = {}
    bb_atoms = ['N','CA','C','O','CB']
    with open(rif_rots_pdb, 'r') as pdb:
        for line in pdb:
            splt = line.split()
            
            if len(splt) > 1:
                irot = int(splt[5])-1
                if irot not in full_rr_dict:
                    full_rr_dict[irot] = []
                # skip hydrogens, backbone atoms, CB (already clash checked)
                if 'H' not in splt[2] and splt[2] not in bb_atoms: 
                    full_rr_dict[irot].append([float(splt[6]),float(splt[7]),float(splt[8]),1])
                
    full_rr_dict_out = {}
    for x in full_rr_dict:
        full_rr_dict_out[x] = np.array(full_rr_dict[x])
    
    return full_rr_dict_out


# get RIF, get RIF rotamers
RIF_dict = getpy.Dict(np.dtype('int64'), np.dtype('int64'), 0)
pyrif_pt = glob.glob('../py_rif.h5')[0]


with h5py.File(pyrif_pt, 'r') as f:

    RIF_dict[f['xbin_key']] = f['offsets']

    RIF_scores = np.zeros(f['scores'].shape, np.float16)
    RIF_irots = np.zeros(f['irots'].shape, np.int16)

    RIF_scores[:] = f['scores']
    RIF_irots[:] = f['irots']

    RIF_binner = xbin.XformBinner(*list(f['cart_ori_bound']))

    
npose_rif_rots, rif_rot_seq = nu.npose_from_file('../rif_rotamers.pdb', aa=True)

# reshape to index by irot
npose_rif_rots = npose_rif_rots.reshape(int(len(npose_rif_rots)/5),5,4)

# get just unchecked heavy atoms of rotamer sidechains for clash checks
full_rif_rots = load_full_rif_rots('../rif_rotamers.pdb')


# ideal and rif_rt for rif scoring
ideal_npose = nu.npose_from_length(1)
ideal_tpose = nu.tpose_from_npose(ideal_npose)
ideal_rif_frame = nu.npose_to_rif_hash_frames(ideal_npose)
rif_rt = np.linalg.inv(ideal_tpose[0]) @ ideal_rif_frame[0]


def dump_rif_rots(npose):

    #tpose = nu.tpose_from_npose( npose )
    #rif_frames = tpose @ rif_rt
    
    rif_frames2 = nu.npose_to_rif_hash_frames( npose )

    # This is only strictly true if npose is ideal
    #print(np.allclose(rif_frames, rif_frames2))

    keys = RIF_binner.get_bin_index(rif_frames2)

    offsets = RIF_dict[keys]
    
    out_rifrots = {}

    for i in range(len(offsets)):
        seqpos = i + 1

        offset = offsets[i]

        while ( RIF_irots[offset] != -1 ):

            irot = RIF_irots[offset]
            score = RIF_scores[offset]
            
            if seqpos not in out_rifrots:
                out_rifrots[seqpos] = ([],[])
            out_rifrots[seqpos][0].append(score)
            out_rifrots[seqpos][1].append(irot)

            offset += 1
    
    # assumes rifrots are sorted already, which seems to be the case
    return out_rifrots



def greedy_rif_choices(rifrots):
    positions = []
    greedy_irots = []
    greedy_score = 0
    greedy_scores = []
    
    for pos in rifrots:
        positions.append(pos)
        minscore = rifrots[pos][0][0]
        greedy_irots.append(rifrots[pos][1][0])
        greedy_score += minscore
        greedy_scores.append(minscore)
        
    return positions, greedy_irots, greedy_score


def calc_rif_score(npose):

    rifrots = dump_rif_rots(npose)
    
    # greedy works for majority, and/or no hits -- relatively fast, worth trying first
    positions, greedy_irots, greedy_score = greedy_rif_choices(rifrots)
    
    return positions, greedy_irots, greedy_score
    

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



rr = np.load('/home/ilutz/BINDERS_RL/2_RL/all_loops_bCov.npz', allow_pickle=True)
all_loops = [rr[f] for f in rr.files][0]



def align_loop(build, end, loop):
    # returns loop aligned to assigned end of build
    tpose1 = nu.tpose_from_npose(loop)
    tpose2 = nu.tpose_from_npose(build)
    
    itpose1 = np.linalg.inv(tpose1)
    
    if end == 'C': # C term addition
        xform = tpose2[-1] @ itpose1[0]
        aligned_npose1 = nu.xform_npose(xform, loop)
        return aligned_npose1[5:]
    
    else: # N term addition
        xform = tpose2[0] @ itpose1[-1]
        aligned_npose1 = nu.xform_npose(xform, loop)
        return aligned_npose1[:-5]
    
    
def extend_helix(build, end, res):
    
    if end == 'C': # C term addition
        ext = align_loop(build, 'C', zero_ih)
        return ext[:(res*5)]
    
    else: # N term addition
        # note: can't handle 0 ext, but doesn't need to
        ext = align_loop(build, 'N', zero_ih)
        return ext[-(res*5):]


def kClusBin(clusts1=1,clusts2=32,clusts3=16):
    # loopData = 'loop_lib_feats'
    # loopData = 'loop_lib_feats_normVec'
    loopData = 'loop_lib_feats_no_norm'
    # loopData = 'loop_lib_feats_two_norm'
    rr = np.load(f'/home/ilutz/BINDERS_RL/2_RL/{loopData}.npz', allow_pickle=True)
    feats, name, fnames = [rr[f] for f in rr.files]
    name = name.astype(np.int32)
    kClus = KMeans(n_clusters=clusts1)
    kClus2 = KMeans(n_clusters=clusts2)
    kClus3 = KMeans(n_clusters=clusts3)
    
    kClus.fit(feats[:,:3])
    kClus2.fit(feats[:,3:-1])
    kClus3.fit(np.reshape(feats[:,-1],(len(feats),1)))
    
    binned_loops1 = {}
    
    for ind in range(len(kClus.labels_)):
        clus1 = kClus.labels_[ind]
        clus2 = kClus2.labels_[ind]
        clus3 = kClus3.labels_[ind]
        
        clusNum = int(clus1*1e6 + clus2*1e3 + clus3)
        
        if clusNum not in binned_loops1:
            binned_loops1[clusNum] = []

        binned_loops1[clusNum].append(all_loops[name[ind]])

    binned_loops = {}

    for k,v in enumerate(binned_loops1):
        binned_loops[k] = binned_loops1[v]
    
    binned_loops2 = {}
    ct = 0
    for i in binned_loops:
        if len(binned_loops[i]) > 4:
            binned_loops2[ct] = binned_loops[i]
            ct += 1
            
    return binned_loops2



def voxelize_mesh(mesh_path, buffer_dist): # can eventually add resolution if desired
    
    mesh = pymesh.load_mesh(mesh_path)
    maxs = mesh.vertices.max(axis=0)
    mins = mesh.vertices.min(axis=0)
    
    wnum_hash = {}
    queries = []

    for i in range(int(mins[0]),int(maxs[0])+1):
        for j in range(int(mins[1]),int(maxs[1])+1):
            for k in range(int(mins[2]),int(maxs[2])+1):
                queries.append([i,j,k])

    wnums = abs(pymesh.compute_winding_number(mesh,queries,engine='fast_winding_number'))
    successes = [queries[x] for x,xval in enumerate(wnums) if xval > 0.99]

    for qi,que in enumerate(queries):
        key = str(int(round(que[0])))+'_'+str(int(round(que[1])))+'_'+str(int(round(que[2])))
        if key not in wnum_hash:
            wnum_hash[key] = wnums[qi] > 0.99
        
    # turning off hash entries within some distance of mesh False entries
    # square buffer shape -- change to sphere eventually
    
    wh_buff = wnum_hash.copy()
    buffer_ds = []
    
    for i in range(-buffer_dist,buffer_dist+1):
        for j in range(-buffer_dist,buffer_dist+1):
            for k in range(-buffer_dist,buffer_dist+1):
                buffer_ds.append(np.array([i,j,k]))

    for i in wnum_hash:
        if not wnum_hash[i]:
            coors = np.array([int(x) for x in i.split('_')])
            for d in buffer_ds:
                d_off = coors + d
                off_key = str(int(round(d_off[0])))+'_'+str(int(round(d_off[1])))+'_'+str(int(round(d_off[2])))
                if off_key in wh_buff:
                    wh_buff[off_key] = False
    
    # trim hash table to minimum size to speed up sampling -- much faster to copy table too
    wh_min = {}
    for i in wh_buff:
        if wh_buff[i]:
            wh_min[i] = True
            
    return wh_min


def check_hash(pt_set,wnum_hash): 
    
    for i in pt_set:
        
        key = str(int(round(i[0])))+'_'+str(int(round(i[1])))+'_'+str(int(round(i[2])))
        
        if key not in wnum_hash:
            return False
            
    return True


def check_clash(build_set, build_end, query_set, clash_threshold = 2.85, diameter_threshold = 999):
    if len(build_set) == 0 or len(query_set) == 0:
        return True

    seq_buff = 5 # +1 from old clash check, should be fine
    if len(query_set) < seq_buff:
        seq_buff = len(query_set)
    elif len(build_set) < seq_buff:
        seq_buff = len(build_set)

    axa = scipy.spatial.distance.cdist(build_set,query_set)
    
    for i in range(seq_buff):
        for j in range(seq_buff-i):
            # add sequence buffer to different ends accordingly
            if build_end == 'C':
                axa[-(i+1)][j] = clash_threshold + 0.1
            else:
                axa[i][-(j+1)] = clash_threshold + 0.1

    if np.min(axa) < clash_threshold: # clash condition
        return False
    if np.max(axa) > diameter_threshold: # compactness condition
        return False
    
    return True


def get_avg_sc_neighbors(ca_cb, care_mask):
    
    conevect = (ca_cb[:,1] - ca_cb[:,0] )

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros(len(ca_cb))

    for i in range(len(ca_cb)):

        if ( not care_mask[i] ):
            continue
        
        vect = ca_cb[:,0] - ca_cb[i,1]
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        vect = vect[(vect_length2 < max2) & (vect_length2 > 4)]
        vect_length2 = vect_length2[(vect_length2 < max2) & (vect_length2 > 4)]

        vect_length = np.sqrt(vect_length2)
        
        vect = vect / vect_length[:,None]

        dist_term = np.zeros(len(vect))
        
        for j in range(len(vect)):
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6

        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5
        angle_term[angle_term < 0] = 0

        neighs[i] = np.sum( dist_term * np.square( angle_term ) )

    return neighs


def score_build(input_pose, input_ss=None):

    pose = input_pose.reshape(int(len(input_pose)/5),5,4)

    ca_cb = pose[:,1:3,:3]
    care_mask = np.ones(len(pose),dtype=int)
    
    neighs = get_avg_sc_neighbors( ca_cb, care_mask )

    #surface = neighs < 2
    is_core_boundary = neighs > 2
    is_core = neighs > 5.2
    
    # to score core/boundary
    score_mask = is_core_boundary
    care_mask = is_core_boundary
    
    # percent of 9mers with hit in core/boundary (brian says >80%, maybe even >90%)
    hits, froms, tos, t_t = motif_stuff2.motif_score_npose( input_pose, score_mask, care_mask )
    
    no_hits = 0
    
    for i in range(len(care_mask)-8):
        hits = False
        for j in range(9):
            if i+j in tos or i+j in froms:
                hits = True
        if hits:
            no_hits += 1

    pc_scn = is_core.mean()
    motif_score = no_hits / (len(care_mask)-8)
    rif_score = -calc_rif_score(input_pose)[-1] # make it positive
    try:
        sasa_scn, sasa_time = sasa_score(input_pose, target_face_centers, target_face_areas, close_to_target)
        int_sasa = sasa_scn
    except:
        int_sasa = 0
    core_pred = (-37.18055422*is_core.mean() + 31.84588908*neighs.mean() + -2.30858968*np.median(neighs) + -20.66928756*np.std(neighs)) + 31.203420096653858

    a_pc = 3
    b_pc = 0.25
    m_pc = 100
    score_pc = m_pc/(1+exp(-a_pc*10*(pc_scn - b_pc)))
    
    a_mot = 3
    b_mot = 0.9
    m_mot = 10
    score_mot = m_mot/(1+exp(-a_mot*10*(motif_score - b_mot))) + 1
    
    a_rif = .015
    b_rif = 40
    m_rif = 125
    if rif_score == 0:
        score_rif = 0
    else:
        score_rif = m_rif/(1+exp(-a_rif*10*(rif_score - b_rif)))
        
    a_sasa = .0011
    b_sasa = B_SASA_TG
    m_sasa = 100
    score_sasa = m_sasa/(1+exp(-a_sasa*10*(int_sasa - b_sasa))) + 1

    a_fix = .02
    b_fix = 60
    m_fix = 2
    core_fix = m_fix/(1+exp(-a_fix*10*(core_pred - b_fix)))
    
    score_wt_pre = score_pc*score_mot*score_rif*score_sasa*core_fix*0.001 # additional reweight -- may actually be key

    if input_ss:

        worst_helix = 999
        prev = 'Y'
        first = 0
        for ind,i in enumerate(input_ss):
            if i == 'H' and prev == 'L':
                first = ind
            if i == 'L' and prev == 'H':
                second = ind-1
                hel_neighs = neighs[first:second+1].mean()
                if hel_neighs < worst_helix:
                    worst_helix = hel_neighs
            elif i == 'H' and ind == len(input_ss)-1:
                second = ind
                hel_neighs = neighs[first:second+1].mean()
                if hel_neighs < worst_helix:
                    worst_helix = hel_neighs
            prev = i

        a_whel = 0.5
        b_whel = 2.1
        m_whel = 1.1
        whel_penalty = m_whel/(1+exp(-a_whel*10*(worst_helix - b_whel)))

        score_wt_pre = score_wt_pre * whel_penalty
    else:
        worst_helix = 0

    if score_wt_pre > 0.25:
        a_final,b_final,m_final = .003, 200, 1
        score_wt = (m_final/(1+exp(-a_final*10*(score_wt_pre - b_final))))
    else:
        score_wt = 0
    
    return pc_scn, motif_score, rif_score, int_sasa, worst_helix, core_pred, score_wt



def test_builder(num_runs = 80000001):
    ct = 0
    while ct < num_runs:

        # if ct % 2000 == 0 and len(g.builds) > 0:
        #     series = [ind for ind,_ in enumerate(g.builds)]
        #     # print('lengths')
        #     # sns.scatterplot(x = series, y = [len(x) for x in g.builds])
        #     # plt.show()
        #     # print('build times')
        #     # sns.scatterplot(x = series, y = g.build_times)
        #     # plt.show()

        #     avg_lengths = []
        #     best_lengths = []
        #     avg_bts = []
        #     avg_series = []
            
        #     avg_pc_scn = []
        #     best_pc_scn = []
            
        #     avg_mots = []
        #     best_mots = []
            
        #     avg_rifs = []
        #     best_rifs = []
            
        #     avg_sasa = []
        #     best_sasa = []
            
        #     avg_overall = []
        #     best_overall = []
            
        #     if len(g.builds) < 10000:
        #         bin_size = 50
        #     elif len(g.builds) < 25000:
        #         bin_size = 250
        #     elif len(g.builds) < 50000:
        #         bin_size = 500
        #     elif len(g.builds) < 100000:
        #         bin_size = 1000
        #     else:
        #         bin_size = 5000

        #     for i in series:
        #         if i % bin_size == 0 and i != 0:
        #             avg_series.append(i)
        #             avg_lengths.append(np.mean([len(x)/5 for x in g.builds[i-bin_size:i]]))
        #             best_lengths.append(max([len(x)/5 for x in g.builds[i-bin_size:i]]))
        #             avg_bts.append(np.mean(g.build_times[i-bin_size:i]))
                    
        #             avg_pc_scn.append(np.mean(g.pc_scns[i-bin_size:i]))
        #             best_pc_scn.append(max(g.pc_scns[i-bin_size:i]))
                    
        #             avg_mots.append(np.mean(g.motif_scores[i-bin_size:i]))
        #             best_mots.append(max(g.motif_scores[i-bin_size:i]))
                    
        #             avg_rifs.append(np.mean(g.rif_scores[i-bin_size:i]))
        #             best_rifs.append(max(g.rif_scores[i-bin_size:i]))
                    
        #             avg_sasa.append(np.mean(g.sasas[i-bin_size:i]))
        #             best_sasa.append(max(g.sasas[i-bin_size:i]))
                    
        #             try:
        #                 avg_overall.append(np.mean(g.overall_scores[i-bin_size:i]))
        #                 best_overall.append(max(g.overall_scores[i-bin_size:i]))
        #             except:
        #                 pass

        #     print('avg lengths, best lengths')
        #     sns.scatterplot(x = avg_series, y = avg_lengths)
        #     sns.scatterplot(x = avg_series, y = best_lengths)
        #     plt.show()
        #     #plt.savefig('avglengths_bestlengths.png')
        #     #plt.clf()
        #     print('avg build times')
        #     sns.scatterplot(x = avg_series, y = avg_bts)
        #     plt.show()
        #     #plt.savefig('avg_build_times.png')
        #     #plt.clf()
        #     print('avg pc scn')#, best pc scn')
        #     sns.scatterplot(x = avg_series, y = avg_pc_scn)
        #     plt.show()
        #     #plt.savefig('avg_pc_scn.png')
        #     #plt.clf()
        #     print('best pc scn')
        #     sns.scatterplot(x = avg_series, y = best_pc_scn)
        #     plt.show()
        #     #plt.savefig('best_pc_scn.png')
        #     #plt.clf()
        #     print('avg motif score')#, best motif score')
        #     sns.scatterplot(x = avg_series, y = avg_mots)
        #     plt.show()
        #     #sns.scatterplot(x = avg_series, y = best_mots)
        #     #plt.savefig('avg_motif_score.png')
        #     #plt.clf()
        #     print('avg rif score')
        #     sns.scatterplot(x = avg_series, y = avg_rifs)
        #     plt.show()
        #     #plt.savefig('avg_rif_score.png')
        #     #plt.clf()
        #     print('best rif score')
        #     sns.scatterplot(x = avg_series, y = best_rifs)
        #     plt.show()
        #     #plt.savefig('best_rif_score.png')
        #     #plt.clf()
        #     print('avg sasa')
        #     sns.scatterplot(x = avg_series, y = avg_sasa)
        #     plt.show()
        #     #plt.savefig('avg_sasa.png')
        #     #plt.clf()
        #     print('best sasa')
        #     sns.scatterplot(x = avg_series, y = best_sasa)
        #     plt.show()
        #     #plt.savefig('best_sasa.png')
        #     #plt.clf()
            
        #     try:
        #         print('avg overall')#, best pc scn')
        #         sns.scatterplot(x = avg_series, y = avg_overall)
        #         plt.show()
        #         #plt.savefig('avg_overall.png')
        #         #plt.clf()
        #         print('best overall')
        #         sns.scatterplot(x = avg_series, y = best_overall)
        #         plt.show()
        #         #plt.savefig('best_overall.png')
        #         #plt.clf()
        #     except:
        #         pass
            
        #     print('#############################################################')
            
        
        g.test_build()
        
        ct += 1



class tree_builder():
    def __init__(self, stub, wnum_hash):
        
        # starting point, wnum hash, build lists
        self.stub = stub
        self.wnum_hash = wnum_hash
        self.builds = []
        self.build_paths = []
        self.build_sstructs = []
        
        # score lists
        self.pc_scns = []
        self.motif_scores = []
        self.rif_scores = []
        self.sasas = []
        self.worst_helices = []
        self.core_preds = []
        self.overall_scores = []
        self.score_lists = [self.pc_scns,self.motif_scores,self.rif_scores,self.sasas,self.worst_helices,self.core_preds,self.overall_scores]
        
        # general build params
        self.max_helix_length = 25
        self.min_helix_length = 9
        self.max_ext_length = self.max_helix_length - 4 # set by max helix length
        self.min_ext_length = self.min_helix_length - 4 # set by min helix length
        self.max_helices = 7
        self.min_helices = 3
        self.max_residues = 120
        self.min_score_residues = int(len(stub)/5) + 50 # score at this length - higher is faster
        self.clash_threshold = 2.85 # 2.75 best? or too close
        self.diam_threshold = 45 # plateaus build lengths
        
        # weight params
        self.starting_wf = 10 # starting weights (changes how quickly choices become dominant) - don't change
        self.ext_multiplier = round((len(binned_loops)*2) / (self.max_ext_length+1-self.min_ext_length))
        self.ext_start_wf = self.ext_multiplier * self.starting_wf # reweight to match loop total
        self.zero_wf_initial = round(0.1*len(binned_loops)*2) # start at 10%
        self.zero_wf_scalar = self.zero_wf_initial / 2 # scales with distance down tree past minimum helices
        self.upweight_cutoff = 0.4 # how dominant choice can be relative to all other choices
        self.score_uw_scalar = 8000 # take score weight (between 0 and 1) and multiply for upweighting
        # score/constraint upweight is relative to starting_wf * len(binned_loops)*2 = sum initial wts
        
        # try/attempt params
        self.try_initial = 4 # starting try limit
        self.try_scalar = 3 # scales with distance down tree

        # choices - C ext, C loop, N ext, N loop, zero choice
        self.all_choices = ['C_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['C_l_'+str(x) for x in binned_loops] + ['N_e_'+str(x) for x in range(self.min_ext_length,self.max_ext_length+1)] + ['N_l_'+str(x) for x in binned_loops] + ['C_e_0'] # call zero choice a C extension of 0
        
        # mapping of choices to weight indices for upweighting
        self.choice_wt_map = {}
        for cnd,chce in enumerate(self.all_choices):
            self.choice_wt_map[chce] = cnd

        # choice options and initial weights depending on previous choice (or start) - zero choice default 0
        self.initial_weights = {}
        self.initial_weights['start'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        self.initial_weights['C_e'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for x in binned_loops] + [0]
        self.initial_weights['C_l'] = [self.ext_start_wf for x in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        self.initial_weights['N_e'] = [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0 for x in range(self.min_ext_length,self.max_ext_length+1)] + [self.starting_wf for _ in binned_loops] + [0]
        self.initial_weights['N_l'] = [0 for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [self.ext_start_wf for _ in range(self.min_ext_length,self.max_ext_length+1)] + [0 for _ in binned_loops] + [0]
        
        # set up build graph with first choice options, weights
        self.build_graph = [{},self.initial_weights['start'].copy()]
        
        # debug
        self.build_times = []

    
    def upweight(self, path, factor):
        
        for path_index,choice_name in enumerate(path):

            if (path_index+1)%2 == 0:
                
                # get appropriate index for upweighting
                wt_ind = self.choice_wt_map[choice_name]
                
                # access build path weights
                t1 = self.call_graph_path(path[:path_index-1])[1] 
                
                # number of choices down the tree, where first choice = 0
                num_choices = int((path_index-1)/2)
                
                # calc upweight cutoff based on position in path
                uw_cutoff = min( (self.upweight_cutoff + 0.025*num_choices), 0.6 )
                
                if (t1[int(wt_ind)]+int(factor)) < (sum(t1)*uw_cutoff):
                    t1[int(wt_ind)] += int(factor)
                else:
                    t1[int(wt_ind)] = int(sum(t1)*uw_cutoff)
                    
    
    def call_graph_path(self, path):
        
        return reduce(operator.getitem, path, self.build_graph)
    
    
    def test_build(self): # v4
        
        # start time, initialize single build
        start = time.time()
        build = self.stub.copy()
        build_path = []
        current_length = round(len(build)/5)
        current_helices = 0
        build_ss = 'H'*current_length
        prev_choice = []
        prev_choice_type = 'e'
        
        # while build does not exceed residue and helix limits
        while current_length < self.max_residues and current_helices < self.max_helices:
            
            # initialize no valid choice, 0 attempts
            valid_choice = False
            tries = 0
            
            # while valid choice not found
            while not valid_choice:
                
                # pick a choice, align it, keep track of end (N or C) and choice type (e or l)
                wts = self.call_graph_path(build_path)[1]
                choice_key = random.choices(self.all_choices, weights=wts, k=1)[0]
                choice_end, choice_type, choice_num = choice_key.split('_')
                
                if choice_type == 'e':
                    if len(build_path) == 0: # stub must be 5 residues, so reduce extension by 1
                        choice = extend_helix(build, choice_end, int(choice_num)-1)
                    else:
                        choice = extend_helix(build, choice_end, int(choice_num))
                    cc_build = build
                else:
                    loop_index = int(random.random()*len(binned_loops[int(choice_num)]))
                    rand_loop = binned_loops[int(choice_num)][loop_index]
                    if choice_end == 'C':
                        cc_build = build[:-4*5]
                    else:
                        cc_build = build[4*5:]
                    choice = align_loop(cc_build, choice_end, rand_loop)
                
                # check length, check hash, check clash - accept if all pass
                if current_length + round(len(choice)/5) <= self.max_residues:
                    if check_hash(choice, self.wnum_hash):
                        if check_clash(cc_build,choice_end,choice,clash_threshold=self.clash_threshold,diameter_threshold=self.diam_threshold):
                            valid_choice = True
                
                # update attempts
                tries += 1
                
                # if tries exceed limit and no choice found, or zero choice picked - end condition
                if ((tries >= (self.try_scalar*current_helices+self.try_initial)) and (not valid_choice)) or (choice_key == 'C_e_0'):
                    
                    # add zero choice to build path if picked (to allow it to be upweighted)
                    if choice_key == 'C_e_0':
                        build_path.append(0)
                        build_path.append(choice_key)
                        
                    # remove loop if it was last choice, update build path, length, secondary struct, helices
                    if prev_choice_type == 'l' and len(prev_choice) > 0:
                        # remove loop, re-extend by loop cap helix residues
                        if prev_choice_end == 'C': 
                            cc_build = build[:-len(prev_choice)]
                            build = np.append(cc_build,extend_helix(cc_build,'C',4),0)
                            build_ss = build_ss[:-round(len(prev_choice)/5)] + 'H'*4
                        else:
                            cc_build = build[len(prev_choice):]
                            build = np.append(extend_helix(cc_build,'N',4),cc_build,0)
                            build_ss = 'H'*4 + build_ss[round(len(prev_choice)/5):]
                        build_path = build_path[:-2]
                        current_length = round(len(build)/5)
                        current_helices = ((len(build_path)/2)+1)/2
                
                    # if above score length threshold
                    if current_length >= self.min_score_residues:
                    
                        # score build, upweight accordingly
                        scores = score_build(build, input_ss=build_ss)
                        if scores[-1] > 0:
                            self.upweight(build_path,scores[-1]*self.score_uw_scalar)
                        for s_ind,score in enumerate(scores):
                            self.score_lists[s_ind].append(score)
                    
                    else: # append zeroes otherwise
                        for s_list in self.score_lists:
                            s_list.append(0)
                    
                    # record build, details, stop time
                    self.builds.append(build)
                    self.build_paths.append(build_path)
                    self.build_sstructs.append(build_ss)
                    end = time.time()
                    self.build_times.append(end-start)
                    
                    return build
            
            # found valid choice
            # update build path, build, length, helices, secondary struct
            build_path.append(0)
            build_path.append(choice_key)
            if choice_end == 'C': # add to C
                build = np.append(cc_build,choice,0)
                if choice_type == 'e':
                    build_ss = build_ss + round(len(choice)/5)*'H'
                else:
                    build_ss = build_ss + (round(len(choice)/5)-8)*'L' + 'H'*4
            else: # add to N
                build = np.append(choice,cc_build,0)
                if choice_type == 'e':
                    build_ss = round(len(choice)/5)*'H' + build_ss
                else:
                    build_ss = 'H'*4 + (round(len(choice)/5)-8)*'L' + build_ss
            current_length = round(len(build)/5)
            current_helices = ((len(build_path)/2)+1)/2
            
            # keep track of previous choice, end, type to trim loops
            prev_choice, prev_choice_end, prev_choice_type = choice, choice_end, choice_type
            
            # if new branch point in tree:
            if choice_key not in self.call_graph_path(build_path[:-2])[0]:
            
                # find new choice list, weights according to previous choice
                new_weights = self.initial_weights[choice_key[:3]].copy()
                
                # add zero choice if path is long enough (at least 3 helices) and last addition was helix
                if current_helices >= self.min_helices and choice_type == 'e':
                    # weight according to build path length (increase bias to terminate if longer)
                    new_weights[-1] == self.zero_wf_initial + max(0,self.zero_wf_scalar*(current_helices-self.min_helices))
                
                # create new branch point in tree with new choices/weights
                t1 = self.call_graph_path(build_path[:-2])[0][choice_key] = [{},new_weights]
            
        # reached length or helix limit
        # remove loop if it was last choice, update build path, length, secondary struct, helices
        if prev_choice_type == 'l' and len(prev_choice) > 0:
            # remove loop, re-extend by loop cap helix residues
            if prev_choice_end == 'C': 
                cc_build = build[:-len(prev_choice)]
                build = np.append(cc_build,extend_helix(cc_build,'C',4),0)
                build_ss = build_ss[:-round(len(prev_choice)/5)] + 'H'*4
            else:
                cc_build = build[len(prev_choice):]
                build = np.append(extend_helix(cc_build,'N',4),cc_build,0)
                build_ss = 'H'*4 + build_ss[round(len(prev_choice)/5):]
            build_path = build_path[:-2]
            current_length = round(len(build)/5)
            current_helices = ((len(build_path)/2)+1)/2
        
        # if above score length threshold
        if current_length >= self.min_score_residues:

            # score build, upweight accordingly
            scores = score_build(build, input_ss=build_ss)
            if scores[-1] > 0:
                self.upweight(build_path,scores[-1]*self.score_uw_scalar)
            for s_ind,score in enumerate(scores):
                self.score_lists[s_ind].append(score)

        else: # append zeroes otherwise
            for s_list in self.score_lists:
                s_list.append(0)

        # record build, details, stop time
        self.builds.append(build)
        self.build_paths.append(build_path)
        self.build_sstructs.append(build_ss)
        end = time.time()
        self.build_times.append(end-start)

        return build


# # make loop bins
# binned_loops = kClusBin(clusts1=1,clusts2=32,clusts3=16)

# # save loop bins, for easier use later
# save_obj(binned_loops,'/home/ilutz/BINDERS_RL/2_RL/binned_loops_no0')

# loading binned loops from .pkl file
binned_loops = load_obj('/home/ilutz/BINDERS_RL/2_RL/binned_loops_no0')

build_mesh = '../build_vol.obj'


# # wnum_hash is slow to compute -- can save it as .pkl and just load to save time
# # buffer distance prevents building directly on surface -- set to 0 currently
if len(glob.glob('../wnum_hash*')) == 0: # only bother making it if it's not there yet
    wnum_hash = voxelize_mesh(build_mesh,0) # buff dist = 0
    save_obj(wnum_hash,'../wnum_hash')
else:
    # loading wnum_hash from .pkl file
    # for sequence buffer of 0:
    wnum_hash = load_obj('../wnum_hash')
    
# for sequence buffer of 1:
#wnum_hash = load_obj('/home/ilutz/RL_RIF/apoe_test_auto/apoe_wnum_hash2_buff1')



def euler_to_R(phi,theta,psi): # in radians
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(phi), -np.sin(phi) ],
                    [0,         np.sin(phi), np.cos(phi)  ]
                    ])

    R_y = np.array([[np.cos(theta),    0,      np.sin(theta)  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta),   0,      np.cos(theta)  ]
                    ])

    R_z = np.array([[np.cos(psi),    -np.sin(psi),    0],
                    [np.sin(psi),    np.cos(psi),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R



def random_init_build(runs):

    final_run = False
    
    while not final_run:
        
        # initialize with random origin, then pick random start
        g.__init__(ori_1res, wnum_hash)
        start = False  

        while not start:

            # pick a random translation
            trans = [random.random()*(maxs[0]-mins[0])+mins[0],
                     random.random()*(maxs[1]-mins[1])+mins[1],
                     random.random()*(maxs[2]-mins[2])+mins[2]]

            # add a random rotation
            rot = euler_to_R(random.random()*2*math.pi-math.pi,
                             random.random()*2*math.pi-math.pi,
                             random.random()*2*math.pi-math.pi)

            # make xform from translation, rotation
            pre = np.concatenate((rot,np.array([[0,0,0]])), axis=0)
            rand_xform = np.concatenate((pre,np.array([[trans[0]],[trans[1]],[trans[2]],[1]])), axis=1)

            # generate potential start with xform
            pot_start = nu.xform_npose(rand_xform, ori_1res)

            if check_hash(pot_start,g.wnum_hash):

                # add reasonable short extension to check too
                ext = extend_helix(pot_start, 'C', 12) # reasonable? 16 before, --> 12 for end helix fix
                if check_hash(ext, g.wnum_hash):

                    # if rif_score of this starting point is above threshold
                    tr_rscore = calc_rif_score(np.append(pot_start,ext,0))[-1]

                    if tr_rscore < -24:

                        tr_sasa,_ = sasa_score(np.append(pot_start,ext,0), target_face_centers, target_face_areas, close_to_target)

                        if tr_sasa > 700:

                            # then go ahead with this starting point for a run
                            start = True

        g.stub = pot_start

        # test with x iterations to make sure builds are working
        test_iters = 5000
        test_builder(num_runs = test_iters)

        with open('trial_starts','a') as file:
            file.write('1 \n')

        # if builds are getting longer than x residues and scoring well enough, do full number of iterations
        if max([len(x)/5 for x in g.builds]) > 40 and max(g.rif_scores) > 40:
                   
            final_run = True

            with open('final_starts','a') as file:
                file.write('1 \n')

            test_builder(num_runs = runs)

            max_dump_per_trial = 50
            dump_ct = 0
            best_builds = {}
            for ind,i in enumerate(g.builds):
                if g.overall_scores[ind] > 0.01:
                    best_builds[g.overall_scores[ind]] = (i, g.build_sstructs[ind])

            sorted_best_builds = [x[1] for x in sorted(best_builds.items(), key = lambda kv: kv[0])]
            sorted_best_builds.reverse()

            for bind,pre_bd in enumerate(sorted_best_builds):
                if dump_ct < max_dump_per_trial:

                    bd = pre_bd[0]
                    bd_ss = pre_bd[1]

                    pc_scn, motif_hits, rif_score, int_sasa, worst_hel, core_pred, score_weight = score_build(bd,input_ss=bd_ss)

                    if pc_scn >= 0.18 and motif_hits >= 0.9 and rif_score >= 25 and int_sasa >= 750 and worst_hel >= 2.0 and core_pred >= 50:

                        nu.dump_npdb(bd,f'outputs/buildv10_{str(pc_scn)[:6]}_{str(motif_hits)[:4]}_{str(rif_score)[:5]}_{str(int_sasa)[:6]}_{str(worst_hel)[:5]}_{str(core_pred)[:5]}_{str(score_weight)[:6]}.pdb')
                        dump_ct += 1

                    # for saving trees for policy network -- only do this once settled on method
                    #fnum = len(glob.glob('/net/scratch/ilutz/forest/tree*.pkl'))
                    #save_obj(g.build_graph, f'/net/scratch/ilutz/forest/tree_{fnum+1}')



tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)

# first x residues of ideal alpha helix with center of helical axis at origin, CA1 at y=0 z=0
ori_1res = tt[10:15].reshape(25,4)

g = tree_builder(ori_1res, wnum_hash)

maxs = [max(tg_atoms[:][0])+15,max(tg_atoms[:][1])+15,max(tg_atoms[:][2])+15]
mins = [min(tg_atoms[:][0])-15,min(tg_atoms[:][1])-15,min(tg_atoms[:][2])-15]

for _ in range(100):
    random_init_build(50000)







