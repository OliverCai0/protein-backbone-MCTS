#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import pymesh

#TEMP
#sys.path.append('/net/scratch/ilutz/TARGETS/APOE_1/2_rl/batch8/')

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

#sys.path.append("/home/bcov/sc/random/npose")
import npose_util as nu
import motif_stuff2

import subprocess
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree


zero_ih = nu.npose_from_file('/home/ilutz/BINDERS_RL/2_RL/zero_ih.pdb')


import math

from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import getpy
import xbin
import h5py

#sys.path.append("/home/bcov/sc/random/npose")
import voxel_array
import npose_util as nu

from importlib import reload
reload(nu)

#MOD
from sasa_util_eff import sasa_score
#MOD2
from sasa_util_eff import get_target_surface


# In[4]:


# TEMP
#os.chdir('/net/scratch/ilutz/TARGETS/APOE_1/2_rl/batch8/')


# In[5]:


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

B_SASA_TG = min(sum(target_face_areas)*2 - 100, 1800)




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

#             my_aa = rif_rot_seq[irot]
#             print(" seqpos:%3i %s score:   0.00 irot:%3i 1-body:   0.00 rif score: %6.2f rif rescore:  00.00 sats: -1  -1"%(
#                 seqpos, my_aa, irot, score ))

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
    

def align_loop(build,loop):
    # returns loop aligned to end of build, overlapping
    tpose1 = nu.tpose_from_npose(loop)
    tpose2 = nu.tpose_from_npose(build)

    itpose1 = np.linalg.inv(tpose1)
    
    # loop[0] to build[-1]
    xform = tpose2[-1] @ itpose1[0]

    aligned_npose1 = nu.xform_npose( xform, loop )
    
    return aligned_npose1[5:]



def extend_helix(build, res):
    
    # TODO: fix up for longer extensions (> len(zero_ih), ~20)
    ext = align_loop(build,zero_ih)
    
    return ext[:(res*5)]


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
            
    binned_loops2[int(len(binned_loops2))] = [zero_ih[:5]]
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
        
        if key not in wnum_hash: # wnum_hash must be trimmed where every entry == True
            #print(key)
            return False
        
        #if not wnum_hash[key]: # second check -- get rid of this if wnum_hash is trimmed
            #return False
            
    return True

def check_clash(build_set, query_set, threshold, diameter_threshold = 999):
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
            axa[-(i+1)][j] = threshold + 0.1
    
    if np.min(axa) < threshold: # clash condition
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


# In[46]:


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
    if pc_scn > 0.1 and motif_score > 0.9: # shitty fix for stupid atomic_depth bug, should also speed up a bit
        sasa_scn, sasa_time = sasa_score(input_pose, target_face_centers, target_face_areas, close_to_target)
        int_sasa = sasa_scn
    else:
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
            prev = i

        a_whel = 0.5
        b_whel = 2.1
        m_whel = 1.1
        whel_penalty = m_whel/(1+exp(-a_whel*10*(worst_helix - b_whel)))

        score_wt_pre = score_wt_pre * whel_penalty

    if score_wt_pre > 0.25:
        a_final,b_final,m_final = .003, 200, 1
        score_wt = (m_final/(1+exp(-a_final*10*(score_wt_pre - b_final))))*8000
    else:
        score_wt = 0
    
    return pc_scn, motif_score, rif_score, int_sasa, score_wt


# In[ ]:


def test_builder(num_runs = 80000001):
    ct = 0
    while ct < num_runs:
        

#         if ct % 2500 == 0 and len(g.builds) > 0:
#             series = [ind for ind,_ in enumerate(g.builds)]
#             # print('lengths')
#             # sns.scatterplot(x = series, y = [len(x) for x in g.builds])
#             # plt.show()
#             # print('build times')
#             # sns.scatterplot(x = series, y = g.build_times)
#             # plt.show()

#             avg_lengths = []
#             best_lengths = []
#             avg_bts = []
#             avg_series = []
            
#             avg_pc_scn = []
#             best_pc_scn = []
            
#             avg_mots = []
#             best_mots = []
            
#             avg_rifs = []
#             best_rifs = []
            
#             avg_sasa = []
#             best_sasa = []
            
#             avg_overall = []
#             best_overall = []
            
#             if len(g.builds) < 10000:
#                 bin_size = 50
#             elif len(g.builds) < 25000:
#                 bin_size = 250
#             elif len(g.builds) < 50000:
#                 bin_size = 500
#             elif len(g.builds) < 100000:
#                 bin_size = 1000
#             else:
#                 bin_size = 5000

#             for i in series:
#                 if i % bin_size == 0 and i != 0:
#                     avg_series.append(i)
#                     avg_lengths.append(np.mean([len(x)/5 for x in g.builds[i-bin_size:i]]))
#                     best_lengths.append(max([len(x)/5 for x in g.builds[i-bin_size:i]]))
#                     avg_bts.append(np.mean(g.build_times[i-bin_size:i]))
                    
#                     avg_pc_scn.append(np.mean(g.pc_scns[i-bin_size:i]))
#                     best_pc_scn.append(max(g.pc_scns[i-bin_size:i]))
                    
#                     avg_mots.append(np.mean(g.motif_scores[i-bin_size:i]))
#                     best_mots.append(max(g.motif_scores[i-bin_size:i]))
                    
#                     avg_rifs.append(np.mean(g.rif_scores[i-bin_size:i]))
#                     best_rifs.append(max(g.rif_scores[i-bin_size:i]))
                    
#                     avg_sasa.append(np.mean(g.sasas[i-bin_size:i]))
#                     best_sasa.append(max(g.sasas[i-bin_size:i]))
                    
#                     try:
#                         avg_overall.append(np.mean(g.overall_scores[i-bin_size:i]))
#                         best_overall.append(max(g.overall_scores[i-bin_size:i]))
#                     except:
#                         pass

#             print('avg lengths, best lengths')
#             sns.scatterplot(x = avg_series, y = avg_lengths)
#             sns.scatterplot(x = avg_series, y = best_lengths)
#             plt.show()
#             #plt.savefig('avglengths_bestlengths.png')
#             #plt.clf()
#             print('avg build times')
#             sns.scatterplot(x = avg_series, y = avg_bts)
#             plt.show()
#             #plt.savefig('avg_build_times.png')
#             #plt.clf()
#             print('avg pc scn')#, best pc scn')
#             sns.scatterplot(x = avg_series, y = avg_pc_scn)
#             plt.show()
#             #plt.savefig('avg_pc_scn.png')
#             #plt.clf()
#             print('best pc scn')
#             sns.scatterplot(x = avg_series, y = best_pc_scn)
#             plt.show()
#             #plt.savefig('best_pc_scn.png')
#             #plt.clf()
#             print('avg motif score')#, best motif score')
#             sns.scatterplot(x = avg_series, y = avg_mots)
#             plt.show()
#             #sns.scatterplot(x = avg_series, y = best_mots)
#             #plt.savefig('avg_motif_score.png')
#             #plt.clf()
#             print('avg rif score')
#             sns.scatterplot(x = avg_series, y = avg_rifs)
#             plt.show()
#             #plt.savefig('avg_rif_score.png')
#             #plt.clf()
#             print('best rif score')
#             sns.scatterplot(x = avg_series, y = best_rifs)
#             plt.show()
#             #plt.savefig('best_rif_score.png')
#             #plt.clf()
#             print('avg sasa')
#             sns.scatterplot(x = avg_series, y = avg_sasa)
#             plt.show()
#             #plt.savefig('avg_sasa.png')
#             #plt.clf()
#             print('best sasa')
#             sns.scatterplot(x = avg_series, y = best_sasa)
#             plt.show()
#             #plt.savefig('best_sasa.png')
#             #plt.clf()
            
#             try:
#                 print('avg overall')#, best pc scn')
#                 sns.scatterplot(x = avg_series, y = avg_overall)
#                 plt.show()
#                 #plt.savefig('avg_overall.png')
#                 #plt.clf()
#                 print('best overall')
#                 sns.scatterplot(x = avg_series, y = best_overall)
#                 plt.show()
#                 #plt.savefig('best_overall.png')
#                 #plt.clf()
#             except:
#                 pass
            
#             print('#############################################################')


        
        g.test_build()
        
        ct += 1


# In[175]:


class tree_mesh_builder():
    def __init__(self, volume_mesh, stub, wnum_hash=None):
        
        if wnum_hash is None:
            # mesh
            self.volume_mesh = volume_mesh
            print('Generating wnum_hash...')
            self.wnum_hash = voxelize_mesh(self.volume_mesh,1) # buffer dist (add as variable later)
            print(f'wnum_hash length: {len(self.wnum_hash)}')
        else:
            self.wnum_hash = wnum_hash
        
        # starting point, build data structures
        self.stub = stub
        self.build_graph = [{},[]]
        self.builds = []
        self.build_paths = []
        
        self.pc_scns = []
        self.motif_scores = []
        self.rif_scores = []
        self.sasas = []
        self.overall_scores = []
        
        # params
        self.max_ext_length = 25
        self.min_ext_length = 6 # loops have partial helices on them, 4*2 loops generally
        self.max_residues = 110 # usually just max out
        self.choice_limit = 12 # for no choice exts/loop bins, limit by tree height
        self.min_score_residues = int(len(stub)/5) + 50 # score at this length - higher is faster
        self.clash_threshold = 2.85 # 2.75 best? or too close
        self.diam_threshold = 45 # plateaus build lengths
        
        # weight params
        self.weight_factor = 1 # normally +1 per success, this can weight up or down
        self.second_ord_wf = 0 # add this to upweight downstream additions even more
        self.starting_wf = 10
        self.upweight_cutoff = 0.4 # how dominant choice can be relative to all other choices
        
        # tries before failing
        self.ext_try_limit = 3 # 2 for unconstrained (but w/o variable trials), 1 good for new
        self.loop_try_limit = 3 # 25 for unconstrained (but w/o variable trials), 1 good for new
        self.loop_try_initial = 2

        #new sampling arrays, starting at zero, need to add min_length to extension try
        self.extension_sample = [i for i in range(self.max_ext_length+1)]
        self.loop_sample = list(binned_loops.keys())
        
        #now each build graph with have a weights adjustment list [], initialized at zero
        #make separate graphs which are paired at each index, will need to alternate between the two

        #first list is graph, list for weights
        temp_wts = [0 for x in range(self.min_ext_length)] +                    [self.starting_wf for x in range(self.min_ext_length,self.max_ext_length+1)]
        self.build_graph = [{},temp_wts]
        
        # debug
        self.build_times = []
        
        
        
    def downweight(self, path, choice):
        
        if not self.downweight_on:
            return
        
        t1 = self.call_graph_path(path)[1]
        if t1[choice] > 1:
            t1[choice] -= 1 
    
    

    def upweight(self, path, factor=0.1):
        
        for path_index,value in enumerate(path):

            if (path_index+1)%2 == 0:
                t1 = self.call_graph_path(path[:path_index-1])[1] #access build path weights one level at a time
                
                uw_cutoff = min(self.upweight_cutoff + (0.025*(path_index-1)/2),0.6)
                
                if (t1[int(value)]+int(factor*len(t1))) < (sum(t1)*uw_cutoff):
                    t1[int(value)] += int(factor*len(t1))
                else:
                    t1[int(value)] = (sum(t1)*uw_cutoff)
                    
    
    def call_graph_path(self, path):
        return reduce(operator.getitem, path, self.build_graph)
    
   
    def test_build(self):

        start = time.time()
        stub = self.stub.copy()
        
        build_path = [] #extension and loop bin list
        build_index = [] #extension and loop indices list
        current_length = round(len(stub)/5)
        build = stub.copy()
        build_ss = 'H'*current_length

        while current_length < self.max_residues and len(build_path) < (self.choice_limit*2):

            # EXTENSION FIRST #####################################################################
            valid_ext = False
            tries = 0
            while not valid_ext:
                
                #NEW addition, call_graph_path(build_path)[1] return weight list at this point in build path
                #index at zero since this must return a list
                ext_length = random.choices(self.extension_sample,weights=self.call_graph_path(build_path)[1],k=1)[0]
                #add min length to change from index to sample
                    
                trial_ext = extend_helix(build,ext_length)

                # check hash, clash violations -- if it works, add it
                if check_hash(trial_ext, self.wnum_hash):
                    
                    if check_clash(build,trial_ext,self.clash_threshold,diameter_threshold=self.diam_threshold):
                        valid_ext = True
        
                tries += 1
                # if number of tries exceeds limit and no valid extension found
                if (tries >= (self.ext_try_limit*((len(build_path)/2)+1))+self.loop_try_initial) and (not valid_ext):
                    
                    # to avoid ending on a loop fragment -- don't keep loops without extensions after
                    if len(build_path) > 3 and len(new_loop) > 0:
                        build = build[:-len(new_loop)]
                    
                    self.builds.append(build)
                    self.build_paths.append(build_path)

                    if current_length >= self.min_score_residues: 

                        pc_scn, motif_hits, rif_score, int_sasa, score_weight = score_build(build, input_ss=build_ss)

                        if score_weight > 0:
                            self.upweight(build_path,factor=score_weight)


                        self.pc_scns.append(pc_scn)
                        self.motif_scores.append(motif_hits)
                        self.rif_scores.append(rif_score)
                        self.sasas.append(int_sasa)
                        self.overall_scores.append(score_weight)
                    else:
                        self.pc_scns.append(0)
                        self.motif_scores.append(0)
                        self.rif_scores.append(0)
                        self.sasas.append(0)
                        self.overall_scores.append(0)
                        
                    end = time.time()
                    self.build_times.append(end-start)
                    
                    return build
                
            
            #make new graph if not already present, loop addition is next so intialize loop weights at 0
            #hack into 
            if ext_length not in self.call_graph_path(build_path)[0]:
                if (len(build_path)+2) < (5*2): # if < 3 helices, don't allow zero choice (wt to 0)
                    temp_wts = [self.starting_wf for x in range(len(self.loop_sample))]
                    temp_wts[-1] = 0
                    t1 = self.call_graph_path(build_path)[0][ext_length] = [{},temp_wts]
                else: # allow zero choice
                    t1 = self.call_graph_path(build_path)[0][ext_length] = [{},[self.starting_wf for x in range(len(self.loop_sample))]]


            # add path, add extra zero in order to reference the graph list, not the weights list each time
            build_path.append(0)
            build_path.append(ext_length) #this length will always need min added since it acts as index

            build_index.append([ext_length])#


            # since we have had a successful addition, upweight this path according to formula
            self.upweight(build_path)

            # add extension if it passes
            build = np.append(build,trial_ext,0)

            build_ss += round(len(trial_ext)/5)*'H'

            current_length = len(build)/5
            self.loop_first = False

        
            # LOOP SECOND ##################################################################### 
            
            valid_loop = False
            tries = 0
            
            while not valid_loop:
                
        
                #weighted sample for a loop bin
                trial_loop_bin = random.choices(self.loop_sample,weights=self.call_graph_path(build_path)[1],k=1)[0]
                #randomly pick a loop from the bin
                trial_loop_index = int(random.random()*len(binned_loops[trial_loop_bin]))
                trial_loop = binned_loops[trial_loop_bin][trial_loop_index]
    
                new_loop = align_loop(build,trial_loop)

                
                if check_hash(new_loop,self.wnum_hash):
                    
                    if check_clash(build,new_loop,self.clash_threshold,diameter_threshold=self.diam_threshold):
                        valid_loop = True
                    
                tries += 1
                
                # tries exceeded, or if picked no loop
                if ((tries >= (self.loop_try_limit*((len(build_path)/2)+1))+self.loop_try_initial) and (not valid_loop)) or (len(new_loop) == 0):

                    
                    self.builds.append(build)
                    self.build_paths.append(build_path)

                    if current_length >= self.min_score_residues: 

                        pc_scn, motif_hits, rif_score, int_sasa, score_weight = score_build(build, input_ss=build_ss)

                        if score_weight > 0:
                            self.upweight(build_path,factor=score_weight)

                        self.pc_scns.append(pc_scn)
                        self.motif_scores.append(motif_hits)
                        self.rif_scores.append(rif_score)
                        self.sasas.append(int_sasa)
                        self.overall_scores.append(score_weight)

                    else:
                        self.pc_scns.append(0)
                        self.motif_scores.append(0)
                        self.rif_scores.append(0)
                        self.sasas.append(0)
                        self.overall_scores.append(0)
                        
                    
                    end = time.time()
                    self.build_times.append(end-start)
                    
                    return build
                
            
                
            if trial_loop_bin not in self.call_graph_path(build_path)[0]:
                temp_wts = [0 for x in range(self.min_ext_length)] +                    [self.starting_wf for x in range(self.min_ext_length,self.max_ext_length+1)]
                t1 = self.call_graph_path(build_path)[0][trial_loop_bin] = [{},temp_wts]

            
            # add path
            build_path.append(0)
            build_path.append(trial_loop_bin)
            
            build_index.append([trial_loop_bin,trial_loop_index])

            self.upweight(build_path)
            
            # add loop if it passes
            build = np.append(build,new_loop,0)

            build_ss += round(len(new_loop)/5)*'L'

            current_length = len(build)/5
                     
                    
        # if max residues is reached

        # to avoid ending on a loop fragment -- don't keep loops without extensions after
        if len(build_path) % 4 == 0 and len(new_loop) > 0:
            build = build[:-len(new_loop)]

        self.builds.append(build)
        self.build_paths.append(build_path)
        if current_length >= self.min_score_residues: 

            pc_scn, motif_hits, rif_score, int_sasa, score_weight = score_build(build, input_ss=build_ss)

            if score_weight > 0:
                self.upweight(build_path,factor=score_weight)

            self.pc_scns.append(pc_scn)
            self.motif_scores.append(motif_hits)
            self.rif_scores.append(rif_score)
            self.sasas.append(int_sasa)
            self.overall_scores.append(score_weight)

        else:
            self.pc_scns.append(0)
            self.motif_scores.append(0)
            self.rif_scores.append(0)
            self.sasas.append(0)
            self.overall_scores.append(0)
            
        end = time.time()
        self.build_times.append(end-start)
        
        return build


# In[27]:


# # make loop bins
# binned_loops = kClusBin(clusts1=1,clusts2=32,clusts3=16)

# # save loop bins, for easier use later
# save_obj(binned_loops,'apoe_test/binned_loops1')


# In[360]:


# loading binned loops from .pkl file
binned_loops = load_obj('/home/ilutz/BINDERS_RL/2_RL/binned_loops1')


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


# In[176]:


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


# In[424]:


def random_init_build(trials, runs_per_trial):

    trial_ct = 0
    
    while trial_ct < trials:
        
        # initialize (outside fxn currently) with just random origin, then pick random start in here
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
                ext = extend_helix(pot_start, 16) # reasonable?
                if check_hash(ext, g.wnum_hash):
                    
                    # if rif_score of this starting point is above threshold
                    tr_rscore = calc_rif_score(np.append(pot_start,ext,0))[-1]

                    if tr_rscore < -24:

                        tr_sasa,_ = sasa_score(np.append(pot_start,ext,0), target_face_centers, target_face_areas, close_to_target)

                        if tr_sasa > 700:
                            
                            # then go ahead with this starting point for a run
                            start = True

        
        #print(f'trial {trial+1}')
        g.stub = pot_start
        # reset build graph each time, but keep builds list
        g.build_graph = [{},[g.starting_wf for x in range(len(g.extension_sample))]]
        g.builds = []
        g.build_paths = []
        g.build_times = []
        g.pc_scns = []
        g.motif_scores = []
        g.rif_scores = []
        g.sasas = []
        g.overall_scores = []
        
        # test with x iterations to make sure builds are working
        test_iters = 5000
        test_builder(num_runs = test_iters)
        # if builds are getting longer than x residues and scoring well enough, do full number of iterations
        if max([len(x)/5 for x in g.builds[-test_iters:]]) > 40 and max(g.rif_scores) > 40:

            trial_ct += 1
            nu.dump_npdb(np.append(pot_start,ext,0),f'starts/start_{tr_rscore}_{tr_sasa}.pdb')
            test_builder(num_runs = runs_per_trial)
            
            max_dump_per_trial = 50
            dump_ct = 0
            best_builds = {}
            
            for ind,i in enumerate(g.builds):
                if ind > (len(g.builds) - (runs_per_trial+250) ):
                    if g.overall_scores[ind] > 50:
                        best_builds[g.overall_scores[ind]] = i
            
            sorted_best_builds = [x[1] for x in sorted(best_builds.items(), key = lambda kv: kv[0])]
            sorted_best_builds.reverse()
            
            for bind,bd in enumerate(sorted_best_builds):
                if dump_ct < max_dump_per_trial:
                    
                    pc_scn, motif_hits, rif_score, int_sasa, score_weight = score_build(bd)
                    # could set int_sasa threshold higher -- check later
                    if pc_scn >= 0.18 and motif_hits >= 0.9 and rif_score >= 25 and int_sasa >= 750:
                        
                        nu.dump_npdb(bd,f'outputs/build2_{str(pc_scn)[:8]}_{str(motif_hits)[:8]}_{str(rif_score)[:8]}_{str(int_sasa)[:8]}_{str(score_weight)[:8]}.pdb')
                        dump_ct += 1
            
            # for saving trees for policy network -- only do this once settled on method
            #fnum = len(glob.glob('/net/scratch/ilutz/forest/tree*.pkl'))
            #save_obj(g.build_graph, f'/net/scratch/ilutz/forest/tree_{fnum+1}')


# In[140]:


tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)

# first 2 residues of ideal alpha helix with center of helical axis at origin, CA1 at y=0 z=0
ori_1res = tt[10:12].reshape(10,4) # using just one residue didn't work, fix later?

g = tree_mesh_builder('precalculated wnum_hash', ori_1res, wnum_hash=wnum_hash)



maxs = [max(tg_atoms[:][0])+15,max(tg_atoms[:][1])+15,max(tg_atoms[:][2])+15]
mins = [min(tg_atoms[:][0])-15,min(tg_atoms[:][1])-15,min(tg_atoms[:][2])-15]


random_init_build(1,50000)

