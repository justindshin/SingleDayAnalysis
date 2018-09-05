# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:52:28 2018

@author: jdshin
"""

#Script for looking at representation shift in CA1 and PFC over learning

import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import struct
import scipy.io
import nelpy as nel
import nelpy.io
import nelpy.plotting as npl

linmat = scipy.io.loadmat('H:\Single_Day_WTrack\JS17_direct\JS17linfields01.mat', 
                       struct_as_record=False, squeeze_me=True)

def load_linfield(linmat):
    linfielddata = []

    data = linmat['linfields']
    for epidx, da in enumerate(data):
        for tetidx, te in enumerate(da): 
            if isinstance(te, np.ndarray):
                for cellidx, cell in enumerate(te):
                    if len(cell) < 20:
                        linfielddata.append({})
                        linfielddata[-1]['Epoch'] = epidx
                        neuron_idx = (tetidx, cellidx)
                        linfielddata[-1]['Tetrode'] = tetidx
                        linfielddata[-1]['Cell'] = cellidx
                        if len(cell) == 4:
                            outl = cell[0]; inl = cell[1]; outr = cell[2]; inr = cell[3] 
                            linfielddata[-1].update({'outleft':outl})
                            linfielddata[-1].update({'inleft':inl})
                            linfielddata[-1].update({'outright':outr})
                            linfielddata[-1].update({'inright':inr})
                    else:  
                         if cellidx == 0:
                            linfielddata.append({})
                            linfielddata[-1]['Epoch'] = epidx
                            neuron_idx = (tetidx, cellidx)
                            linfielddata[-1]['Tetrode'] = tetidx
                            linfielddata[-1]['Cell'] = cellidx
                            outl = te[0]; inl = te[1]; outr = te[2]; inr = te[3] 
                            linfielddata[-1].update({'outleft':outl})
                            linfielddata[-1].update({'inleft':inl})
                            linfielddata[-1].update({'outright':outr})
                            linfielddata[-1].update({'inright':inr})
                         else:
                            continue
                    
    return linfielddata

lfields = load_linfield(linmat)

#get rid of any cells that do not have linfield data

linfields = []
for dat in lfields:
    lfielddata_cell = dat
    if  len(lfielddata_cell) > 3:
        linfields.append({})
        linfields[-1] = lfielddata_cell
        del lfielddata_cell
    else:
        del lfielddata_cell
        continue

del dat
del lfields
del linmat  

###############################################################################

ca1tet = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25] #JS15
pfctet = [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31]  #JS15

epochlist = [1, 3, 5, 7, 9, 11, 13, 15]

ca1_linfields = []
for ep in epochlist:
    for tet in ca1tet:
        for t, field in enumerate(linfields):
            if field['Epoch'] == ep:
                if field['Tetrode'] == tet:
                    field['Area'] = 'CA1'
                    ca1_linfields.append({})
                    ca1_linfields[-1] = field
                    linfields[t]['Area'] = 'CA1'
                else:
                    continue

epochlist = [1, 3, 5, 7, 9, 11, 13, 15]

pfc_linfields = []
for ep in epochlist:
    for tet2 in pfctet:
       for t, field in enumerate(linfields):
            if field['Epoch'] == ep:
                if field['Tetrode'] == tet2:
                    field['Area'] = 'PFC'
                    pfc_linfields.append({})
                    pfc_linfields[-1] = field
                    linfields[t]['Area'] = 'PFC'
                else:
                    continue
        
del field
del tet2
del pfctet

###############################################################################


matchidx = []
for x in pfc_linfields:
    if x['Epoch'] == 1:
        cell = x['Cell']
        tet = x['Tetrode'] 
        matchidx.append({})
        matchidx[-1]['Cell'] = cell
        matchidx[-1]['Tetrode'] = tet

cellidx=[]        
for match in matchidx:
    c = match['Cell']
    t = match['Tetrode']
    idx = [t, c]
    cellidx.append(idx)
    cellidx.sort()
    
eparray = [1]  
allep_cellidx = []
for ep in eparray: 
    for c in cellidx:
        cellinep_count = []
        for sp in pfc_linfields:
            if sp['Tetrode'] == c[0] and sp['Cell'] == c[1]:
                cellinep_count.append(sp)
        if len(cellinep_count) == 8: #dirty way to specify number of epochs to match over
            allep_cellidx.append(c)
            
###############################################################################            

#peak and center of mass (COM) shift analysis over learning
template_ep = [1, 3, 5, 7, 9, 11, 13]
epoch = [3, 5, 7, 9, 11, 13, 15]

ca1_comshift = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
ca1_peakshift = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
ca1_bandwidthshift = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
example = linfields[3]
for temp, ep in zip(template_ep, epoch):
    #ca1shift.append({str(ep):[]}
    for cell in ca1_linfields:
        cshift_all = [[],[],[],[]]
        pshift_all = [[],[],[],[]]
        bwshift_all = [[],[],[],[]]
        for cell2 in ca1_linfields:
            for i, traj in enumerate(sorted(cell)[4:]): #iterate over all trajectories
                if cell['Tetrode'] == cell2['Tetrode'] and cell['Cell'] == cell2['Cell'] and cell['Epoch'] == temp and cell2['Epoch'] == ep:
                    peak1 = max(cell[traj][:,4])                                        
                    peak2 = max(cell2[traj][:,4])
                    if peak1 > 0 and peak2 > 0 and sum(cell[traj][:,2]) > 30 and sum(cell2[traj][:,2]) > 30: #peak is nonzero and there are more than 30 spikes
                       pshift = {traj:[]} 
                       cshift = {traj:[]}
                       pshift[traj] = []
                       peak1idx = np.where(cell[traj][:,4] == peak1)
                       peak1loc = cell[traj][peak1idx, 0]
                       thresh1 = peak1 * 0.25 #find surrounding points that are at least 25% of peak
                       thresh1idx = cell[traj][:,4] > thresh1
                       
                       
                       #need to define the boundaries of the field
                       #Calculate the average FR within the field
                       #find neareset FR value to the mean and use that index as the COM
                       
                       #field1 = 
                       
                       #com1 = np.mean(cell[traj][]) #calculate the mean for firing field: FILL IN
                       com1closest = (np.abs(cell[traj][:,4] - com1)).argmin() #need to modify to be more specific here
                       com1idx = cell[traj].index(com1closest)
                       com1loc = cell[traj][com1idx, 0]                                            
                                             
                       peak2idx = np.where(cell2[traj][:,4] == peak2)
                       peak2loc = cell2[traj][peak2idx, 0]
                       thresh2 = peak2 * 0.25
                       thresh2idx = cell2[traj][:,4] > thresh2
                       
                       #com2 = np.mean(cell2[traj][]) #calculate the mean for firing field: FILL IN
                       com2closest = (np.abs(cell2[traj][:,4] - com2)).argmin()
                       com2idx = cell2[traj].index(com2closest)
                       com2loc = cell2[traj][com2idx, 0]   
                       
                       #field2 = 
                       
                       peakshift = np.abs(peak1loc - peak2loc)
                       comshift = np.abs(com1loc - com2loc) 
                       bandwidthshift = abs(len() - len())
                       
                       pshift[traj] = peakshift 
                       cshift[traj] = comshift   
                       bwshift[traj] = bandwidthshift
                    else:
                       pshift[traj] = np.nan 
                       cshift[traj] = np.nan 
                       bwshift[traj] = np.nan
                        #put nan there
                else:
                    continue
                
        pshift_all[i] = pshift
        cshift_all[i] = cshift
        bwshift_all[i] = bwshift
                
    ca1_comshift['ep' + str(ep)] = {'Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])}
    ca1_comshift['ep' + str(ep)]['Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])] = cshift_all
    
    ca1_peakshift['ep' + str(ep)] = {'Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])}
    ca1_peakshift['ep' + str(ep)]['Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])] = pshift_all
    
    ca1_bandwidthshift['ep' + str(ep)] = {'Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])}
    ca1_bandwidthshift['ep' + str(ep)]['Tetrode Cell':str(cell['Tetrode']) + ' ' + str(cell['Cell'])] = bwshift_all
    
                

###############################################################################

#def getcomshift(linfields, 'area'):
#    for field in linfields:
        

















