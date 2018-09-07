# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 12:16:23 2018

@author: jdshin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:52:28 2018

@author: jdshin
"""

#Script for looking at representation shift in CA1 and PFC over learning

import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats import sem
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

linmat = scipy.io.loadmat('H:\Single_Day_WTrack\JS15_direct\JS15linfields01.mat', 
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
for x in ca1_linfields:
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
        for sp in ca1_linfields:
            if sp['Tetrode'] == c[0] and sp['Cell'] == c[1]:
                cellinep_count.append(sp)
        if len(cellinep_count) == 8: #dirty way to specify number of epochs to match over
            allep_cellidx.append(c)
            
###############################################################################            

#peak and center of mass (COM) shift analysis over learning
#CA1
template_ep = [1, 3, 5, 7, 9, 11, 13]
epoch = [3, 5, 7, 9, 11, 13, 15]

ca1_peakshift = {'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}

for temp, ep in zip(template_ep, epoch):
    #ca1shift.append({str(ep):[]}
    s = 0
    for cell in ca1_linfields:
        for cell2 in ca1_linfields:
            if cell['Tetrode'] == cell2['Tetrode'] and cell['Cell'] == cell2['Cell'] and cell['Epoch'] == temp and cell2['Epoch'] == ep:
                pshift_all = [[],[],[],[]]
                for i, traj in enumerate(sorted(cell)[4:]): #iterate over all trajectories
                    peak1 = np.nanmax(cell[traj][:,4])                                        
                    peak2 = np.nanmax(cell2[traj][:,4])
                    if peak1 > 0 and peak2 > 0 and sum(cell[traj][:,2]) > 50 and sum(cell2[traj][:,2]) > 50: #peak is nonzero and there are more than 30 spikes
                       pshift = {traj:[]} 
                       pshift[traj] = []
                       
                       peak1idx = np.where(cell[traj][:,4] == peak1)
                       peak1loc = cell[traj][peak1idx, 0]                                           
                                             
                       peak2idx = np.where(cell2[traj][:,4] == peak2)
                       peak2loc = cell2[traj][peak2idx, 0]                                                                                         
                       peakshift = np.abs(peak1loc - peak2loc)   
                                         
                       pshift[traj] = peakshift 
                       pshift_all[i] = pshift
                       del pshift
                       
                    else:
                       pshift = {traj:[]} 
                       pshift[traj] = []
                       pshift[traj] = np.nan 
                       pshift_all[i] = pshift
                       del pshift
                       

                ca1_peakshift['ep' + str(ep)].append({str(cell['Tetrode']) + ' ' + str(cell['Cell']):[]})
                ca1_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].extend(pshift_all)
                ca1_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].append({'Tetrode':cell['Tetrode']})
                ca1_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].append({'Cell':cell['Cell']})
                s = s + 1                  
                
#DIRTY - get avg field shift for each epoch for all trajectories
epochs = [3, 5, 7, 9, 11, 13, 15]
shiftvalues = {'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
trajectories = ['inleft', 'inright', 'outleft', 'outright']
for ep in epochs:
    epshiftvals = []
    epstr = 'ep' + str(ep)
    for cellidx in allep_cellidx:
        for cell in ca1_peakshift:
            if cell == epstr:
                for i, p in enumerate(ca1_peakshift['ep' + str(ep)]):
                    for k in p:
                        if len(k) == 3:
                            if cellidx[0] == int(k[0]) and cellidx[1] == int(k[2]):
                                for t, val in ca1_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)
                                                
                        elif len(k) == 4: 
                            if cellidx[0] == int(k[0:2]) and cellidx[1] == int(k[3]):
                                for t, val in ca1_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)
                                             
                        elif len(k) > 4:
                            if cellidx[0] == int(k[0:2]) and cellidx[1] == int(k[3:5]):
                                for t, val in ca1_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)                        
                            
                                    
    shiftvalues['ep' + str(ep)] = epshiftvals                            
                            
                
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
allep_cellidx_pfc = []
for ep in eparray: 
    for c in cellidx:
        cellinep_count = []
        for sp in pfc_linfields:
            if sp['Tetrode'] == c[0] and sp['Cell'] == c[1]:
                cellinep_count.append(sp)
        if len(cellinep_count) == 8: #dirty way to specify number of epochs to match over
            allep_cellidx_pfc.append(c)
            
            
#PFC peak shift
    
template_ep = [1, 3, 5, 7, 9, 11, 13]
epoch = [3, 5, 7, 9, 11, 13, 15]

pfc_peakshift = {'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}

for temp, ep in zip(template_ep, epoch):
    #ca1shift.append({str(ep):[]}
    s = 0
    for cell in pfc_linfields:
        for cell2 in pfc_linfields:
            if cell['Tetrode'] == cell2['Tetrode'] and cell['Cell'] == cell2['Cell'] and cell['Epoch'] == temp and cell2['Epoch'] == ep:
                pshift_all = [[],[],[],[]]
                for i, traj in enumerate(sorted(cell)[4:]): #iterate over all trajectories
                    peak1 = np.nanmax(cell[traj][:,4])                                        
                    peak2 = np.nanmax(cell2[traj][:,4])
                    if peak1 > 0 and peak2 > 0 and sum(cell[traj][:,2]) > 50 and sum(cell2[traj][:,2]) > 50: #peak is nonzero and there are more than 30 spikes
                       pshift = {traj:[]} 
                       pshift[traj] = []
                       
                       peak1idx = np.where(cell[traj][:,4] == peak1)
                       peak1loc = cell[traj][peak1idx, 0]                                           
                                             
                       peak2idx = np.where(cell2[traj][:,4] == peak2)
                       peak2loc = cell2[traj][peak2idx, 0]                                                                                         
                       peakshift = np.abs(peak1loc - peak2loc)   
                                         
                       pshift[traj] = peakshift 
                       pshift_all[i] = pshift
                       del pshift
                       
                    else:
                       pshift = {traj:[]} 
                       pshift[traj] = []
                       pshift[traj] = np.nan 
                       pshift_all[i] = pshift
                       del pshift
                       

                pfc_peakshift['ep' + str(ep)].append({str(cell['Tetrode']) + ' ' + str(cell['Cell']):[]})
                pfc_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].extend(pshift_all)
                pfc_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].append({'Tetrode':cell['Tetrode']})
                pfc_peakshift['ep' + str(ep)][s][str(cell['Tetrode']) + ' ' + str(cell['Cell'])].append({'Cell':cell['Cell']})
                s = s + 1                  
                
#DIRTY - get avg field shift for each epoch for all trajectories
epochs = [3, 5, 7, 9, 11, 13, 15]
pfc_shiftvalues = {'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
trajectories = ['inleft', 'inright', 'outleft', 'outright']
for ep in epochs:
    epshiftvals = []
    epstr = 'ep' + str(ep)
    for cellidx in allep_cellidx_pfc:
        for cell in pfc_peakshift:
            if cell == epstr:
                for i, p in enumerate(pfc_peakshift['ep' + str(ep)]):
                    for k in p:
                        if len(k) == 3:
                            if cellidx[0] == int(k[0]) and cellidx[1] == int(k[2]):
                                for t, val in pfc_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)
                                                
                        elif len(k) == 4: 
                            if cellidx[0] == int(k[0:2]) and cellidx[1] == int(k[3]):
                                for t, val in pfc_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)
                                             
                        elif len(k) > 4:
                            if cellidx[0] == int(k[0:2]) and cellidx[1] == int(k[3:5]):
                                for t, val in pfc_peakshift['ep' + str(ep)][i].items():
                                    if val[4]['Tetrode'] == cellidx[0] and val[5]['Cell'] == cellidx[1]:
                                        for index, traj in zip(range(4), trajectories):
                                            if np.isnan(val[index][traj]):
                                                continue
                                            else:
                                                shiftval = int(val[index][traj])
                                                epshiftvals.append(shiftval)                        
                            
                                    
    pfc_shiftvalues['ep' + str(ep)] = epshiftvals 
        
shift_mean = []
shift_mean.append(np.mean(shiftvalues['ep3']))
shift_mean.append(np.mean(shiftvalues['ep5']))
shift_mean.append(np.mean(shiftvalues['ep7']))
shift_mean.append(np.mean(shiftvalues['ep9']))
shift_mean.append(np.mean(shiftvalues['ep11']))
shift_mean.append(np.mean(shiftvalues['ep13']))
shift_mean.append(np.mean(shiftvalues['ep15']))
x_ax = ['ep1toep3','ep3toep5','ep5toep7','ep7toep9','ep9toep11','ep11toep13','ep13toep15']
plt.plot(x_ax,shift_mean)















