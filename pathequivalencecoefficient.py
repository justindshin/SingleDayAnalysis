# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:02:04 2018

@author: jdshin
"""

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


#Import linfields from .mat format

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

#ca1tet = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25] #JS15
pfctet = [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31]  #JS15
#pfctet = [0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 26, 28, 29, 30, 31]; #ER1
#pfctet = [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31] #JS17
     
pfc_linfields = []
for tet2 in pfctet:
    for t, field in enumerate(linfields):
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

#Match cell idx for cells in all epochs
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

#get bad positions (nan in spatial bin for any of the trajectories)        
pfc_linfields_nonan = []
for p in pfc_linfields:
    isnan = []
    pp = p
    inleft = p['inleft'][:,4]
    inleftnanidx = np.argwhere(np.isnan(inleft))
    inright = p['inright'][:,4]
    inrightnanidx = np.argwhere(np.isnan(inright))
    outleft = p['outleft'][:,4]
    outleftnanidx = np.argwhere(np.isnan(outleft))
    outright = p['outright'][:,4]
    outrightnanidx = np.argwhere(np.isnan(outright))
    
    isnan = inleftnanidx
    isnan = np.concatenate((isnan, inrightnanidx),axis=0)
    isnan = np.concatenate((isnan, outleftnanidx),axis=0)
    isnan = np.concatenate((isnan, outrightnanidx),axis=0)
    isnanallidx = isnan
    
    isnanallidx = np.sort(isnanallidx)
   
    pp['inleft'] = np.delete(inleft, isnanallidx)
    pp['inright'] = np.delete(inright, isnanallidx)
    pp['outleft'] = np.delete(outleft, isnanallidx)
    pp['outright'] = np.delete(outright, isnanallidx) 
    
    pfc_linfields_nonan.append({})
    pfc_linfields_nonan[-1] = pp
    del isnan
    del isnanallidx
    del pp
    del p
###############################################################################    
    
#Interpolation for trajectories of differing lengths
fixed_pfc_linfields = []

for l in pfc_linfields_nonan:
    left_length = len(l['inleft'])
    right_length = len(l['inright']) 
    
    if left_length > right_length:
        new_length = left_length
        newr_x = np.linspace(0, len(l['inleft']), new_length)
        newr_yin = interp.interp1d(np.arange(right_length), l['inright'], kind='linear', fill_value='extrapolate')(newr_x)    
        newr_yout = interp.interp1d(np.arange(right_length), l['outright'], kind='linear', fill_value='extrapolate')(newr_x)  
        l['inright'] = newr_yin
        l['outright'] = newr_yout
        
        fixed_pfc_linfields.append({})
        fixed_pfc_linfields[-1] = l

    elif left_length < right_length:
        new_length = right_length
        newl_x = np.linspace(0, len(l['inleft']), new_length)
        newl_yin = interp.interp1d(np.arange(left_length), l['inleft'], kind='linear', fill_value='extrapolate')(newl_x)    
        newl_yout = interp.interp1d(np.arange(left_length), l['outleft'], kind='linear', fill_value='extrapolate')(newl_x) 
        l['inleft'] = newl_yin
        l['outleft'] = newl_yout
        
        fixed_pfc_linfields.append({})
        fixed_pfc_linfields[-1] = l

    elif left_length == right_length:
        fixed_pfc_linfields.append({})
        fixed_pfc_linfields[-1] = l

        continue
        
del newl_x
del newr_x
del new_length
del newr_yin
del newr_yout
del newl_yin
del newl_yout
del inleft
del inright
del outleft
del outright    
del left_length
del right_length
del inleftnanidx
del inrightnanidx
del outleftnanidx
del outrightnanidx
del l; del t
del linfields
del pfc_linfields
del pfc_linfields_nonan     
###############################################################################      
        
#Get r values for Pearson correlation
epochlist = [1, 3, 5, 7, 9, 11, 13, 15]
maxr_pec = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
pec = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
pec_pfc_shuf = {'ep1':[],'ep3':[],'ep5':[],'ep7':[],'ep9':[],'ep11':[],'ep13':[],'ep15':[]}
for ep in epochlist:
    for f in fixed_pfc_linfields:
        rvals = []
        pvals = []
        rvals_shuf = []
        pvals_shuf = []
        t = f['Tetrode']
        c = f['Cell']
        e = f['Epoch']
        if f['Epoch'] == ep:
            
            r, p = pearsonr(f['inleft'], f['inright']) #get rid of center arm
            rvals.append(r)
            pvals.append(p)
            r, p = pearsonr(f['inleft'], f['outleft'])
            rvals.append(r)
            pvals.append(p)
            r, p = pearsonr(f['inleft'], f['outright'])
            rvals.append(r)
            pvals.append(p)
            r, p = pearsonr(f['inright'], f['outleft'])
            rvals.append(r)
            pvals.append(p)
            r, p = pearsonr(f['inright'], f['outright'])
            rvals.append(r)
            pvals.append(p)
            
            r, p = pearsonr(f['outleft'], f['outright']) #get rid of center arm
            rvals.append(r)
            pvals.append(p)
            for c in allep_cellidx:
                if f['Tetrode'] == c[0] and f['Cell'] == c[1]:
                    maxr = np.nanmax(rvals)
                    minp = np.nanmin(pvals)
            
                    f['rvalues'] = rvals
                    f['pvalues'] = pvals
                    f['maxp'] = minp
                    f['maxr'] = maxr
            
                    maxr_pec['ep' + str(ep)].append(maxr)

                    rmean = np.mean(rvals)
                    #if rmean > 0: #some positive degree of path equiv
                    pec['ep' + str(ep)].append(rmean)
#                    else:
#                        continue
                    trajtoshuf = [f['inleft'], f['inright'], f['outleft'], f['outright']]
                    trajkeys = ['inleft', 'inright', 'outleft', 'outright']
                    for shuf in trajtoshuf:
                        if len(shuf) % 2 == 0:
                            mididx = len(shuf) / 2
                            abmirror = shuf[0:int(mididx + 1)]
                            abmirror = abmirror.tolist()
                            abmirror.reverse()
                            cd = shuf[int(mididx + 1):]
                            cd = cd.tolist()
                            cdab_shuf = cd + abmirror
                            if np.mean(cdab_shuf) != 0:
                                for key in trajkeys:
                                    r_shuf, p_shuf = pearsonr(f[key], cdab_shuf)
                                    rvals_shuf.append(r_shuf)
                                    pvals_shuf.append(p_shuf)
                            else:
                                continue
                        
                        else:
                            mididx = int((len(shuf) / 2) + 0.5)
                            abmirror = shuf[0:int(mididx + 1)]
                            abmirror = abmirror.tolist()
                            abmirror.reverse() 
                            cd = shuf[int(mididx + 1):]
                            cd = cd.tolist()
                            cdab_shuf = cd + abmirror
                            if np.mean(cdab_shuf) != 0:
                                for key in trajkeys:
                                    r_shuf, p_shuf = pearsonr(f[key], cdab_shuf)
                                    rvals_shuf.append(r_shuf)
                                    pvals_shuf.append(p_shuf)
                            else:
                                continue
                                
                    maxr_shuf = np.nanmax(rvals_shuf)                                                                                    
                    pec_pfc_shuf['ep' + str(ep)].append(maxr - maxr_shuf)

    










