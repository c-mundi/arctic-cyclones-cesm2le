#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2025
Updated Wed July 23 2025
Prepared for publication - 08 Aug 2025
figures.py

final figures for paper: Figs 1, 2
Fig S2 in supplemental materials

other figures from paper:
% figure 3: miz_storms.py
% figure 4: spatial_changes.py
% fig s1 (supplemental methods): supplemental_methods.py

@author: mundi
"""

root_path = '/Users/mundi/Desktop/cesm-code/'

SAVEFIG=False
savepath =  root_path+'processed-data/figures/'

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cmocean.cm as cmo
import xarray as xr

from scipy.stats import ttest_1samp

import pickle
from datetime import datetime, timedelta
import glob
import string
import gc, os

import sys
import cesm_functions as cf
import warnings

min_p = 986
months = [6,7,8,9]

fontsize=14

class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

#%% functions
def get_spaghetti(normalize = True):
    path1 = root_path +'cesm_census/'
    hemisphere = 'n'
    cens_name = 'census_test1_all_'
    ncadd = '_seaice'
    ens_dirs = glob.glob(path1+'out_'+hemisphere+'_1*/')
    ice_lims = [20,80]
    
    IND = 0
    miz_names = ['sia_miz_1000','sia_miz2_1000'] # original miz (15/80), RMS miz
    clim_names = ['sia_clim_miz_1000','sia_clim_miz2_1000']
    VAR = miz_names[IND]
    VAR_CLIM = clim_names[IND]
    

    TOTAL_ALL = []
    ens_frac = []
    
    ens_net = {}
    for ens_dir in ens_dirs:
        ens_mem = ens_dir[-9:-1]
        # collect all lines for summary figure
        TOTAL2 = {}
        for era in ['0','1']:
            for momo in ['6','7','8','9']:
                TOTAL2[era+'_'+momo]=[]
        table2 = {}
        
        all_ice = {}
        
        ### FRACTION OF INCREASING/DECREASING STORMS
        frac = {}
        ips = {} # compare ice area percent
        net = {} # net impact
        for era in [0,1]:
            for aa, mm in enumerate(['67', '89']):
                frac[str(era)+'_'+mm+'_incr']=0
                frac[str(era)+'_'+mm+'_decr']=0
            
                ips[str(era)+'_'+mm+'_incr']=[]
                ips[str(era)+'_'+mm+'_decr']=[]
                
                table2[str(era)+str(aa)] = []
        
                net[str(era)+'_'+mm+'_incr']=[]
                net[str(era)+'_'+mm+'_decr']=[]
            
        for era in [0,1]:
            if era==0:
                myyears = np.arange(2010, 2019+1)
            elif era==1:
                myyears = np.arange(1982, 1991+1)
            
            all_pcd67, all_pcd89 = [],[]
            total_storms=0
            count67, count89 = 0,0
            
            for year in myyears:
               for mm in months: all_ice[str(year)+'_'+str(mm)] = []
               # open census (instead of storm_ranges)
               try:
                   census_file = ens_dir+'/'+cens_name+str(year)+'.csv'
                   [startdate, enddate] = cf.readCensus(census_file, convertDT=True)[0]
               except ValueError as ve:
                   print('- Empty Census: ' + str(year)); print(ve); 
                   continue
               
               timing_grid = []
               for xx in range(0,len(startdate)):
                   timing_grid.append((startdate[xx], enddate[xx]))
               
               storm_ranges = []
               analysis_ranges = []
               for startdt, enddt in timing_grid:
                   week_ago = startdt - timedelta(days=7)
                   two_week = startdt + timedelta(days=14) # relative to start date, since different storm lengths
                   analysis_ranges.append(cf.daterange(week_ago, two_week, dt=24))
                   storm_ranges.append(cf.daterange(startdt, enddt, dt=24))   
               
               # open ice area
               ds_area = xr.open_dataset(ens_dir+'/'+ str(year) +'_area.nc')
               ice_sorter = ds_area['ice_area80'].values
               # ice_area15 = ds_area['ice_area15'].values
               box_area = ds_area['box_area'].values
               
               try:
                   ds = xr.open_dataset(ens_dir +'seaice/'+ str(year) +ncadd+'.nc')
               except:
                   print('- skip: '+ens_dir +'seaice/'+ str(year) +ncadd+'.nc')
                   continue
                
               for storm_num, strm in enumerate(storm_ranges):
                    month = int(strm[0].month)
                    # get ice area
                    ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]
                    if np.isnan(ice_frac) or np.isinf(ice_frac):
                        ice_frac=0
                        
                    if (ice_frac<np.min(ice_lims) or ice_frac>np.max(ice_lims)):
                        continue
                        
                    ### remove storms that don't interact with the ice
                    try:
                        sia = ds[VAR].values[storm_num]
                    except:
                        print('-- sia error')
                        continue
                    
                    if (len(np.unique(sia))==1 and np.unique(sia)[0] == 0) or np.isnan(np.mean(sia[0:10+1])):
                        print('no ice: '+strm[0].strftime('%Y-%m-%d')+str(np.unique(sia)))
                        # continue #!!!
                    
                    total_storms += 1
                 
                    # get time series                                              
                    sia_clim = ds[VAR_CLIM].values[storm_num]
                    ### relativize
                    ss = sia-sia_clim
                    if normalize:
                        standardized_area = (ss-ss[0])/(np.nanmax(ss)-np.nanmin(ss))
                        pcd = standardized_area
                    else:
                        pcd = ss-ss[0]
                    
                    if month in months[0:2]:
                        count67+=1
                        all_pcd67.append(pcd)
                        table2[str(era)+'0'].append(ss)
                        TOTAL2[str(era)+'_'+str(month)].append(pcd)
                        
                        if pcd[-1]>0:
                            frac[str(era)+'_67_incr']+=1
                            ips[str(era)+'_67_incr'].append(ice_frac)
                            net[str(era)+'_67_incr'].append(pcd[-1])
                        elif pcd[-1]<0:
                            frac[str(era)+'_67_decr']+=1
                            ips[str(era)+'_67_decr'].append(ice_frac)
                            net[str(era)+'_67_decr'].append(pcd[-1])
                    elif month in months[2:]:
                        count89+=1
                        all_pcd89.append(pcd)
                        table2[str(era)+'1'].append(ss)
                        TOTAL2[str(era)+'_'+str(month)].append(pcd)
                        
                        if pcd[-1]>0:
                            frac[str(era)+'_89_incr']+=1
                            ips[str(era)+'_89_incr'].append(ice_frac)
                            net[str(era)+'_89_incr'].append(pcd[-1])
                        elif pcd[-1]<0:
                            frac[str(era)+'_89_decr']+=1
                            ips[str(era)+'_89_decr'].append(ice_frac)
                            net[str(era)+'_89_decr'].append(pcd[-1])
                    else:
                        break
                    
                    all_ice[str(year)+'_'+str(month)].append(ss[-1] - ss[0]) ##### ICE AREA CHANGE?
                    
               try: ds.close()
               except: pass
           
        ens_net[ens_mem] = all_ice
           
        TOTAL_ALL.append(TOTAL2) 
       
        ens_frac.append(frac)

    return TOTAL_ALL, ens_frac, ens_net

def set_box_color(bp, color, lw=1.5):
    plt.setp(bp['boxes'], color=color, lw=lw)
    plt.setp(bp['whiskers'], color=color, lw=lw)
    plt.setp(bp['caps'], color=color, lw=lw)
    plt.setp(bp['medians'], color=color, lw=lw)

def calc_deriv(series, dt=3):
    
    deriv_series = [] #np.nan, np.nan
    for idx, ts in enumerate(series[:-dt]):
        ds = series[idx+dt] - series[idx]
        deriv_series.append(ds/dt)
        
    deriv_series.append(np.nan)
    
    return deriv_series

def era_ice():
    ice_lims = [20,80]
    all_ice_e = {}
    for era in [0,1]:
        if era==0:
            myyears = np.arange(2010, 2019+1)
        elif era==1:
            myyears = np.arange(1982, 1991+1)
        
        for year in myyears:
           for mm in months: all_ice_e[str(year)+'_'+str(mm)] = []
            
           # composite climatology stuff, newnc=split_winds.nc
           census_path = root_path+'era5_data/original_census/'
           if year >= 2000:
               ncpath_area = root_path+'era5_data/areas/'
           elif year < 2000:
               ncpath_area = root_path+'era5_data/areas/'
           ncadd = '_areas'
           
           ncpath = 'era5_data/seaice/'
           ncadd='_seaice' 
           
           # open census (instead of storm_ranges)
           census_file = census_path+'census_'+str(year)+'.csv'
           [startdate, enddate] = cf.readCensus(census_file, convertDT=True)[0]
           
           timing_grid = []
           for xx in range(0,len(startdate)):
               timing_grid.append((startdate[xx], enddate[xx]))
           
           storm_ranges = []
           analysis_ranges = []
           for startdt, enddt in timing_grid:
               week_ago = startdt - timedelta(days=7)
               two_week = startdt + timedelta(days=14) # relative to start date, since different storm lengths
               analysis_ranges.append(cf.daterange(week_ago, two_week, dt=24))
               storm_ranges.append(cf.daterange(startdt, enddt, dt=24))   
           
           # open ice area
           ncpath_area1 = ncpath_area
           ds_area = xr.open_dataset(ncpath_area1 + str(year) +'_area.nc')
           ice_area80 = ds_area['ice_area80'].values
           box_area = ds_area['box_area'].values
           ice_sorter = ice_area80 
           
           try:
               ds = xr.open_dataset(ncpath + str(year) +ncadd+'.nc')
           except:
               print('- skip: '+ncpath + str(year) +ncadd+'.nc')
               continue
            
           for storm_num, strm in enumerate(storm_ranges):
                month = int(strm[0].month)
                if month not in months: continue
                # get ice area
                app1 = ice_sorter[storm_num]*100/box_area[storm_num]
                if np.isnan(app1) or np.isinf(app1):
                    app1=0
                    
                ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]             
                if (ice_frac<np.min(ice_lims) or ice_frac>np.max(ice_lims)):
                    continue
                    
                ### remove storms that don't interact with the ice
                try:
                    sia = ds['sia_miz'+'2_1000'].values[storm_num]
                except:
                    print('-- sia error')
                    continue
                
                if np.nanmean(sia[0:10+1]) == 0 or np.isnan(np.mean(sia[0:10+1])):
                    # print('no ice: '+stormstr)
                    continue
                
                # get time series                                              
                sia_clim = ds['sia_clim_miz'+'2_1000'].values[storm_num]
                ### relativize
                ss = sia-sia_clim
                # standardized_area = (ss-ss[0])/(np.nanmax(ss)-np.nanmin(ss))
                # pcd = (standardized_area)
                
                all_ice_e[str(year)+'_'+str(month)].append(ss[-1] - ss[0]) ##### ICE AREA CHANGE?
                
           try: ds.close()
           except: pass

    return all_ice_e

#%%% functions
import netCDF4

def geoplot_2d(x,y,z=None):    
    #do masked-array on the lon
    x_greater = np.ma.masked_greater(x, -0.01)
    x_lesser = np.ma.masked_less(x, 0)    
    # apply masks to other associate arrays: lat
    y_greater = np.ma.MaskedArray(y, mask=x_greater.mask)
    y_lesser = np.ma.MaskedArray(y, mask=x_lesser.mask)
    
    if z is None:
        return [x_greater, x_lesser], [y_greater, y_lesser]
    else:
        # apply masks to other associate arrays
        z_greater = np.ma.MaskedArray(z, mask=x_greater.mask)
        z_lesser = np.ma.MaskedArray(z, mask=x_lesser.mask)
        return [x_greater, x_lesser], [y_greater, y_lesser], [z_greater, z_lesser]


def load_netcdf(filepath, in_vars):
    """open netcdf file, load variables from list in_vars and output dictionary of variables"""

    out_vars = {}

    open_netcdf = netCDF4.Dataset(filepath, mode = 'r')
    #print open_netcdf
    for var in in_vars:
        out_vars[var] = open_netcdf.variables[var][:]
    open_netcdf.close()

    return out_vars

def load_seaice_v4(root_dir, year, month, day, latlon=True):
    seaice_daily, xgrid, ygrid = [],[],[]
    
    # convert date inputs to strings for filename if needed
    if not isinstance(year, str):
        year = str(year)
    if not isinstance(month, str):
        if month<10: month = '0'+str(int(month))
        else: month = str(int(month))
    if not isinstance(day, str):
        if day<10: day = '0'+str(int(day))
        else: day = str(int(day))
    
    # get file(s)
    all_files = glob.glob(root_dir + '*' + year+month+day + '*.nc')
    all_files.sort()
    variable_names = ['nsidc_bt_seaice_conc','xgrid','ygrid']

    if not all_files:
        print('Error with V4 filename: ' + root_dir + '*' + year+month+day + '*.nc')
        raise NameError(' bad filename in sip.load_seaice')
        
    for n, filename in enumerate(all_files):
        try:
            cdr_dic = load_netcdf(filename, variable_names)
            seaice = cdr_dic['nsidc_bt_seaice_conc']
        except KeyError:
            print('V3 sea ice data used !')
            variable_names = ['goddard_bt_seaice_conc','xgrid','ygrid']
            cdr_dic = load_netcdf(filename, variable_names)
            seaice = cdr_dic['goddard_bt_seaice_conc']
        
        if latlon:
            ygrid    = cdr_dic['ygrid']
            xgrid    = cdr_dic['xgrid']
        else:
            ygrid=[];xgrid=[]
       
        seaice_daily = seaice.mean(axis=0)
    
    return seaice_daily, xgrid, ygrid


def load_seaice(root_dir, year, month, day, latlon=True):
    
    seaice_daily, x, y = load_seaice_v4(root_dir, year, month, day, latlon=True)
    
    if not latlon: return seaice_daily
    
    ds = xr.open_dataset(root_dir + 'seaice_lonlat_v03.nc', decode_times=False)
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    
    return seaice_daily, lon, lat

def cesm_seaice(glob_file):
    dsi = xr.open_dataset(glob_file)
    grid = xr.open_dataset(grid_path+'pop_grid.nc')

    dsi = xr.merge([dsi['aice_d'].drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                   grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                  compat='identical', combine_attrs='no_conflicts')
    dsi.close()
    grid.close()
    gc.collect()
    del grid
    return dsi['aice_d']

def plot_seaicecontour(si_day, si_lon, si_lat, ax, label=[], loc='lower right',
                       color='k', linewidth=1, levels=[0.15,0.80], ls='solid', 
                      ):
    
    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(si_lon, -0.01)
    lon_lesser = np.ma.masked_less(si_lon, 0)    
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(si_lat, mask=lon_greater.mask)
    lat_lesser = np.ma.MaskedArray(si_lat, mask=lon_lesser.mask)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(si_day, mask=lon_greater.mask)
    si_lesser = np.ma.MaskedArray(si_day, mask=lon_lesser.mask)
    
    # contours
    c1 = ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
              linewidths = linewidth, transform=ccrs.PlateCarree(),
              linestyles=ls) 
    c2 = ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
              linewidths = linewidth, transform=ccrs.PlateCarree(),
              linestyles=ls)
    
    return c1, c2


#%% Storm Counts and Location (Fig 1)
gc.collect()
subsubcount = 2

fig = plt.figure(figsize=(15,10))
gs = fig.add_gridspec(2*subsubcount,4, width_ratios=(0.8,0.2,1,1))

bar1 = fig.add_subplot(gs[0:subsubcount,0])
bar2 = fig.add_subplot(gs[subsubcount:subsubcount*2,0])
axes= [bar1, bar2]

bax1 = [fig.add_subplot(gs[x,1]) for x in np.arange(0,subsubcount)]
bax2 = [fig.add_subplot(gs[x,1]) for x in np.arange(subsubcount, subsubcount*2)]
baxes = [bax1, bax2]


axes_all = []
for row in [0,1]:
    axr = []
    for col in [-2,-1]:
        axr.append(fig.add_subplot(gs[row*subsubcount:(row+1)*subsubcount, col], projection=ccrs.NorthPolarStereo()))
    axes_all.append(axr)
axes_all = np.array(axes_all)

spacing=' '*50
title = fig.suptitle('\n'+'Storm Counts'+spacing+'Minimum Pressure Frequency: '
                     +str(min_p)+' hPa', fontsize=fontsize+2)


alph1 = iter(list(string.ascii_lowercase))
for aa in [bar1, bar2]:
    aa.text(-0.15, 1.025, '('+next(alph1)+')', transform=aa.transAxes, fontsize=fontsize, 
                                bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                      'edgecolor':'white', 'lw':0.75},zorder=50)
sub_list = []
bti = iter(['Total','Remaining']*2)
for ba in np.array(baxes).flatten():
    sub_list.append('('+next(alph1)+')\n'+next(bti))
    
for aa in list(axes_all.flatten()):
    aa.text(0.0225, 1.025, '('+next(alph1)+')', transform=aa.transAxes, fontsize=fontsize, 
                                bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                      'edgecolor':'white', 'lw':0.75},zorder=50)
    
plt.subplots_adjust(hspace=0.5)

###################################################
#### - Census Breakdown: Bars (ensemble variability)
###################################################

MONTHS = [6,7,8,9]

e_prefix = ['1231','1251','1281','1301']
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]

# labels = ['ice_distance','duration','pressure','removed-icethresh', 'good storms']
# cinds = [0,6,2,8,4]
labels_text = ['Location', 'Characteristics', 'Insufficient MIZ Interaction', 'Remaining Cyclones']
labels = ['ice_distance','algorithm','removed-icethresh', 'good storms']
cinds = [0,2,8,4]
tab20 = mpl.colormaps['Paired'].colors; 
colors = [tab20[cc] for cc in cinds]

plot_colors = [tab20[cc] for cc in [1,5]] if subsubcount==2 else [tab20[cc] for cc in [1,3,5]]

# sub_iter = iter(['(i)', '(ii)', '(iii)','(iv)'])
sub_iter = iter(sub_list)

era_offset = 0.95
bar_width = 0.85

for era, YEARS in enumerate([np.arange(2010,2019+1),np.arange(1982,1991+1)]):
    
    ax = axes[era]
    bax = baxes[era]
    yrstr = str(YEARS[0])+'-'+str(YEARS[-1])
    
    ax.set_ylim([0,118])
    
    ##########
    #### CESM
    ##########
    
    bars = {}
    for yi, year in enumerate(YEARS): bars[year] = []
    percents = {}
    for lb in labels: 
        percents[lb]=[]
    yearly_good_counts = []
    all_counts = {}
    
    total_storm_counts = []
    loc_storm_counts = []
    net_storm_counts = []
    
    for yi, year in enumerate(YEARS):
        all_cesm_counts = []
        
        for lb in labels+['unique_removed']: 
            all_counts[lb]=[]
        
        ensemble_totals = []
        ensemble_locs = []
        ensemble_nets = []
        for ens_mem in ensemble_members:
            bars_year = []
            counts = {}
            census_path = root_path+'cesm_census/'+'out_n_'+ens_mem+'/'
    
            comparison_tuples = {}
            for filename in ['all', 'duration','ice_distance','pressure','removed-icethresh']:
                census_file = census_path + 'census_test1_'+filename+'_'+str(year)+'.csv'
            
                try:
                    [startdate, enddate], [[startlon, startlat],[endlon,endlat]], pressure = \
                        cf.readCensus(census_file, convertDT=True)
                        
                    counts[filename] = len(startdate)
                except (FileNotFoundError,ValueError):
                    counts[filename] = 0
                    
                ### colelct dates for comparison
                storm_tuples = []
                for xx in range(0,len(startdate)):
                    if startdate[xx].month not in MONTHS: continue
                    storm_tuples.append((startdate[xx], enddate[xx]))    
                comparison_tuples[filename] = storm_tuples
                
            unique_removed_storms = []
            for filename in ['duration','ice_distance','pressure','removed-icethresh']:      
                    for tup in comparison_tuples[filename]:
                        if tup not in unique_removed_storms:
                            unique_removed_storms.append(tup)
                        else:
                            # print(filename, ':', tup)
                            counts[filename] -= 1
                            
            counts['algorithm'] = counts['duration'] + counts['pressure']
            
            counts['good'] = counts['all'] - counts['removed-icethresh']
            yearly_good_counts.append(counts['good'])
            counts['unique_removed'] = len(unique_removed_storms)
            
                                # threhold removals          # good storms
            total_counts =  counts['unique_removed'] + (counts['all']-counts['removed-icethresh'])
            bars_year.append(total_counts )
            all_cesm_counts.append(total_counts)
            ensemble_totals.append( total_counts )
            
            for lb in labels[:-1]+['unique_removed']:
                all_counts[lb].append(counts[lb])
            all_counts['good storms'].append(counts['good'])
            
            remaining_counts = total_counts
            
            for ct in labels[:-1]:
                remaining_counts = remaining_counts - counts[ct]
                bars_year.append(remaining_counts)
                
            ensemble_locs.append( total_counts - counts['ice_distance'])
            ensemble_nets.append(counts['good'])
         
            bars[year].append(bars_year)   
            
            total1 = sum(all_cesm_counts)
            for bb, by in enumerate(bars_year):
                percents[labels[bb]].append(np.sum(all_counts[labels[bb]])*100/total1)
            
        total_storm_counts.append(ensemble_totals)
        loc_storm_counts.append(ensemble_locs)
        net_storm_counts.append(ensemble_nets)
    
    yearly_mean_bars = [np.nanmean(bars[ky],axis=0) for ky in bars]  
    decade_mean_bars = np.nanmean(yearly_mean_bars, axis=0)  

    mypercs = [str(round(np.nanmean(percents[lb]),1))+'%' for li,lb in enumerate(labels)]
    for bb, bar in enumerate(decade_mean_bars):
        ax.bar(0, bar, label = labels_text[bb], color = colors[bb], width=bar_width)
        try:
            ax.text(-0.4, ((bar+decade_mean_bars[bb+1])/2)-1, mypercs[bb], fontsize=fontsize-1)
        except IndexError:
            ax.text(-0.4, 1, mypercs[-1]+' ('+str(round(bar,1))+')', fontsize=fontsize-1)
    
        if bb == 0:   
            ax.text(-0.33, bar+1, 'n='+str(round(bar,1)), fontsize=fontsize-1)

  
    if subsubcount==2: plot_vars = [total_storm_counts,net_storm_counts]
    else: plot_vars = [total_storm_counts, loc_storm_counts, net_storm_counts]
    plot_ind = -1
    for var, col1, bx, limy in zip(plot_vars, plot_colors, bax, [[85,125],[2,12]]):
        plot_ind+=1
        ax.set_ylabel('Detected Storms (per year)', fontsize=fontsize)
        
        for ens_pt in np.nanmean(var, axis=0):
            bx.plot(0.45, ens_pt, '-o', markeredgecolor=col1, markerfacecolor=col1, alpha = 0.5)
            
        bp1 = bx.boxplot(np.nanmean(var, axis=0), positions = [0.66],
                   flierprops={'marker': '+', 'markersize': 8, 'markeredgecolor':col1})
        set_box_color(bp1, color=col1)
        
        bx.set_ylim(limy)
        bx.tick_params(axis='both', which='major', labelsize=fontsize)
        bx.set_title(next(sub_iter), fontsize=fontsize)

    ##########
    #### ERA
    ##########
    
    bars = {}
    for yi, year in enumerate(YEARS): bars[year] = []
    percents = {}
    for lb in labels: 
        percents[lb]=[]
    yearly_good_counts = []
    all_counts = {}
    
    totals, locs, nets = [],[],[]
    
    for yi, year in enumerate(YEARS):
        all_era_counts = []
        
        for lb in labels+['unique_removed']: 
            all_counts[lb]=[]
        
        bars_year = []
        counts = {}
        census_path = root_path+'era5_data/out_n_era3/'

        comparison_tuples = {}
        for filename in ['all', 'duration','ice_distance','pressure','removed-icethresh']:
            census_file = census_path + 'census_'+filename+'_'+str(year)+'.csv'
        
            try:
                [startdate, enddate], [[startlon, startlat],[endlon,endlat]], pressure = \
                    cf.readCensus(census_file, convertDT=True)
                    
                counts[filename] = len(startdate)
            except (FileNotFoundError,ValueError):
                counts[filename] = 0
                
            ### collect dates for comparison
            storm_tuples = []
            for xx in range(0,len(startdate)):
                if startdate[xx].month not in MONTHS: continue
                storm_tuples.append((startdate[xx], enddate[xx]))    
            comparison_tuples[filename] = storm_tuples
            
        unique_removed_storms = []
        for filename in ['duration','ice_distance','pressure','removed-icethresh']:      
                for tup in comparison_tuples[filename]:
                    if tup not in unique_removed_storms:
                        unique_removed_storms.append(tup)
                    else:
                        # print(filename, ':', tup)
                        counts[filename] -= 1
                        
        counts['algorithm'] = counts['duration'] + counts['pressure']
        
        counts['good'] = counts['all'] - counts['removed-icethresh']
        yearly_good_counts.append(counts['good'])
        counts['unique_removed'] = len(unique_removed_storms)
        
                            # threhold removals          # good storms
        total_counts =  counts['unique_removed'] + (counts['all']-counts['removed-icethresh'])
        bars_year.append(total_counts )
        all_era_counts.append(total_counts)
        
        for lb in labels[:-1]+['unique_removed']:
            all_counts[lb].append(counts[lb])
        all_counts['good storms'].append(counts['good'])
        
        remaining_counts = total_counts
        
        for ct in labels[:-1]:
            remaining_counts = remaining_counts - counts[ct]
            bars_year.append(remaining_counts)
            
        totals.append( total_counts )
        locs.append( total_counts - counts['ice_distance'])
        nets.append(counts['good'])
     
        bars[year].append(bars_year)   
        
        total1 = sum(all_era_counts)
        for bb, by in enumerate(bars_year):
            percents[labels[bb]].append(np.sum(all_counts[labels[bb]])*100/total1)
            
    yearly_mean_bars = [np.nanmean(bars[ky],axis=0) for ky in bars]  
    decade_mean_bars = np.nanmean(yearly_mean_bars, axis=0)  

    mypercs = [str(round(np.nanmean(percents[lb]),1))+'%' for li,lb in enumerate(labels)]
    
    for bb, bar in enumerate(decade_mean_bars):
        ax.bar(era_offset, bar, label = labels_text[bb], color = colors[bb], width=bar_width)
        try:
            ax.text(era_offset-0.4, ((bar+decade_mean_bars[bb+1])/2)-2, mypercs[bb], fontsize=fontsize-1.25)
        except IndexError:
            ax.text(era_offset-0.425, 2, mypercs[-1]+' ('+str(round(bar,1))+')', fontsize=fontsize-1.25)
    
        if bb == 0:   
            ax.text(era_offset-0.33, bar+1, 'n='+str(round(bar,1)), fontsize=fontsize-1)

    
    if subsubcount==2: plot_vars = [totals, nets]
    else: plot_vars = [totals, locs, nets]
    plot_ind = -1
    for var, col1, bx in zip(plot_vars, plot_colors, bax):
        plot_ind+=1
        bx.plot(0.95, np.nanmean(var), marker='x', markersize=9,
                markeredgecolor=col1, markerfacecolor=col1)
        bx.set_xticklabels([])
        bx.set_xticks([])
    
    #### axis organization
    handles, labels1 = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels1)) if l not in labels1[:i]]

    unique2=[unique[ii] for ii in [0,1,2]]
    unique3 = [(unique[-1][0],'Remaining cyclones after the filtering process\n(Cyclones considered in this study)')]
    
    if era==1: ax.legend(*zip(*unique2), bbox_to_anchor = (1.95,-0.125), 
                         ncol = 3, fontsize=fontsize, #(1,-0.075)
                         handletextpad=0.5, handlelength=1.25,
                         columnspacing=0.8,
                         title='Number of Cyclones Removed by These Factors:',
                         title_fontproperties={'weight':535,
                                               'size':fontsize} ) 
    elif era==0: ax.legend(*zip(*unique3), bbox_to_anchor = (0.66,-1.725), #-1.66
                         ncol = 1, fontsize=fontsize, #(1,-0.075)
                         handletextpad=0.5, handlelength=1.25,
                         columnspacing=0.8,
                         loc='lower center',
                         # title='Number of Cyclones Removed by These Factors:',
                         title_fontproperties={'weight':535,
                                               'size':fontsize} ) 

    ax.set_xticks([0,era_offset])
    ax.set_xticklabels(['CESM2','ERA5'], fontsize=fontsize)
    ax.set_ylabel('Detected Storms (per year)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_title(yrstr+': Ensemble Mean',  fontsize=fontsize)
  

bx.plot([],[], marker='x', color='k', markersize=8, linestyle='None', label='ERA5')
bx.plot([],[], '-o', color='k', markersize=8, alpha=0.5, label='CESM2')
bx.legend(bbox_to_anchor=(2.25,-0.095), ncol=2)

###################################################
#### - Pressure Maps
###################################################

import time as timeIN
import warnings
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

cmap_f = cmo.amp
# fontweight='bold',

ice_fname = root_path+'seaice/'
cesm_ice = root_path+'data/aice_d/'
grid_path = root_path+'data/'

e_prefix = ['1231','1251','1281','1301']
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]


pkl_file = open(root_path+'processed-data/figures/cesm_coords.pkl', 'rb')
[cesm_lon, cesm_lat] = pickle.load(pkl_file)
pkl_file.close()

titles = ['ERA5','CESM2']
LEVELS2 = np.arange(0,0.40,0.05)*100

### set up plot
for ax in axes_all.flatten(): 
    ax = cf.setup_plot(ax, extent=[0,360,60,90], labels=False)
    ax.set_boundary(circle, transform=ax.transAxes)


for yi, years in enumerate([np.arange(2010,2019+1),np.arange(1982,1991+1)]):
    ystr = str(years[0])+'-'+str(years[-1])
    print(); print('Starting '+ystr+' map')
    year_ax = axes_all[yi]

    #### *CESM
    pkl_file = open(root_path+'processed-data/figures/'+'ens_freq_'+str(ystr)+'.pkl', 'rb')
    ens_freq = pickle.load(pkl_file)
    pkl_file.close()
    gc.collect()

    #### *ERA 
    freq_e = np.zeros(np.shape(cesm_lon))
    for year in years:
        time1 = cf.daterange(datetime(year,6,1), datetime(year,9,30,23), dt=6)
        
        ## load era
        era5 = xr.open_dataset(root_path+'processed-data/figures/'+'era-cesm-interp/'+str(year)+'_arctic.nc')
        
        ### day loop:
        for ti, dt in enumerate(time1):
            dtstr = dt.strftime('%Y-%m-%d %H:%M:%S')
            msl_e = np.squeeze( era5.isel(time=ti).to_array().values )
            
            min_inds = np.where(msl_e < min_p)
            freq_e[min_inds] += 1
        
    # cesm plot
    ax2 = year_ax[0]
    ax2.set_title(titles[1]+': '+ystr, fontsize=fontsize)
    var2 = np.nanmean(np.array(ens_freq), axis=0)
    cyclic_data2, cyc_lon = add_cyclic_point(var2, coord=np.where(cesm_lon[0,:]<0, cesm_lon[0,:]+360, cesm_lon[0,:]))
    slp_c = ax2.contourf(cyc_lon, cesm_lat[:,0], cyclic_data2*100,
                            transform=ccrs.PlateCarree(), alpha=0.95,
                            cmap=cmap_f, levels=LEVELS2)

    # era plot
    ax1 = year_ax[1]
    ax1.set_title(titles[0]+': '+ystr,fontsize=fontsize)
    var1 = freq_e/len(time1)
    cyclic_data1, cyc_lon = add_cyclic_point(var1, coord=np.where(cesm_lon[0,:]<0, cesm_lon[0,:]+360, cesm_lon[0,:]))
    slp_e = ax1.contourf(cyc_lon, cesm_lat[:,0], cyclic_data1*100,
                                transform=ccrs.PlateCarree(),alpha=0.95,
                                cmap=cmap_f, levels=LEVELS2)
    
    ## add ice contours
    try:
        # raise FileNotFoundError()
        pkl_file = open(root_path+'processed-data/figures/'+'summer_seaice_cesm_'+str(ystr)+'.pkl', 'rb')
        seaice_cesm = pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open(root_path+'processed-data/figures/'+'summer_seaice_era_'+str(ystr)+'.pkl', 'rb')
        seaice_era = pickle.load(pkl_file)
        pkl_file.close()
        gc.collect()
    except FileNotFoundError:
        print('... sea ice calculations')
        seaice_time = timeIN.time()
        seaice_cesm = []
        seaice_era = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for yy in years:
                ytime = timeIN.time()
                print('- '+str(yy), end=', ')
                ## cesm sea ice
                seaice_ens = []
                for ens_mem in ensemble_members:
                    dsi = cesm_seaice(glob.glob(cesm_ice +'*'+ ens_mem +'*h1.aice_d*.'+ cf.file_year(yy)+'01*.nc')[-1])
                    ice_year = dsi.sel(time=dsi.time.dt.year == yy)
                    summerice = ice_year.sel(time = np.isin(ice_year.time.dt.month, months))
                    seaice_ens.append(np.nanmean(summerice.values, axis=0))
                    gc.collect()
                seaice_cesm.append(np.nanmean(seaice_ens, axis=0))
                    
                ## era sea ice
                time = cf.daterange(datetime(yy,6,1), datetime(yy,9,30,23), dt=6)
                ices = []
                for dt1 in time:
                    si = load_seaice(ice_fname, yy, dt1.month, dt1.day, latlon=False)
                    ices.append(si)
                seaice_era.append(np.nanmean(ices, axis=0))
                
                print(round(timeIN.time()-ytime,2), 'seconds')
            
        print('* TOTAL SEAICE TIME: '+str(round((timeIN.time()-seaice_time)/60, 1))+' mins')
    
        ystr = str(years[0])+'-'+str(years[-1])
        output = open(root_path+'processed-data/figures/'+'summer_seaice_cesm_'+str(ystr)+'.pkl', 'wb')
        pickle.dump(seaice_cesm, output)
        output.close()
        output = open(root_path+'processed-data/figures/'+'summer_seaice_era_'+str(ystr)+'.pkl', 'wb')
        pickle.dump(seaice_era, output)
        output.close()
        gc.collect()
    
    
    dsi = cesm_seaice(glob.glob(cesm_ice +'*'+ '1301.020'+'*h1.aice_d*.'+ cf.file_year(2015)+'01*.nc')[-1])

    
    for lvl,lw in zip([[0.15],[0.80]],[1.5,2]):
    
        _, si_lon, si_lat = load_seaice(root_path+'seaice/', 2010, 6, 1, latlon=True)
        sc_e1, sc_e2 = plot_seaicecontour(np.nanmean(seaice_era, axis=0), si_lon, si_lat, 
                                          ax1, levels=lvl,color='k', linewidth=lw)
        
        si_lon = dsi['TLONG'].values
        si_lon = np.where(si_lon>180, si_lon-360, si_lon)
        si_lat = dsi['TLAT'].values
        
        sc_c1, sc_c2 = plot_seaicecontour(np.nanmean(seaice_cesm, axis=0), si_lon, si_lat, ax2, 
                                          levels=lvl,color='k', linewidth=lw)
    


cax1 = fig.add_axes([0.5,0.05,0.33,.033]) 
cbar1 = fig.colorbar(slp_e, cax=cax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize=fontsize-1)
added_label = '(number of timesteps with SLP < 986 hPa / all timesteps in JJAS)'
cbar1.set_label('Frequency (%)', fontsize=fontsize) # per 6h?
ax1.text(-0.825, -0.425, added_label, transform=ax.transAxes, fontsize=fontsize-2)
        # -0.75, -0.425

#### --> save
if SAVEFIG:
    fig.savefig(savepath+'bar_pmaps.pdf')

gc.collect()

#%% MIZ impacts, boxplots, net impact: combined (Fig 2)
from matplotlib.gridspec import GridSpec
xxx = np.arange(-7,14+1,1)
title1, title2 = 'June, July: ', 'Aug, Sep: '
alph = iter(list(string.ascii_lowercase))
fs = 12

#### set up plot
fig = plt.figure(layout="constrained", figsize=(12,16))
gs = GridSpec(4,4, figure=fig, 
              width_ratios=(1, 1,1,1), height_ratios=(0.75,0.75, 1,1))


axes_all = [[fig.add_subplot(gs[0, 0:2]),fig.add_subplot(gs[0, 2:4])],
            [fig.add_subplot(gs[1, 0:2]),fig.add_subplot(gs[1, 2:4])]]
axes_all = np.array(axes_all)

labels1 = [-7] +['']*6 + [0] + ['']*6 + [7] + ['']*6 +[14]

stat_thresh = 0.001

################
#### spaghetti 
################

with HidePrint():
    TOTAL_ALL, ens_frac, ens_net = get_spaghetti(normalize = False)

era_names = ['2010-2019','1982-1991']
mean_lines = {'0_early':[], '0_late':[], '1_early':[], '1_late':[]}

for ensemble in TOTAL_ALL:
    for era in [0,1]:
        simple_mean1, simple_mean2 = [],[]
        for month in months:
            if month in months[0:2]: simple_mean1+= ensemble[str(era)+'_'+str(month)]
            elif month in months[-2:]: simple_mean2+= ensemble[str(era)+'_'+str(month)]
            
        mean_lines[str(era)+'_early'].append(np.nanmean(np.array(simple_mean1), axis=0)/1e6)
        mean_lines[str(era)+'_late'].append(np.nanmean(np.array(simple_mean2), axis=0)/1e6)
        
# load era comparison
savepath_era1 = root_path+'processed-data/figures/'
savename_era1 = 'non-norm_mean_era_lines.npy'
era_mean = np.load(savepath_era1+savename_era1)

for ax in axes_all[:,0]:
    ax.set_ylabel('Relative Change in\nIce Area '+r'($\times 10^5$km$^2$)', fontsize=fs)
for ax in axes_all[1,:]:
    ax.set_xticklabels(labels1, minor=False, rotation=0, fontsize=fs)
    ax.set_xlabel('Days Since Storm Start', fontsize=fs)
for axl in axes_all:
    for ax1 in axl:
        ax1.set_xlim(-7,14)
        ax1.set_ylim(-2.25,0.75)
        ax1.axhline(0, ls='-', color='k', lw=1)
        ax1.axvline(0, ls=':', color='darkgray', lw=0.75)
        ax1.set_xticks(xxx)
        ax1.tick_params(axis='both', which='major', labelsize=fs)  
for ax1 in axes_all[0]:
    ax1.set_xticks(xxx)
    ax1.tick_params(axis='both', which='major', labelsize=fs) 
    ax1.set_xticklabels(labels1, minor=False, rotation=0, fontsize=fs)
        
spread_color = 'maroon'
line_color = 'dimgray'

scale = 1e5
era_index = 0
era_color = 'navy'

for era in [0,1]:
    axes_all[0][era].plot(xxx, np.array(mean_lines[str(era)+'_early']).T/scale, 
                          color=line_color, lw=0.5)
    axes_all[1][era].plot(xxx, np.array(mean_lines[str(era)+'_late']).T/scale, 
                          color=line_color, lw=0.5)
    
    axes_all[0][era].set_title(title1+era_names[era], fontsize=fontsize)
    axes_all[1][era].set_title(title2+era_names[era], fontsize=fontsize)
    
    for season, name in zip([0,1], ['_early', '_late']):
        mean = np.nanmean( np.array(mean_lines[str(era)+name]), axis=0 )/scale
        stdev = np.nanstd( np.array(mean_lines[str(era)+name]), axis=0 )/scale
        
        axes_all[season][era].plot(xxx, mean, color=spread_color, lw=2)
        axes_all[season][era].plot(xxx, mean+stdev, color=spread_color, lw=1.5, ls='--')
        axes_all[season][era].plot(xxx, mean-stdev, color=spread_color, lw=1.5, ls='--')
        axes_all[season][era].fill_between(xxx, mean-stdev, mean+stdev,
                               alpha=0.5, color=spread_color)
        
        axes_all[season][era].text(0.0225, 0.915, '('+next(alph)+')', 
                                   transform=axes_all[season][era].transAxes, 
                                   fontsize=fontsize, 
                                bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                      'edgecolor':'white', 'lw':0.75},zorder=50)
        
        print('***  '+str(era)+name)
        print(mean[-1])
        print(era_mean[era_index][-1])
        print( round(mean[-1] / era_mean[era_index][-1], 2) )
        
        
        axes_all[season][era].plot(xxx, era_mean[era_index], color=era_color, lw=2)
        era_index+=1
        
        
# legend?
leg_ax = axes_all[0][0]
loc = 'lower left' #'upper right'
## CESM
leg_ax.plot([],[], color=line_color, label = "CESM2 Ensemble Member")
leg_ax.plot([],[], color=spread_color, label = "CESM2 Mean, Std.Dev")
leg_ax.plot([],[], color=era_color, ls='-', alpha=1, label = "ERA5 Mean")
leg_ax.legend(loc=loc, fontsize=fs)

#############################
#### Storm Fraction Boxplots
#############################

data_locs = {'0_67':0, '0_89':1, '1_67':2.5, '1_89':3.5}
erastrs = ['2010-2019', '1982-1991']
seastrs = ['June & July', 'Aug & Sep']
mycol = 'maroon'
ecol = 'navy'

sst_thresh = 0.01 #0.01
ice_lims = [20,80]

IND = 0 # original miz (15/80), RMS miz
VAR = ['sia_miz_1000','sia_miz2_1000'][IND]

myeras = [np.arange(1982,1992), np.arange(2010,2020)]
months = [6,7,8,9]

path1 = root_path+'cesm_census/'
hemisphere = 'n'
cens_name = 'census_test1_all_'
ens_dirs = glob.glob(path1+'out_'+hemisphere+'_1*/')

# plot characterstics

ax1 = fig.add_subplot(gs[2, 0:2])
ax2 = fig.add_subplot(gs[2, 2:4])

for ax in [ax1,ax2]:
    ax.set_xticks(list(data_locs.values()))
    ax.set_xticklabels([erastrs[e]+'\n'+seastrs[s] for e in range(len(myeras)) for s in [0,1]], fontsize=fs)
    ax.tick_params(axis='y', which='major', labelsize=fs)
    ax.plot([],[], '-o', color=mycol, markeredgecolor=mycol, markerfacecolor=mycol, alpha = 0.5,label='CESM2')
    ax.plot([],[], 'x', markersize=8, color = ecol, label='ERA5', linestyle='None')
    ax.axhline(0.5, lw=1, ls=':', color='gray')
    ax.text(0.01, 1.025, '('+next(alph)+')', transform=ax.transAxes, fontsize=fs, 
            bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                  'edgecolor':'white', 'lw':0.75},zorder=50)


#### sea ice boxplot (incr/decr)
##################################

# get values: cesm, era
TOTAL_ALL1, ens_frac1, ens_net = get_spaghetti(normalize = True)
pkl_file = open(root_path+'processed-data/figures/frac.pkl', 'rb')
era_frac = pickle.load(pkl_file)
pkl_file.close()

# ensembles !
xtix1 = []
for era in [0,1]:
    for s, seas in enumerate(['_67', '_89']):
        mybox = []
        for counters in ens_frac1:
            total = counters[str(era)+seas+'_incr'] + counters[str(era)+seas+'_decr']
            frac_decr = counters[str(era)+seas+'_decr']/total
            mybox.append(frac_decr)
            ax1.plot(data_locs[str(era)+seas]-0.15, frac_decr, marker='o', markersize=8,
                    markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
            
        bp = ax1.boxplot(mybox, positions = [data_locs[str(era)+seas]+0.05],
                        flierprops={'marker': '+', 'markersize': 8, 'markeredgecolor':mycol})
        set_box_color(bp, color=mycol)
        
        # ERA5
        total = era_frac[str(era)+seas+'_incr'] + era_frac[str(era)+seas+'_decr']
        frac_decr = era_frac[str(era)+seas+'_decr']/total
        ax1.plot(data_locs[str(era)+seas]+0.15, frac_decr, marker='x', markersize=8,
                markeredgecolor=ecol, markerfacecolor=ecol)

        t_stat, p_val = ttest_1samp(mybox, frac_decr, alternative='greater')[0:2]
        print(str(era)+seas+'_frac'+': '+ str(p_val))
        if p_val > stat_thresh: s67 = '*'
        else: s67=''
        xtix1.append(erastrs[era]+'\n'+seastrs[s]+s67)
        
        
ax1.set_xticks(list(data_locs.values()))
ax1.set_xticklabels(xtix1, fontsize=fs)

ax1.set_ylabel('Fraction of MIZ Ice Area\nDecreasing Storms', fontsize=fs)
ax1.set_title('\nCESM2-LE has a greater fraction\nof ice-decreasing storms than ERA5', fontsize=fs+2)


#### sst boxplot (incr/decr)
################################ # future_decs.py, tseries_trends.py

ens_frac_s = []

for ens_dir in ens_dirs:
    ens_mem = ens_dir[-9:-1]
    ncpath = ens_dir+'seaice/'
    sort_path = ens_dir+'winds_sst/'
    
    ### FRACTION OF INCREASING/DECREASING STORMS
    frac_winds = {}
    frac_sst = {}
    for era in range(len(myeras)):
        for mm in ['67', '89']:
            frac_winds[str(era)+'_'+mm+'_neg']=0
            frac_winds[str(era)+'_'+mm+'_pos']=0
            frac_sst[str(era)+'_'+mm+'_incr']=0
            frac_sst[str(era)+'_'+mm+'_decr']=0
        
    casestudynum=1
    for era in range(len(myeras)):
        myyears = myeras[era]
        
        all_pcd67, all_pcd89 = [],[]
        total_storms=0
        count67, count89 = 0,0
        
        for year in myyears:
           # open census (instead of storm_ranges)
           try:
               census_file = ens_dir+cens_name+str(year)+'.csv'
               [startdate, enddate] = cf.readCensus(census_file, convertDT=True)[0]
               pressures = cf.readCensus(census_file, convertDT=True)[-1]
           except ValueError as ve:
               print('- Empty Census: ' + str(year)); print(ve); 
               continue
           
           timing_grid = []
           for xx in range(0,len(startdate)):
               timing_grid.append((startdate[xx], enddate[xx]))
           
           storm_ranges = []
           analysis_ranges = []
           for startdt, enddt in timing_grid:
               week_ago = startdt - timedelta(days=7)
               two_week = startdt + timedelta(days=14) # relative to start date, since different storm lengths
               analysis_ranges.append(cf.daterange(week_ago, two_week, dt=24))
               storm_ranges.append(cf.daterange(startdt, enddt, dt=24))   
           
           # open ice area
           ds_area = xr.open_dataset(ens_dir+'/'+ str(year) +'_area.nc')
           ice_area80 = ds_area['ice_area80'].values
           ice_area15 = ds_area['ice_area15'].values
           box_area = ds_area['box_area'].values
           
           ice_sorter = ice_area80
           
           try:
               ds = xr.open_dataset(ncpath + str(year) +'_seaice.nc')
           except:
               print('- skip: '+ncpath + str(year) +'_seaice.nc')
               continue
           
           ### load wind and sst files
           try:
               dsws = xr.open_dataset(sort_path+str(year)+'_sort.nc')
               pass
           except:
               print('no sorting file: '+ens_mem+' ('+str(year)+')')
            
           for storm_num, strm in enumerate(storm_ranges):
                month = int(strm[0].month)
                # get ice area
                with warnings.catch_warnings(action="ignore"):
                    ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]
                if np.isnan(ice_frac) or np.isinf(ice_frac):
                    ice_frac=0
                    
                if (ice_frac<np.min(ice_lims) or ice_frac>np.max(ice_lims)):
                    continue
                    
                ### remove storms that don't interact with the ice
                try:
                    sia = ds[VAR].values[storm_num]
                except:
                    continue
                
                if (len(np.unique(sia))==1 and np.unique(sia)[0] == 0) or np.isnan(np.mean(sia[0:10+1])):
                    # print('no ice: '+strm[0].strftime('%Y-%m-%d')+' ('+str(np.unique(sia))+')')
                    continue 
                
                ### get wind tseries
                sst2 = dsws['sst_1000'].sel(nstorms=storm_num+1).sel(miz_type=IND)
                sst1 = calc_deriv(sst2, dt=1)
                
                ### get sst tseries
                total_storms += 1
                
                storm_sst = np.mean( sst1[0:len(storm_ranges[storm_num])] )
             
                if month in months[0:2]:
                    count67+=1
                    
                    if storm_sst > sst_thresh:
                        frac_sst[str(era)+'_67_incr']+=1
                    elif storm_sst < -sst_thresh:
                        frac_sst[str(era)+'_67_decr']+=1
                
                elif month in months[2:]:
                    count89+=1
                    
                    if storm_sst > sst_thresh:
                        frac_sst[str(era)+'_89_incr']+=1
                    elif storm_sst < -sst_thresh:
                        frac_sst[str(era)+'_89_decr']+=1
                        
                else:
                    break
         
        
           try: ds.close()
           except: pass
    ens_frac_s.append(frac_sst)
gc.collect()

### plot

# era5
era_sst = {'0_67_decr': 23, '0_67_incr':24,
             '0_89_decr':29, '0_89_incr':16,
             '1_67_decr':15, '1_67_incr':11,
             '1_89_decr':42, '1_89_incr':18}

xtix2 = []
for era in range(len(myeras)):
    for s, seas in enumerate(['_67', '_89']):
        box_sst = []
        for s_count in ens_frac_s:
            total = s_count[str(era)+seas+'_incr'] + s_count[str(era)+seas+'_decr']
            if total==0: continue
            frac_incr = s_count[str(era)+seas+'_incr']/total
            box_sst.append(frac_incr)
            ax2.plot(data_locs[str(era)+seas]-0.15, frac_incr, marker='o', markersize=8,
                    markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
            
        bp_s = ax2.boxplot(box_sst, positions = [data_locs[str(era)+seas]+0.05],
                        flierprops={'marker': '+', 'markersize': 8, 'markeredgecolor':mycol})
        set_box_color(bp_s, color=mycol)

        ### era5
        total = era_sst[str(era)+seas+'_incr'] +era_sst[str(era)+seas+'_decr']
        if total==0: continue
        frac_pos_e = era_sst[str(era)+seas+'_incr']/total
        ax2.plot(data_locs[str(era)+seas]+0.25, frac_pos_e, marker='x', markersize=9,
                markeredgecolor=ecol, markerfacecolor=ecol)
        
        t_stat, p_val = ttest_1samp(box_sst, frac_pos_e)[0:2]
        print(str(era)+seas+'_sst'+': '+ str(p_val))
        if p_val > stat_thresh: s67 = '*'
        else: s67=''
        xtix2.append(erastrs[era]+'\n'+seastrs[s]+s67)
        
        
ax2.set_xticks(list(data_locs.values()))
ax2.set_xticklabels(xtix2, fontsize=fs)

# ax2.legend(loc='lower left', ncol=1, fontsize=fs)

ax2.set_title('\nEarly-Summer Storms Correspond to SST Increases\nMore Frequently in CESM2-LE than ERA5', fontsize=fs+2)
ax2.set_ylabel('Fraction of Storms with\nIncreasing SST within the MIZ', fontsize=fs)

gc.collect()

#### net impact boxplots
##################################

# f6, ax6 = plt.subplots(1,1, figsize=(6,4))
ax6 = fig.add_subplot(gs[-1, 1:3])

all_ice_e = era_ice()

e_prefix = ['1231','1251','1281', '1301']
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]

data_locs = {'0_67':0, '0_89':1, '1_67':2.5, '1_89':3.5}

erastrs = ['2010-2019', '1982-1991']
seastrs = ['June & July', 'Aug & Sep']
mycol = 'maroon'
ecol = 'navy'

stats_add =[]
for era, years in enumerate([np.arange(2010,2019+1), np.arange(1982,1991+1)]):
    
    ens67, ens89 = [],[]
    for ens_mem in ensemble_members:
        changes = ens_net[ens_mem]
        
        change67 = []
        change89 = []
        for year in years:
            change67.append(np.sum(changes[str(year)+'_6'])+np.sum(changes[str(year)+'_7']))
            # ax6.plot(data_locs[str(era)+'_67']-0.15, change67[-1]/1e6/1e6, marker='o', markersize=8,
            #         markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
            
            change89.append(np.sum(changes[str(year)+'_8'])+np.sum(changes[str(year)+'_9']))
            # ax6.plot(data_locs[str(era)+'_89']-0.15, change89[-1]/1e6/1e6, marker='o', markersize=8,
            #         markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
            
        ens67.append(np.nansum(change67)/1e6/1e6)
        ens89.append(np.nansum(change89)/1e6/1e6)

    bp = ax6.boxplot(ens67, positions = [data_locs[str(era)+'_67']-0.05],
                    flierprops={'marker': '+', 'markersize': 8, 'markeredgecolor':mycol})
    set_box_color(bp, color=mycol)
    
    bp = ax6.boxplot(ens89, positions = [data_locs[str(era)+'_89']-0.05],
                    flierprops={'marker': '+', 'markersize': 8, 'markeredgecolor':mycol})
    set_box_color(bp, color=mycol)
    
    for e6 in ens67:
        ax6.plot(data_locs[str(era)+'_67']-0.25, e6, marker='o', markersize=8,
                markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
    for e8 in ens89:
        ax6.plot(data_locs[str(era)+'_89']-0.25, e8, marker='o', markersize=8,
                markeredgecolor=mycol, markerfacecolor=mycol, alpha=0.5)
    
    ## era 
    era67, era89 = [],[]
    for year in years:
        era67.append(np.sum(all_ice_e[str(year)+'_6'])+np.sum(all_ice_e[str(year)+'_7']))
        era89.append(np.sum(all_ice_e[str(year)+'_8'])+np.sum(all_ice_e[str(year)+'_9']))
        
    ax6.plot(data_locs[str(era)+'_67']+0.25, np.sum(era67)/1e6, marker='x', markersize=8,
             markeredgecolor=ecol, markerfacecolor=ecol)
    ax6.plot(data_locs[str(era)+'_89']+0.25, np.sum(era89)/1e6, marker='x', markersize=8,
             markeredgecolor=ecol, markerfacecolor=ecol)
            
    
    t_stat, p_val = ttest_1samp(ens67, np.sum(era67)/1e6)[0:2]
    print(str(era)+'_67: '+ str(p_val))
    if p_val > stat_thresh: s67 = '*'
    else: s67=''
    t_stat, p_val = ttest_1samp(ens89, np.sum(era89)/1e6)[0:2]
    print(str(era)+'_89: '+ str(p_val))
    if p_val > stat_thresh: s89 = '*'
    else: s89=''
    stats_add.append([s67,s89])
            
## plot characterstics
fs = 11
ax6.set_xticks(list(data_locs.values()))
ax6.set_xticklabels([erastrs[e]+'\n'+seastrs[s]+stats_add[e][s] for e in [0,1] for s in [0,1]], fontsize=fs)
ax6.tick_params(axis='y', which='major', labelsize=fs)
ax6.set_ylabel('Change in MIZ Ice Area Due to\nCyclone Activity '+r'($\times$10$^6$ km$^2$)', fontsize=fs)
ax6.axhline(0, lw=0.5, color='gray', ls=':')

ax6.plot([],[], '-o', color=mycol, markersize=8, alpha=0.75, label='CESM2')
ax6.plot([],[], 'x', markersize=8, color = ecol, label='ERA5', linestyle='None')
ax6.legend(loc='lower right', fontsize=fs, ncol=1)
ax6.set_title('\nNet Impacts', fontsize=fs+4);

ax6.text(0.0225, 1.025, '('+next(alph)+')', transform=ax6.transAxes, 
        fontsize=fontsize, bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                 'edgecolor':'white', 'lw':0.75},zorder=50)
gc.collect()
if SAVEFIG:
    fig.savefig(savepath+'all_seaice_sst_net.pdf')
    
#%%--- supplemental figures
import calendar

cesm_ice = root_path+'data/aice_d/'
grid_path = root_path+'data/'

dsi = cesm_seaice(glob.glob(cesm_ice +'*'+ '1301.020'+'*h1.aice_d*.'+ cf.file_year(2015)+'01*.nc')[-1])
si_lon = dsi['TLONG'].values
si_lon = np.where(si_lon>180, si_lon-360, si_lon)
si_lat = dsi['TLAT'].values


#%% make indiv. storm impact comparison subplots (not shown in paper)
# def align_zeros(axes):

#     ylims_current = {}   #  Current ylims
#     ylims_mod     = {}   #  Modified ylims
#     deltas        = {}   #  ymax - ymin for ylims_current
#     ratios        = {}   #  ratio of the zero point within deltas

#     for ax in axes:
#         ylims_current[ax] = list(ax.get_ylim())
#                         # Need to convert a tuple to a list to manipulate elements.
#         deltas[ax]        = ylims_current[ax][1] - ylims_current[ax][0]
#         ratios[ax]        = -ylims_current[ax][0]/deltas[ax]
    
#     for ax in axes:      # Loop through all axes to ensure each ax fits in others.
#         ylims_mod[ax]     = [np.nan,np.nan]   # Construct a blank list
#         ylims_mod[ax][1]  = max(deltas[ax] * (1-np.array(list(ratios.values()))))
#                         # Choose the max value among (delta for ax)*(1-ratios),
#                         # and apply it to ymax for ax
#         ylims_mod[ax][0]  = min(-deltas[ax] * np.array(list(ratios.values())))
#                         # Do the same for ymin
#         ax.set_ylim(tuple(ylims_mod[ax]))


# #### miz area
# f = root_path+'processed-data/figures/miz_data_all.nc'
# miz_data = xr.open_dataarray(f) 
# months = [5,6,7,8,9]
# decades = [np.arange(1982,1992), np.arange(1990,2000),
#            np.arange(2000,2010), np.arange(2010,2020),
#            np.arange(2020,2030), np.arange(2030,2040),
#            np.arange(2040,2050), np.arange(2050,2060),
#            np.arange(2060,2070), np.arange(2070,2080),
#            np.arange(2080,2090), np.arange(2090,2100)
#             ]

# miz_avg = miz_data.mean(dim = ('years','ensemble_members','days')).sel(lat_bands='total')

# miz_decs = [(miz_avg.sel(decades=dec)) for dec in miz_data.decades]

# #### impact calculations
# annual_sum = []
# month_sum = {mm:[] for mm in months}

# for yx, years in enumerate(decades):
#     year_maps = []
#     for mm, month in enumerate(months):
#         savepath5 = root_path+'processed-data/spatial/'
#         savename5 = 'plot_difference2_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
#         grd = np.load(savepath5+savename5)
#         if np.all(np.isnan(grd)):
#             grd = np.zeros(np.shape(si_lon))
        
#         year_maps.append( grd )
#         month_sum[month].append( np.nansum(grd) )
        
#     plot_change = np.nansum(year_maps,axis=0)
#     plot_change = np.where(plot_change==0, np.nan, plot_change)  
#     annual_sum.append( np.nansum(plot_change) )

# #### relative changes: plot
# mcolors = ['#E68310', '#E73F74', '#7F3C8D', '#008695', '#4b4b8f']
# scale=1e6
# fig, axes = plt.subplots(6,1, figsize=(8,12), sharex=True)

# rel_list=[]
# for mi, mm in enumerate(months):
#     # old
#     ax1 = axes[mi].twinx()
#     old = ax1.plot(np.array(month_sum[mm])/scale, color=mcolors[mi], lw=1.75, ls=':')
    
#     # scaled
#     rel_changes = np.array(month_sum[mm])/np.array(miz_decs)[:,mi]
#     rel_list.append(rel_changes)
#     axes[mi].plot(np.array(rel_changes)*100, color=mcolors[mi], lw=2)
    
#     # organize plot
#     axes[mi].set_title(calendar.month_name[mm])
#     axes[mi].set_xticks(np.arange(len(decades)))
#     axes[mi].axhline(0, lw=0.75, color='gray', ls=':')
#     if mi==1 or mi==4: ax1.set_ylabel('Change in MIZ Ice Area'+r'($\times 10^6$ km$^2$)')
#     align_zeros([ax1, axes[mi]])

# ## total
# axes[mi+1].plot(np.nanmean(rel_list,axis=0)*100, color='k', lw=2)
# axes[mi+1].axhline(np.nanmin(np.nanmean(rel_list,axis=0)*100), lw=0.55, color='gray', ls=':')
# # axes[mi+1].axhline(0, lw=0.75, color='gray', ls=':')
# axes[mi+1].set_title('Total')
# ax2 = axes[mi+1].twinx()
# ax2.plot(np.array(annual_sum)/scale, color='k', lw=2, ls=':')
# align_zeros([ax2, axes[mi+1]])
 
# for ax in [axes[-2], axes[1]]: ax.set_ylabel('Total Relative Change per MIZ Area (%)')
# axes[-1].set_xticklabels(['\''+str(yy[0])[2]+'0s' for yy in decades])



# axes[-1].plot([],[], lw=2, ls='-', color='gray', label ='Normalized by MIZ Area\n          (left axis)')
# axes[-1].plot([],[], lw=2, ls=':', color='gray', label ='Change in MIZ Ice Area (as in Fig. 4)\n             (right axis)')
# axes[-1].legend(loc='lower left', bbox_to_anchor=(0.1, -0.66),
#                 ncol=2, handlelength=2.5, handletextpad=0.5)

#%% fig s2: individual month impacts 
def era_tseries():
    ice_lims = [20,80]
    all_ice_e = {}
    for era in [0,1]:
        if era==0:
            myyears = np.arange(2010, 2019+1)
        elif era==1:
            myyears = np.arange(1982, 1991+1)
        
        for year in myyears:
           for mm in months: all_ice_e[str(year)+'_'+str(mm)] = []
            
           # composite climatology stuff, newnc=split_winds.nc
           census_path = root_path+'era5_data/original_census/'
           if year >= 2000:
               ncpath_area = root_path+'era5_data/areas/'
           elif year < 2000:
               ncpath_area = root_path+'era5_data/areas/'
           ncadd = '_areas'
           
           ncpath = 'era5_data/seaice/'
           ncadd='_seaice' 
           
           # open census (instead of storm_ranges)
           census_file = census_path+'census_'+str(year)+'.csv'
           [startdate, enddate] = cf.readCensus(census_file, convertDT=True)[0]
           
           timing_grid = []
           for xx in range(0,len(startdate)):
               timing_grid.append((startdate[xx], enddate[xx]))
           
           storm_ranges = []
           analysis_ranges = []
           for startdt, enddt in timing_grid:
               week_ago = startdt - timedelta(days=7)
               two_week = startdt + timedelta(days=14) # relative to start date, since different storm lengths
               analysis_ranges.append(cf.daterange(week_ago, two_week, dt=24))
               storm_ranges.append(cf.daterange(startdt, enddt, dt=24))   
           
           # open ice area
           ncpath_area1 = ncpath_area+str(year)+'/'
           ds_area = xr.open_dataset(ncpath_area1 + str(year) +'_area.nc')
           ice_area80 = ds_area['ice_area80'].values
           box_area = ds_area['box_area'].values
           ice_sorter = ice_area80 
           
           try:
               ds = xr.open_dataset(ncpath + str(year) +ncadd+'.nc')
           except:
               print('- skip: '+ncpath + str(year) +ncadd+'.nc')
               continue
            
           for storm_num, strm in enumerate(storm_ranges):
                month = int(strm[0].month)
                if month not in months: continue
                # get ice area
                app1 = ice_sorter[storm_num]*100/box_area[storm_num]
                if np.isnan(app1) or np.isinf(app1):
                    app1=0
                    
                ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]             
                if (ice_frac<np.min(ice_lims) or ice_frac>np.max(ice_lims)):
                    continue
                    
                ### remove storms that don't interact with the ice
                try:
                    sia = ds['sia_miz'+'2_1000'].values[storm_num]
                except:
                    print('-- sia error')
                    continue
                
                if np.nanmean(sia[0:10+1]) == 0 or np.isnan(np.mean(sia[0:10+1])):
                    # print('no ice: '+stormstr)
                    continue
                
                # get time series                                              
                sia_clim = ds['sia_clim_miz'+'2_1000'].values[storm_num]
                ### relativize
                ss = sia-sia_clim
                standardized_area = (ss-ss[0])/(np.nanmax(ss)-np.nanmin(ss))
                pcd = (standardized_area)
                
                all_ice_e[str(year)+'_'+str(month)].append(pcd) 
                
           try: ds.close()
           except: pass

    return all_ice_e




months = [6,7,8,9]
mcolors = ['#0571b0', '#92c5de','#f4a582','#ca0020']
xxx = np.arange(-7,14+1,1)
labels1 = [-7] +['']*6 + [0] + ['']*6 + [7] + ['']*6 +[14]

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(5.5,10))
for ax in axes:
    ax.axhline(0, lw=0.55, color='gray', ls=':')
    ax.axvline(0, lw=0.55, color='gray', ls=':')


TOTAL_ALL, ens_frac, ens_net = get_spaghetti(normalize = True)

for era in [0,1]:
    
    if era==0: myyears = np.arange(2010, 2019+1)
    elif era==1: myyears = np.arange(1982, 1991+1)
    
    axes[era].set_title(str(myyears[0])+'-'+str(myyears[-1]))
    axes[era].set_ylabel('Normalized Relative Change in MIZ Area')
    
    
    month_lines = {mm:[] for mm in months}
    for ens_total in TOTAL_ALL:
        for mm in months:
            meanline = np.nanmean(ens_total[str(era)+'_'+str(mm)], axis=0)
            if ~np.all(np.isnan(meanline)):
                month_lines[mm].append( meanline )

    for mi, mm in enumerate(months):
        # axes[era].plot(xxx, np.array(month_lines[mm]).T, 
        #                color = mcolors[mi], lw=0.55)        
    
        axes[era].plot(xxx, np.nanmean(month_lines[mm], axis=0), 
                       color = mcolors[mi], lw=3)    
        
for mi, mm in enumerate(months): axes[1].plot([],[], color = mcolors[mi], lw=3, label=calendar.month_name[mm])    
axes[1].legend(ncol=1, fontsize=11,loc='lower left',framealpha=0)

ax.set_xlabel('Days Since Storm Start', fontsize=11)
ax.set_xlim(-7,14)
ax.set_xticks(xxx)
ax.set_xticklabels(labels1, minor=False, rotation=0, fontsize=fontsize)


axes[0].plot([],[], lw=3, ls='-', color='gray', label='CESM2')
axes[0].plot([],[], lw=3, ls='--', color='gray', label='ERA5')
axes[0].legend(ncol=1, fontsize=11,loc='lower left',framealpha=0)


### era
path = root_path+'processed-data/figures/'
era_meanlines = np.load(path+'era_monthly_mean.npy')
for era, years, ax in zip([0,1], [np.arange(2010,2020), np.arange(1982,1992)], axes):
    for mi, mm in enumerate(months):
        axes[era].plot(xxx, era_meanlines[era][mi], 
                       lw=3, color = mcolors[mi], ls='--')
        
'''
Normalized changes in the MIZ ice area relative to the start of the analysis 
time window for each month in (top) 2010–19 and (bottom) 1982–91.
'''

#%% end 
    
    