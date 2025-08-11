#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 2024 Update Thu Apr 24 2025
Prepared for publication - 08 Aug 2025
miz_storms.py

(a) avg MIZ area
(b) number of storms (total)
(c) number of MIZ-interacting storms

in 5º latitude bands (60º-85º)
for each decade? 
with era/nsidc comparison

- add satellite miz trends

@author: mundi
"""

root_path = '/Users/mundi/Desktop/cesm-code/'

SAVEFIG=False

#%% imports and files
if True:
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean.cm as cmo
    import cmocean.tools as cmtools
    from glob import glob
    from datetime import datetime, timedelta
    import calendar
    import string
    import gc
    import sys, os, warnings
    import time as timeIN


data_root = root_path + 'data/'
storm_root = root_path + 'cesm_census/'

data_path = root_path + '/processed-data/miz/'

era_path = root_path+'/era5_data/'

hemi='n'

special_years = [33,35,36,37,38,39,42,43,48,49]
special_years = [yy+2000 for yy in special_years]

cens_name = 'census_test1_all_'
contour_name = '_contours.nc'

import matplotlib.path as mpath
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

miz = [0.15, 0.80]
lat_bands = np.arange(60,85,5)

lat_colors = ['#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']


e_prefix = ['1231','1251','1281','1301'] 
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]


grid = xr.open_dataset(data_root+'pop_grid.nc')

#%%% functions

def file_year(year):
    if year >= 2010 and year < 2015:
        return str(2010)
    elif year >=2015 and year < 2025:
        return str(2015)
    elif year >=1980 and year <1990:
        return str(1980)
    elif year >=1990 and year <2000:
        return str(1990)
    elif year >=2000 and year <2010:
        return str(2000)
    elif year >=2025 and year < 2035:
        return str(2025)
    elif year >=2035 and year<2045:
        return str(2035)
    elif year >=2045 and year<2055:
        return str(2045)
    elif year >=2055 and year<2065:
        return str(2055)
    elif year >=2065 and year<2075:
        return str(2065)
    elif year >=2075 and year<2085:
        return str(2075)
    elif year >=2085 and year<2095:
        return str(2085)
    elif year >=2095 and year<=2100:
        return str(2095)
    else:
        raise NameError('Check CESM file_year: '+ str(year))


def readCensus_str(file, convertDT=True):
    import csv
    
    # Load file
    csv_file = open(file,'r')
    startdate, enddate = [],[]
    startlon, endlon = [], []
    startlat, endlat = [],[]
    pressure = []
    
    # Read off and discard first line, to skip headers
    csv_file.readline()
    
    # Split columns while reading
    for a, b, c, d, e, f, g in csv.reader(csv_file, delimiter=','):
        # Append each variable to a separate list
        try:
            startdate.append(a) 
            startlat.append(float(b))
            startlon.append(float(c))
            pressure.append(float(d))
            enddate.append(e)
            endlat.append(float(f))
            endlon.append(float(g))
        except ValueError:
            # restart lists if double-printed in csv
            startdate, enddate = [],[]
            startlon, endlon = [], []
            startlat, endlat = [],[]
            pressure = []
            continue
            
            
    csv_file.close()
    
    if convertDT:
        startDT, endDT = [],[]                            
        for i, pres in enumerate(pressure):
            startDT.append( datetime(int(startdate[i][:4]), int(startdate[i][5:7]), 
                                           int(startdate[i][8:10]), 0) )
                           # int(startdate[i][-2:]))
            endDT.append( datetime(int(enddate[i][:4]), int(enddate[i][5:7]), 
                                           int(enddate[i][8:10]), 0) )
                            # int(enddate[i][-2:])
                            
        # startdate, enddate = startDT, endDT
        
        ### sort times in ascending order
        startdate, enddate, startlon, startlat, endlon, endlat, pressure = \
            map(list, zip(*sorted(zip(startDT, endDT, startlon, startlat, endlon, endlat, pressure))))

    
    dates = [startdate, enddate]
    coords = [[startlon, startlat],[endlon,endlat]]
    return dates, coords, pressure 

#%% MIZ AREA
savename = 'miz_data_all.nc'

try:
    f = data_path + savename
    miz_data = xr.open_dataarray(f) 
    months = [5,6,7,8,9]
    decades = [np.arange(1982,1992), np.arange(1990,2000),
               np.arange(2000,2010), np.arange(2010,2020),
               np.arange(2020,2030), np.arange(2030,2040),
               np.arange(2040,2050), np.arange(2050,2060),
               np.arange(2060,2070), np.arange(2070,2080),
               np.arange(2080,2090), np.arange(2090,2100)
                ]
    
except FileNotFoundError:
    
    months = [5,6,7,8,9]
    decades = [np.arange(1982,1992), np.arange(1990,2000),
               np.arange(2000,2010), np.arange(2010,2020),
               np.arange(2020,2030), np.arange(2030,2040),
               np.arange(2040,2050), np.arange(2050,2060),
               np.arange(2060,2070), np.arange(2070,2080),
               np.arange(2080,2090), np.arange(2090,2100)
                ]

    areas = {}
    for month in months:
        print('***', month , '***')
        month_start_time = timeIN.time()
        areas[month] = []
        
        for years in decades:
            print(str(years[0])+'-'+str(years[-1]))
            year_list = []
            for year in years:
                ens_list = []
                for ens_mem in ensemble_members:
                    ### open sea ice file
                    si_file = glob(data_root+'aice_d/*'+ens_mem+'*.'+file_year(year)+'*.nc')[0]
                    with xr.open_dataset(si_file) as ds_si:
                        # open grid info
                        TAREA = grid.TAREA.values/1e6
                        ds_si = xr.merge([ds_si['aice_d'].drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                                       grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                                      compat='identical', combine_attrs='no_conflicts')
                        
                        # get vars
                        si_lon = ds_si['TLONG'].values
                        si_lat = ds_si['TLAT'].values
                        seaice = ds_si['aice_d']
                        del ds_si
                        
                    ### isolate daily miz
                    miz_area = []
                    for day in np.arange(1,calendar.monthrange(year, month)[-1]+1):
                        si = seaice.sel(time=datetime(year,month,day).strftime('%Y-%m-%d')).squeeze()
                        si = np.where(si_lat<55, np.nan, si) # arctic!
                        si_extent = np.where(si<0.15, np.nan, si)
                        si_miz = np.where(si>0.80, np.nan, si_extent)
                        
                        ### latitude bands
                        miz_bands = []
                        for lat1 in lat_bands:
                            miz_band = np.where(np.logical_and(si_lat>=lat1, si_lat<lat1+5), si_miz, np.nan)
                            miz_bands.append( np.nansum(miz_band*TAREA) )
                        miz_bands.append(np.nansum(si_miz*TAREA)) # total miz area too
                    
                        miz_area.append(miz_bands)
                        
                    while day < 31:
                        miz_area.append(np.nan*np.array(miz_bands))
                        day += 1
                        
                    
                    ens_list.append(miz_area)
                year_list.append(ens_list)
            areas[month].append(year_list)  
        print('> '+str(round((timeIN.time()-month_start_time)/60,1))+' min')
        print()
        
                
    #### data   
    areas_array=[]
    for mm in months:
        areas_array.append(areas[mm])     
    
    miz_data = xr.DataArray(
                    areas_array,
                    coords={
                        "months": months,
                        "decades": np.arange(0,len(decades)),
                        "years": np.arange(0,10),
                        "ensemble_members": ensemble_members,
                        "days": np.arange(1,31+1),
                        "lat_bands":list(lat_bands)+['total']
                    },
                    dims=["months","decades","years","ensemble_members","days","lat_bands"],
                )        
    miz_data.to_netcdf(path=data_path+savename, mode='w')




#%% STORM COUNTS

dec_total, dec_good = [],[]
for years in decades:
    
    
    year_total, year_good = [],[]
    
    for year in years:
        ens_total = []
        ens_good = []
        
        startdate, startlat = [],[]
        enddate, endlat = [],[]
        
        for ens_mem in ensemble_members:
            ens_dir = 'out_'+hemi+'_'+ens_mem+'/'
            
            month_total = [[0]*len(months) for _ in range(len(lat_bands))]
            month_good = [[0]*len(months) for _ in range(len(lat_bands))]
            
            try:
                [startdate, enddate], [[startlon, startlat],[endlon,endlat]] = \
                    readCensus_str(storm_root+ens_dir+cens_name+str(year)+'.csv')[0:2]
            except (ValueError, FileNotFoundError):
                startdate, enddate = [],[]
                print('= '+ens_dir+cens_name+str(year)+'.csv')
            try:
                [startdate5, enddate5], [[startlon5, startlat5],[endlon5,endlat5]] = \
                    readCensus_str(storm_root+'may2/'+cens_name+str(year)+'.csv')[0:2]
            except (ValueError, FileNotFoundError):
                startdate5, enddate5, startlat5, endlat5 = [],[],[],[]
                
            startdate+=startdate5; enddate+=enddate5
            startlat+=startlat5; endlat+=endlat5
                
            try:
                [removed_start, removed_end] = readCensus_str(storm_root+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')[0]
            except FileNotFoundError:
                removed_start, removed_end = [],[]
                # print('- '+sens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')
            try:
                [removed_start5, removed_end5] = readCensus_str(storm_root+'may2/'+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')[0]
            except FileNotFoundError:
                removed_start5, removed_end5 = [],[]
                
            if 5 not in np.unique([sd.month for sd in startdate]):
                removed_start+=removed_start5; removed_end+=removed_end5
                
            storm_loc = np.nanmean([startlat,endlat], axis=0)
            
            for sd, sl in zip(startdate, storm_loc):
                if sd.month not in months: continue
                for li, lat1 in enumerate(lat_bands):
                    if (sl>=lat1 and sl<(lat1+5)):
                        month_total[li][sd.month - months[0]] += 1
                        
                        if sd not in removed_start:
                            month_good[li][sd.month - months[0]] += 1
                            
                        break # dont need to loop through rest of latbnds
                        
                
            ens_total.append(month_total)
            ens_good.append(month_good)
            # [[sd,sl] for sd, sl in zip(startdate, storm_loc) if sd not in removed_start]

        year_total.append(ens_total)
        year_good.append(ens_good)

    dec_total.append(year_total)
    dec_good.append(year_good)
    
#%% ERA COUNTS

era_root = era_path + 'original_census/'
era_census_name = 'census_'
may_root = era_path + 'cyclone_tracker_out_may/'
may_area_path = may_root + 'area/'

era_decades = decades[0:4]

era_total, era_good = [],[]
for years in era_decades:
    
    year_total, year_good = [],[]
    
    for year in years:
        month_total = [[0]*len(months) for _ in range(len(lat_bands))]
        month_good = [[0]*len(months) for _ in range(len(lat_bands))]
        
        ### [6,7,8,9]
        
        try:
            if year==1990 or year==1991: raise FileNotFoundError
            [startdate, enddate],[[startlon, startlat],[endlon,endlat]] = \
                readCensus_str(era_root+era_census_name+str(year)+'.csv')[0:2]
        except FileNotFoundError:
            census_dec_path = era_path+'decades/census/'
            census_file = census_dec_path+'census_'+str(year)+'.csv'
            [startdate, enddate],[[startlon, startlat],[endlon,endlat]] = \
                readCensus_str(census_file)[0:2]
            
        storm_loc = np.nanmean([startlat,endlat], axis=0)
        
        if year in np.array(decades[1:3]).flatten():
            areapath = era_path + 'decades/areas/'
        elif year >= 2000:
            areapath = era_path+ 'areas/'
        elif year < 2000:
            areapath = era_path+ 'areas/'
        
        
        for storm_num, sd in enumerate(startdate):
            if sd.month not in months: continue
            sl = storm_loc[storm_num]
        
            ## load ice areas
            ds_area = xr.open_dataset(areapath + str(year) +'_area.nc')
            ice_area80 = ds_area['ice_area80'].values
            box_area = ds_area['box_area'].values
            ice_sorter = ice_area80 
            
            ## sort by ice area [20,80]
            ice_lims = [20,80]
            ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]
            if (ice_frac<np.min(ice_lims)) or (ice_frac>np.max(ice_lims)):
                removed=True
            else: removed = False
        
            for li, lat1 in enumerate(lat_bands):
                if (sl>=lat1 and sl<(lat1+5)):
                    month_total[li][sd.month - months[0]] += 1
                    
                    if not removed:
                        month_good[li][sd.month - months[0]] += 1
                        
                    break # dont need to loop through rest of latbnds
                    
        ### repeat for [5]
        try:
            [startdate, enddate],[[startlon, startlat],[endlon,endlat]] = \
                readCensus_str(may_root+era_census_name+str(year)+'.csv')[0:2]
        except (FileNotFoundError, ValueError):
            startdate = []
        
        storm_loc = np.nanmean([startlat,endlat], axis=0)
        
        for storm_num, sd in enumerate(startdate):
            if sd.month not in months: continue
            sl = storm_loc[storm_num]
        
            ## load ice areas
            ds_area = xr.open_dataset(may_area_path + str(year) +'_area.nc')
            ice_sorter = ds_area['ice_area80'].values
            
            ## sort by ice area [20,80]
            ice_lims = [20,80]
            ice_frac = ice_sorter[storm_num]*100/box_area[storm_num]
            if (ice_frac<np.min(ice_lims)) or (ice_frac>np.max(ice_lims)):
                removed=True
            else: removed = False
        
            for li, lat1 in enumerate(lat_bands):
                if (sl>=lat1 and sl<(lat1+5)):
                    month_total[li][sd.month - months[0]] += 1
                    
                    if not removed:
                        month_good[li][sd.month - months[0]] += 1
                        
                    break # dont need to loop through rest of latbnds
        
        ### append!
        year_total.append(month_total)
        year_good.append(month_good)

    era_total.append(year_total)
    era_good.append(year_good)


        
#%%% statistics!
from scipy.stats import linregress

def statistic_str(x,y, pval = True, m_n=2, p_n=2):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if m_n > 0:
        slope_str = str(round(slope,m_n))   
    else:
        slope_str = str(int(round(slope,m_n)))   
     
    if p_n > 0:
        pval_str = str(round(p_value,p_n))
    else:
        pval_str = str(int(round(p_value,p_n)))

    if pval:
        return 'm='+slope_str+', p='+pval_str
    else:
        return 'm='+slope_str



#%% ---

#%% 2D PLOT (Fig 3)

#%%% organize data

decade_miz = xr.DataArray(
                dec_good,
                coords={
                    "decades": np.arange(0,len(decades)),
                    "years": np.arange(0,10),
                    "ensemble_members": ensemble_members,
                    "lat_bands":list(lat_bands),
                    "months": months
                },
                dims=["decades","years","ensemble_members","lat_bands","months"],
            ) 

decade_total = xr.DataArray(
                dec_total,
                coords={
                    "decades": np.arange(0,len(decades)),
                    "years": np.arange(0,10),
                    "ensemble_members": ensemble_members,
                    "lat_bands":list(lat_bands),
                    "months": months
                },
                dims=["decades","years","ensemble_members","lat_bands","months"],
            )        

miz_avg = miz_data.mean(dim = ('years','ensemble_members','days')).isel(lat_bands=[0,1,2,3,4]) 
# (month, dec, lats)  # remove 'total'


#%%% combined plot

VMIN = 0
cmap = cmo.haline
fontsize=12  

nrow = 5 # number of plots         

cb_width = 0.02
cb_height = 0.125

decade_spacing = 0.35
                            # spcing between
fig111, axes = plt.subplots(nrow+3, 1, figsize=(12,3.15*nrow), height_ratios=(1,0.03,1,1,0.025,0.75,0.15,0.75))

titles = ['Total', 'MIZ-Interacting']
plt.subplots_adjust(hspace=0.35)

#### MIZ
MIZ_MAX = 7
expo = 5

# axes[2].axis("off")
axes[1].axis("off")

ax = axes[0] # axes[3]
ax.set_title('MIZ Ice Area', fontsize=fontsize+4, fontweight='bold')

x1 = 0
y = [int(y1) for y1 in miz_avg.lat_bands.values]
xticks = []
xmin = []
for dec in miz_data.decades:
    x = np.arange(x1, x1+len(miz_data.months))
    x1 = x[-1]+1+decade_spacing # blank space
    xmin += list(x)
    xticks.append(np.mean(x))
    
    xx, yy = np.meshgrid(list(x), list(y))

    
    pcm = ax.pcolormesh(xx, yy+2.5, miz_avg.sel(decades=dec).T/(10**expo), 
                        vmin=VMIN, vmax=MIZ_MAX, cmap=cmap)

ax.set_yticks(y+[85])
ax.set_yticklabels(y+[85], fontsize=fontsize);

ax.set_xticks(xmin)
scount = decade_miz
ax.set_xticklabels([calendar.month_name[mm][0] for dd in scount.decades.values for mm in scount.months.values],
                   fontsize=fontsize-2);

ax.set_ylabel('Latitude Band', fontsize=fontsize);

cax1 = fig111.add_axes([0.925,0.75, cb_width, cb_height]) 
cbar1 = fig111.colorbar(pcm, cax=cax1, orientation='vertical')
cbar1.set_label(r'MIZ Ice Area ($\times 10^{}$ km$^2$)'.format(expo), fontsize=fontsize)

#### STORM COUNTS
idx = -1
storm_axes = axes[2:4] 
for ax, scount, title, VMAX, yi in zip(storm_axes, [decade_total, decade_miz], titles, [18,10],[0.575,0.425]):
    idx += 1
    ax.set_title(title, fontsize=fontsize+2)
    
    scount = scount.sum(dim='years').mean(dim='ensemble_members')
    
    x1 = 0
    y = [int(y1) for y1 in scount.lat_bands.values]
    xticks = []
    xmin = []
    for dec in scount.decades:
        x = np.arange(x1, x1+len(scount.months))
        x1 = x[-1]+1+decade_spacing # blank space
        xmin += list(x)
        xticks.append(np.mean(x))
        
        xx, yy = np.meshgrid(list(x), list(y))

        
        pcm = ax.pcolormesh(xx, yy+2.5, scount.sel(decades=dec), 
                            vmin=VMIN, vmax=VMAX, cmap=cmap)
        print('maximim count '+str(idx)+': '+str(np.max(scount.sel(decades=dec).values)))

        
    if idx==0:
        ax.set_xticks(xticks)
        ax.set_xticklabels(['']*len(list(miz_data.decades.values)),
                           fontsize=fontsize);
        ax.set_xticks(xmin, minor=True);
        
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(decades[di][0])+'-\n'+str(decades[di][-1]) 
                            for di in list(miz_data.decades.values)],
                           fontsize=fontsize);
        ax.set_xticks(xmin, minor=True);

    ax.set_yticks(y+[85])
    ax.set_yticklabels(y+[85], fontsize=fontsize);
    
    ax.set_ylabel('Latitude Band', fontsize=fontsize);

    cax1 = fig111.add_axes([0.925,yi,cb_width, cb_height]) 
    cbar1 = fig111.colorbar(pcm, cax=cax1, orientation='vertical', extend='max')
    cbar1.set_label('Number of Storms', fontsize=fontsize)

axes[1].text(0.405, -1.5, 'Storm Counts', fontsize=fontsize+4, fontweight='bold')

##############
#### LINES
##############

# COLOR1 = 'dimgray'
# COLOR2 = cmap(0)

#88CCEE,#CC6677,#DDCC77,#117733,#332288,#AA4499,#44AA99,#999933,#882255,#661100,#6699CC,#888888
COLOR1 = '#117733'
COLOR2 = '#332288'

# COLOR1 = cmap(0.7)
# COLOR2 = cmap(0)

COLOR1 = '#8c510a'
COLOR2 = '#01665e'


axes[4].axis("off")
axl = axes[5]
axl.set_ylim([0,225])
axl.tick_params(axis='y', colors=COLOR2)
lw = 2

axl.set_title(r'All Storm Counts and MIZ Ice Area North of 60$^\circ$', 
              fontsize=fontsize+2, fontweight='bold')


for var, title, ls in zip([[dec_good,era_good], [dec_total,era_total]], 
                      ["MIZ-Interacting Storms", "Total Storms"], [(0, (5, 1)),'-']):
    era_marker = '.'
    
    mean_var = np.nanmean(np.nansum(var[0], axis=(1,-1)), axis=1)
    mean_era = np.nansum(var[1], axis=(1,-1))
    
    axl.plot(np.arange(0,len(decades)), np.nansum(mean_var[:, :],axis=1), 
                  color=COLOR2,ls=ls, lw=lw)
    axl.plot(np.arange(0,len(era_decades)), np.nansum(mean_era[:, :],axis=1), 
                  color=COLOR2,ls=ls, lw=lw, marker = era_marker)
    
    axl.plot([],[], color='k', alpha=0.85,ls=ls, lw=lw, label=title)


# add miz lines to above !


miz_color = COLOR1
miz_ls = '-'
miz_marker = 'x'


miz_ax = axl.twinx()
miz_series = miz_data.mean(dim = ('months','years','ensemble_members','days','lat_bands'))
miz_ax.plot(np.arange(0,len(decades)), miz_series, 
            color=miz_color, ls=miz_ls, lw=lw, marker=miz_marker)
miz_ax.set_ylim([0,600000])
miz_ax.set_yticks(np.array([200,400,600])*1e3)
miz_ax.set_yticklabels(['200','400','600'], color = miz_color,fontsize=fontsize)
miz_ax.set_ylabel(r'Area ($\times 10^5$ km$^2$)', color = miz_color,fontsize=fontsize)


#### legend and etc.
xlabs = [str(yrs[0])+'-\n'+str(yrs[-1]) for yrs in decades]
axl.set_xticks(np.arange(0,len(decades)))
axl.set_xticklabels(xlabs, fontsize=fontsize, rotation=0)
axl.set_ylabel('Storm Counts',color=COLOR2,fontsize=fontsize)
    
axl.plot([],[], color='k', ls='-', lw=lw, marker=era_marker, label='ERA5')
axl.plot([],[], color=miz_color, ls=miz_ls, lw=lw, marker=miz_marker, label='MIZ Area')
axl.legend(ncol=6, bbox_to_anchor=(0.85,-0.275), fontsize=fontsize)


slopestr = statistic_str(np.arange(0,len(decades)), miz_series, pval = False, m_n=0)
legend_ax = axl.twinx()
legend_ax.axis("off")
legend_ax.plot([],[], color=miz_color, ls=miz_ls, lw=lw, marker=miz_marker,
               label = slopestr + r' km$^2$ per decade')
legend_ax.legend(loc='lower left', fontsize=fontsize);


#### SATELLITE MIZ
axes[6].axis("off")
ax8 = axes[7]
ax9 = ax8.twinx()
ax8.tick_params(axis='y', colors=COLOR2)
ax9.tick_params(axis='y', colors=COLOR1)
lw = 2
ax8.set_xlim([1980,2021])
ax9.set_xlim([1980,2021])

years8 = np.arange(1982,2020)
yr8a = np.arange(1981,2021)
months8 = np.arange(1,12+1)
plot_months = [5,6,7,8,9]

ax8.set_title(r'     Summer Arctic MIZ has narrowed and shifted northward over the satellite record', 
              fontsize=fontsize+2, fontweight='bold')
ax8.tick_params(axis='both', which='major', labelsize=fontsize)
ax9.tick_params(axis='both', which='major', labelsize=fontsize)

ax8.set_ylabel(r'Mean MIZ Latitude ($^\circ$N)', color=COLOR2, fontsize=fontsize)
miz_lats = np.load(data_path+'miz_storms2_lats.npy') 
miz_iter = iter(miz_lats)
annual_data = []
for year in years8:
    month_data = []
    for month in months8:
        val = next(miz_iter)
        if month in plot_months: month_data.append(val)
    annual_data.append(np.nanmean(month_data))
slope, intercept, r, p, se = linregress(years8, annual_data)
ax8.plot(years8, np.array(annual_data), color=COLOR2, lw=lw,
         label=str(round(slope,3))+r' deg yr$^{-1}$')
ax8.plot(yr8a, (slope*yr8a)+intercept, ls='--', color=COLOR2, lw=1.5)


ax9.set_ylabel(r'MIZ Area ($\times10^5$ km$^2$)', color=COLOR1, fontsize=fontsize)
miz_areas = np.load(data_path+'miz_storms2_area.npy')
miz_iter = iter(miz_areas)
annual_data = []
for year in years8:
    month_data = []
    for month in months8:
        val = next(miz_iter)
        if month in plot_months: month_data.append(val)
    annual_data.append(np.nanmean(month_data))
slope, intercept, r, p, se = linregress(years8, annual_data)
ax9.plot(years8, np.array(annual_data)/1e5, color=COLOR1, lw=lw,
         label=str(round(slope,3))+r' deg yr$^{-1}$')
ax9.plot(yr8a, ((slope*yr8a)+intercept)/1e5, ls='--', color=COLOR1, lw=1.5)


#### letters and save

alph1 = iter(list(string.ascii_lowercase))
for aa in [ axes[0],  storm_axes[0], storm_axes[1], axl, ax8 ]:
    aa.text(0.005, 1.05, '('+next(alph1)+')', transform=aa.transAxes, fontsize=fontsize, 
                                bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                      'edgecolor':'white', 'lw':0.75},zorder=50)
if SAVEFIG:
    fig111.savefig(data_path+'miz_storms_combined.pdf')


#%%% count trends 
from scipy import stats

decade1 = decade_total.sel(months=slice(6,9))
counts1 = decade1.sum(dim=('years','months','lat_bands')).mean(dim='ensemble_members')

slope1, intercept, r, p, se = stats.linregress(counts1.decades, counts1.values)
# plt.plot(counts1.decades, counts1.values)
print(round(slope1,2),'total storms per decade')

decade2 = decade_miz.sel(months=slice(6,9))
counts2 = decade2.sum(dim=('years','months','lat_bands')).mean(dim='ensemble_members')

slope2, intercept, r, p, se = stats.linregress(counts2.decades, counts2.values)

# plt.plot(counts2.decades, counts2.values)
print(round(slope2,2),'MIZ storms per decade')


#%% end