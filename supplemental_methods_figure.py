#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 2025
Prepared for publication - 08 Aug 2025
supplemental_methods_figure.py

Supplemental Methods Figure (Figure S1)

@author: mundi
"""

root_path = '/Users/mundi/Desktop/cesm-code/'

SAVEFIG = False

#%% starting info
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import xarray as xr
from cartopy.util import add_cyclic_point

import pickle
from datetime import datetime, timedelta
import calendar
import cesm_functions as cf

import gc
gc.collect()

run_rms = False

year = 2017
month = 8
case_dt = datetime(year, month, 5)

offset = 2
plot_date = (case_dt+timedelta(days=offset)).strftime('%Y-%m-%d')

storm_lon_lims = [60,-60]

# sattelite sea ice data path
ice_fname = root_path+ 'seaice/'
# era5 data
slp_file = root_path+'processed-data/supplemental_methods/july_aug_2017_10mwinds_pressure_arctic.nc' 

file_path = root_path+'processed-data/supplemental_methods/' 
cesm_file = file_path+'b.e21.BSSP370smbb.f09_g17.LE2-1231.015.cam.h2.PSL.2015010100-2024123100.nc'
si_file = file_path+'b.e21.BSSP370smbb.f09_g17.LE2-1231.015.cice.h1.aice_d.20150101-20250101.nc'

census_path = root_path+'era5_data/original_census/'
census_name = 'census_'+str(year)+'.csv'
contour_path = census_path + 'cyclone_tracker_out/'
contour_name = '_contours.nc'

grid_path = root_path+'data/'

#%%% functions
def LZ(day):
    ''' get leading zero string'''
    if day>=10:
        return str(day)
    elif day<10:
        return '0'+str(day)
    
def setup_plot2(ax, extent=[-160,90,50,60], title=[], labels=True):
    ax.coastlines('50m',edgecolor='black',linewidth=0.75)
    ax.set_extent(extent, ccrs.PlateCarree())
    try:
        ax.gridlines(draw_labels=labels)
    except:
        print('Unable to create grid lines on map')
    ax.add_feature(cfeature.LAND, facecolor='0.75')
    ax.add_feature(cfeature.LAKES, facecolor='0.85')
    if title:
        if type(title)!= str: title=str(title)
        ax.set_title(title)
    return ax

def plot_geocontour(ax, lon, lat, var, levels, color='k', lw=3, ls='solid', zorder=5, alpha=1):
    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(lon, -0.01)
    lon_lesser = np.ma.masked_less(lon, 0)    
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(lat, mask=lon_greater.mask)
    lat_lesser = np.ma.MaskedArray(lat, mask=lon_lesser.mask)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(var, mask=lon_greater.mask)
    si_lesser = np.ma.MaskedArray(var, mask=lon_lesser.mask)

    # contours
    # si_g_cp, lon_g_cp = add_cyclic_point(si_greater, coord=lon[0,:])
    # si_l_cp, lon_l_cp = add_cyclic_point(si_lesser, coord=lon[0,:])
    
    # ax.contour(lon_g_cp, lat_greater[:,0], si_g_cp, colors=color, levels=levels, 
    #           linewidths = lw, zorder=zorder, transform=ccrs.PlateCarree(),
    #           linestyles=ls, alpha=alpha) 
    # ax.contour(lon_l_cp, lat_lesser[:,0], si_l_cp, colors=color, levels=levels, 
    #           linewidths = lw, zorder=zorder, transform=ccrs.PlateCarree(),
    #           linestyles=ls, alpha=alpha)
    
    ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
              linewidths = lw, zorder=zorder, transform=ccrs.PlateCarree(),
              linestyles=ls, alpha=alpha) 
    ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
              linewidths = lw, zorder=zorder, transform=ccrs.PlateCarree(),
              linestyles=ls, alpha=alpha)
    return ax

#%% --- PRESSURE DIFFS ---

#%% create cesm grid
cesm = xr.open_dataset(cesm_file)
clon = cesm['lon'].values
clon = np.where(clon>180, clon-360, clon)
cesm_lon, cesm_lat = np.meshgrid(clon, cesm['lat'].sel(lat=slice(60,90)).values)
cesm.close()

#%% get era data

with xr.open_dataset(slp_file) as ds:
    try: 
        ds = ds.rename({'valid_time': 'time'})
    except KeyError: pass
    time = ds['time']
    

    slp = ds['msl']/100
    lon, lat = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    
slp1 = slp.sel(time=plot_date).mean(dim='time')
        


### convert to model resolution

TIME = []
for day in list(np.arange(1,calendar.monthrange(year, month)[1]+1)):
    for hr in [0,6,12,18]:
        TIME.append(str(year)+'-'+LZ(month)+"-"+LZ(day))

with open(root_path+'processed-data/supplemental_methods/2010_slp_interp2.pkl', 'rb') as handle:
    MSL = pickle.load(handle)

inds = [idx for idx, tt in enumerate(TIME) if tt==plot_date]
msl1 = np.nanmean(MSL[inds], axis=0)


    
from scipy.interpolate import griddata
msl1 = griddata(np.array([lon.flatten(), lat.flatten()]).T, slp1.values.flatten(),
                    np.array([cesm_lon.flatten(), cesm_lat.flatten()]).T, 
                    method = 'nearest')
msl1 = msl1.reshape(np.shape(cesm_lon))



#%% get contours and storm area
gc.collect()

#%%% functional stuff
def open_cont_nc(ncfile):
    import netCDF4
    # loads pressure contours to form bbox
    out = netCDF4.Dataset(ncfile)
    all_contours = []
    
    # loop thru all the variables
    for v in out.variables:
        all_contours.append(np.array(out.variables[v]))

    return all_contours

def get_bbox_edges(all_contours):
    minlon, minlat, maxlon, maxlat = 999,999,-999,-999

    isDivided = False
    alerted=False
    for cidx, contour in enumerate(all_contours):
        lons = contour[:,0]
        lats = contour[:,1]
        
        ### get rid of boundaries too close to pole
        for li, lat in enumerate(lats):
            if lat>82.5: 
                lons[li]=np.nan
                lats[li]=np.nan
                if not alerted:
                    print('bbox too close too pole; artifical boundary applied (85N)')
                    alerted=True

        ### convert longitude to 0-360 system
        lons1 = lons.copy()
        lons1 = np.where(lons1<0, lons1+360, lons1)
        lons1.sort()
        lons1 = lons1[~np.isnan(lons1)]
        
        if len(lons1) == 0: 
            print('skip')
            continue
        
        ### find e/w lons
        if np.nanmax(lons1) - np.nanmin(lons1) > 180:
            for li, ll in enumerate(lons1):
                if li == len(lons1)-1: break
                if lons1[li+1] - ll > 20: 
                    if not isDivided:
                        eastlon = ll
                        westlon = lons1[li+1]
                        isDivided = True
                        break
                    else:
                        if ll > eastlon:
                            eastlon = ll
                        if lons1[li+1] < westlon:
                            westlon = lons1[li+1]
                        break
        else:
            if lons1[0] < minlon:
                minlon = lons1[0]
                print('minlon ', minlon)
            if lons1[-1] > maxlon:
                maxlon = lons1[-1]
                print('maxlon ', maxlon)
        
        ### get min/max lat
        if np.nanmin(lats) < minlat:
            minlat = np.nanmin(lats)
        if np.nanmax(lats) > maxlat:
            maxlat = np.nanmax(lats)
     
        # end contour loop
        
    if not isDivided:    
        print('done easy: ', [minlon, maxlon, minlat, maxlat])
        return cf.get_edge_lines(minlon, maxlon, minlat, maxlat, reverse=False)
    else: # isDivided
        if minlon == 999:
            print('only divide: ', [westlon, eastlon, minlat, maxlat])
            bbox_edges = cf.get_edge_lines(westlon, eastlon, minlat, maxlat, reverse=True)
            bbox_edges = np.where(bbox_edges>180, bbox_edges-360, bbox_edges)
            return bbox_edges
        else:
            print('combo!', [minlon, maxlon], [westlon, eastlon])
            bbox_edges = cf.get_edge_lines(westlon, maxlon, minlat, maxlat, reverse=True)
            if (eastlon<maxlon) and (westlon<minlon):
                bbox_edges = cf.get_edge_lines(westlon,eastlon, minlat, maxlat, reverse=True)
            bbox_edges = np.where(bbox_edges>180, bbox_edges-360, bbox_edges)
            return bbox_edges


census_path = root_path+'era5_data/original_census/'
contour_path1 = census_path + 'cyclone_tracker_out/'

stormstr = '2017_0805'
ncname = stormstr + '_contours.nc'
all_contours = open_cont_nc(contour_path1+ncname)
bbox_edges_era = get_bbox_edges(all_contours)


#%% --- RMS SEA ICE ---
gc.collect()
#%%% functions
import glob

def load_seaice1(root_dir, year, month, day, hemisphere, latlon=True):
    import glob
    from pyproj import Transformer
    
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

    if not all_files:
        print('Error with filename: ' + root_dir + '*' + year+month+day + '*.nc')
        raise NameError(' bad filename in sip.load_seaice')
        
    cdr_dic = xr.open_dataset(all_files[0])
    seaice = np.squeeze( cdr_dic['nsidc_bt_seaice_conc'].values )
    
    for flag in [251,252,253,254,255]:
        seaice= np.where(seaice==flag/100, np.nan, seaice)
   
    if latlon:
        x,y = np.meshgrid( cdr_dic['xgrid'].values, cdr_dic['ygrid'].values )
        if hemisphere=='n': epsg = "EPSG:3413"
        elif hemisphere=='s': epsg = "EPSG:3412"
        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True) #south, north:3413
        lon, lat = transformer.transform(x, y)
        return seaice, lon, lat
    else:
        return seaice
    
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
    else:
        raise NameError('Check CESM file_year: '+ str(year))

def load_cesm1(si_file, year, hemisphere, lat_bound, ens_mem, grid_path):
    
    ds = xr.open_dataset(si_file)
    ds.close()


    #### grid and final
    grid = xr.open_dataset(grid_path+'pop_grid.nc')

    var_in = 'aice_d'     
    var_to_keep = ds[var_in]

    ds = xr.merge([var_to_keep.drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                   grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                  compat='identical', combine_attrs='no_conflicts')
    grid.close()
    
    # get regional ice
    if hemisphere == 'n':
        return ds.where(ds.TLAT>lat_bound, drop=True)
    elif hemisphere == 's':
        return ds.where(ds.TLAT<-lat_bound, drop=True)
    
#%%% data
gc.collect()
import time as timeIN
from rms_si_fxn import rms_si
hemisphere = 'n'

ice_path = root_path+'data/'

ens_mem = '1231.015'

if run_rms:
    print('* RMS *')
    start_time = timeIN.time()
    
    min_conc_15, min_rms_15 = rms_si(15, ens_mem, 'n', ice_fname, 
                               ice_path, grid_path, conc_spacing=1, years=[])
    min_conc_80, min_rms_80 = rms_si(80, ens_mem, 'n', ice_fname, 
                               ice_path, grid_path, conc_spacing=1, years=[])
    
    print('*', (timeIN.time()-start_time)/60, 'min *')
    
else:
    min_conc_15 = 0
    min_conc_80 = 4

#%% --- PLOT ---
import string
from cmocean.tools import crop_by_percent
gc.collect()

###!!!

cmap1 = cmo.ice
cmap1 = crop_by_percent(cmap1, 20, which='min')

fig = plt.figure(figsize=(8,12))
gs = fig.add_gridspec(4,6, height_ratios=(0.66,0.66,1,1))

axs_e = [fig.add_subplot(gs[0, 2*col:(2*col)+2], projection=ccrs.NorthPolarStereo()) for col in [0,1,2]]
axs_c = [fig.add_subplot(gs[1, 2*col:(2*col)+2], projection=ccrs.NorthPolarStereo()) for col in [0,1,2]]
          
[ax1,ax2] = [ fig.add_subplot(gs[2,0:3], projection=ccrs.NorthPolarStereo()), 
             fig.add_subplot(gs[2,3:6], projection=ccrs.NorthPolarStereo()) ]

ax_m = fig.add_subplot(gs[3,1:5], projection=ccrs.NorthPolarStereo())

alph1 = iter(list(string.ascii_lowercase))
for aa in axs_e + axs_c + [ax1,ax2] +[ax_m]:
    aa.text(0.02, 1.05, '('+next(alph1)+')', transform=aa.transAxes, fontsize=11, 
                                bbox={'facecolor': 'white', 'alpha': 0, 'pad':5, 
                                      'edgecolor':'white', 'lw':0.75},zorder=50)

slp_cb = axs_e[0].pcolormesh(np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), 
                             vmin=980, vmax=1020, 
                             cmap=cmo.haline,
                             transform=ccrs.PlateCarree())
cax1 = fig.add_axes([0.925, 0.5825, .025, 0.3]) 
cbar1 = fig.colorbar(slp_cb, cax=cax1, extend='both', orientation='vertical')
cbar1.ax.tick_params(labelsize=10)
cbar1.set_label('Sea Level Pressure (hPa)', fontsize=10) 

si_cb = ax1.pcolormesh(np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)),
               transform=ccrs.PlateCarree(),cmap=cmap1, vmin=0, vmax=100)
cax2 = fig.add_axes([0.925, 0.35, .025, 0.2]) 
cbar2 = fig.colorbar(si_cb, cax=cax2, orientation='vertical')
cbar2.ax.tick_params(labelsize=10)
cbar2.set_label('Sea Ice Concentration (%)', fontsize=10) 


#%%% function
import matplotlib.ticker as mticker

def setup_plot(ax, extent=[-160,90,50,60], title=[], labels=True):
    ax.coastlines('50m',edgecolor='black',linewidth=0.75)
    ax.set_extent(extent, ccrs.PlateCarree())
    
    LATS = [70,75,80,85]
    LONS = [-120, -60, 0, 60, 120, 180]
    
    gl = ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.75)
    gl.xlocator = mticker.FixedLocator(LONS)
    gl.ylocator = mticker.FixedLocator(LATS)
    # gl.xlabels_top = False
    # gl.xlabels_bottom = False
    
    if labels:
        for LAT in LATS[1:]: 
            ax.text(164, LAT+2, str(LAT)+r'$^\circ$N', fontsize=8,
                    transform=ccrs.PlateCarree(), zorder=999)
        for LON, loc in zip([0,60,120, 180,-120],[83.5,78,78,76.5,75]): 
            ax.text(LON+2, loc, str(LON)+r'$^\circ$', fontsize=8,
                    transform=ccrs.PlateCarree(), zorder=999)
    
    ax.add_feature(cfeature.LAND, facecolor='0.75')
    ax.add_feature(cfeature.LAKES, facecolor='0.85')
    if title:
        if type(title)!= str: title=str(title)
        ax.set_title(title)
    return ax

#%%% STORM AREAS

era_text = 'ERA5'+'\n'+r'$0.25^\circ\times0.25^\circ$'+'\n'+'Hourly'
era_text += '\n'+r'$p_{min} = 984$ hPa'

cesm_text = 'ERA5 Coarse'+'\n'+r'$\sim(1^\circ\times1^\circ)$'+'\n'+'6-Hourly'
cesm_text += '\n'+r'$p_{min} = 986$ hPa'

for loc, text in zip([0.815,0.65], [era_text, cesm_text]):
    fig.text(0.055, loc, text, va='center', ha='center')

# myextent = [-180,180, 75, 90]
myextent = [60, 250, 75, 90]

for ax_group in [axs_e, axs_c]:
    for ax in ax_group:
        setup_plot(ax, extent=myextent, labels=True)

hrs=4
plot_times_era = [timedelta(hours=12+hrs),
              timedelta(hours=18+hrs), timedelta(days=1, hours=0+hrs)]

plot_times_cesm = [timedelta(hours=12), timedelta(hours=18), timedelta(days=1)]

#### era

for axe, dt in zip (axs_e, plot_times_era):
    tt = case_dt + dt
    axe.set_title(tt.strftime('%Y-%m-%d %H:00'), fontsize=11)
    
    slp0 = slp.sel(time = tt.strftime('%Y-%m-%d')).isel(time=tt.hour)
    
    axe.pcolormesh(lon, lat, slp0, vmin=980, vmax=1020, cmap=cmo.haline,
                   transform=ccrs.PlateCarree())
    
    axe.contour(lon, lat, slp0, levels=[984], transform=ccrs.PlateCarree(),
                linewidths=2, colors='m', zorder=500)
    
    slp0cp, lon_cp = add_cyclic_point(slp0, coord=lon[0,:])
    axe.contour(lon_cp, lat[:,0], slp0cp, levels=[1000], transform=ccrs.PlateCarree(),
                linewidths=1.5, colors='#F71735' )
    
    # axe.plot(bbox_edges_era[:,0], bbox_edges_era[:,1], transform=ccrs.PlateCarree(),
    #          lw=2, color='navy')
    
    # for cont in all_contours:
    #     [x1,x2], [y1,y2] = cf.geoplot_2d(cont[:,0], cont[:,1]) 
    #     axe.plot(x1, y1, 'r', linewidth=0.55, transform=ccrs.PlateCarree())
    #     axe.plot(x2, y2, 'r', linewidth=0.55, transform=ccrs.PlateCarree())


    min_slp = np.nanmin(np.where(lat<75, np.nan, slp0))
    minlon = lon[np.where(slp0==min_slp)]
    minlat = lat[np.where(slp0==min_slp)]
    axe.plot(minlon, minlat, color='y', marker='*', markersize=8,
             transform=ccrs.PlateCarree(),
             label=str(round(min_slp, 1))+' hPa')
    axe.legend(loc='lower left', handlelength=1, handletextpad=0.33, fontsize=10)

#### cesm

for axc, dt in zip (axs_c, plot_times_cesm):
    
    tt = case_dt + dt
    axc.set_title(tt.strftime('%Y-%m-%d %H:00'), fontsize=11)
    
    if tt.hour==0:
        sl = [0,6]
    elif tt.hour==6:
        sl = [6,12]
    elif tt.hour==12:
        sl = [12,18]
    elif tt.hour==18:
        sl = [18,24]
    
    slp0 = slp.sel(time = tt.strftime('%Y-%m-%d')).isel(time=slice(sl[0],sl[1])).mean(dim='time')
    
    msl0 = griddata(np.array([lon.flatten(), lat.flatten()]).T, slp0.values.flatten(),
                        np.array([cesm_lon.flatten(), cesm_lat.flatten()]).T, 
                        method = 'nearest')
    msl0= msl0.reshape(np.shape(cesm_lon)) 
    
    alpha_ln1 = 0 # 0.33
    axc.pcolormesh(cesm_lon, cesm_lat, msl0, vmin=980, vmax=1020, cmap=cmo.haline,
                   transform=ccrs.PlateCarree(), linewidths=0.25,
                   edgecolors=(0,0,0,alpha_ln1))

    plot_geocontour(axc, cesm_lon, cesm_lat, msl0, levels=[986], 
                       lw=2, color='m', zorder=500)
    
    plot_geocontour(axc, cesm_lon, cesm_lat, msl0, levels=[1002], 
                       lw=1.5, color='#F71735')
    cesm_lon2 = np.where(cesm_lon<0, cesm_lon+360, cesm_lon)
    plot_geocontour(axc, cesm_lon2, cesm_lat, msl0, levels=[1002], 
                       lw=1.5, color='#F71735')

        
    min_msl = np.nanmin(np.where(cesm_lat<75, np.nan, msl0))
    minlon = cesm_lon[np.where(msl0==min_msl)]
    minlat = cesm_lat[np.where(msl0==min_msl)]
    axc.plot(minlon, minlat, color='y', marker='*', markersize=8,
             transform=ccrs.PlateCarree(),
             label=str(round(min_msl, 1))+' hPa')
    axc.legend(loc='lower left', handlelength=1, handletextpad=0.33, fontsize=9)




#%%% RMS
gc.collect()

c1 = '#F71735' 
c2 = '#D3D643'

sample_day = datetime(2017,8,1)
    
    
title1 = 'Observations '
title2 = 'Model ('+str(ens_mem)+')'

for ax, title in zip([ax1, ax2], [title1, title2]):
    ax.coastlines('50m',edgecolor='black',linewidth=0.75, zorder=27)
    ax.set_extent([0,360, 65, 90], ccrs.PlateCarree())
    
    LATS = [70,75,80,85]
    LONS = [-120, -60, 0, 60, 120, 180]
    gl = ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.75)
    gl.xlocator = mticker.FixedLocator(LONS)
    gl.ylocator = mticker.FixedLocator(LATS)
    
    ax.add_feature(cfeature.LAND, facecolor='0.75', zorder=25)
    ax.add_feature(cfeature.LAKES, facecolor='0.85', zorder=26)
    ax.set_title(title)
    
        
#### OBSERVATIONS
si, si_lon, si_lat = load_seaice1(ice_fname, sample_day.year, sample_day.month, 
                 sample_day.day, hemisphere, latlon=True)
si*=100
ax1.pcolormesh(si_lon, si_lat, np.where(si==0,np.nan,si), 
               transform=ccrs.PlateCarree(),cmap=cmap1, vmin=0, vmax=100)

# plot
ax1 = cf.plot_geocontour(ax1, si_lon, si_lat, si, levels=[15], color=c1, lw=1.25, ls='solid')
ax1 = cf.plot_geocontour(ax1, si_lon, si_lat, si, levels=[80], color=c1, lw=1.25, ls='--')

#### MODEL
ds = load_cesm1(si_file, sample_day.year, hemisphere,60, ens_mem, grid_path)
sample_date = sample_day + timedelta(days=1) # data recorded next day
sic = ds['aice_d'].sel(time=sample_date.strftime('%Y-%m-%d')).squeeze().values*100
tlong, tlat = ds.TLONG.values, ds.TLAT.values
tlong = np.where(tlong>180, tlong-360, tlong)
ax2.pcolormesh(tlong, tlat, np.where(sic==0,np.nan,sic), 
               transform=ccrs.PlateCarree(), cmap=cmap1, vmin=0, vmax=100)

# plot

ax2 = cf.plot_geocontour(ax2, tlong, tlat, sic, levels=[min_conc_15], color=c2, lw=1.5, ls='solid')
ax2 = cf.plot_geocontour(ax2, tlong, tlat, sic, levels=[min_conc_80], color=c2, lw=1.5, ls='--')
 
ax2 = cf.plot_geocontour(ax2, tlong, tlat, sic, levels=[15], color=c1, lw=1.25, ls='-')
ax2 = cf.plot_geocontour(ax2, tlong, tlat, sic, levels=[80], color=c1, lw=1.25, ls='--')

lw1 = 1.5
ax1.plot([],[], ls='-', lw=lw1, color = c1, label = '15%')
ax1.plot([],[], ls='--', lw=lw1, color = c1, label = '80%')
ax1.plot([],[], ls='-', lw=lw1, color = 'white', alpha=0, label = 'RMS')
ax1.plot([],[], ls='-', lw=lw1, color = c2, label = '15%: '+str(int(min_conc_15))+'%')
ax1.plot([],[], ls='--', lw=lw1, color = c2, label = '80%: '+str(int(min_conc_80))+'%')
ax1.legend(loc='lower right', handlelength=1.5, handletextpad=0.33,
           bbox_to_anchor=(-0.05,.25), 
           facecolor='whitesmoke', framealpha=0.5)

ax1.text(-0.475, 0.75, sample_day.strftime('%Y-%m-%d'),
         transform = ax1.transAxes)

#%%% MIZ ICE AREA
gc.collect()

import matplotlib as mpl

ens_mem = '1231.015'

year = 2017
month = 8
dates = (15,17)

case_dt = datetime(year, month, dates[0])
end_dt = datetime(year, month, dates[1])

stormstr = case_dt.strftime('%Y_%m%d')

#%%%% data

grid = xr.open_dataset(grid_path+'pop_grid.nc')
contour_name = '_contours.nc'

si, si_lon, si_lat = cf.load_seaice(si_file, case_dt.year, case_dt.month, case_dt.day, grid, latlon=True)

storm_slp = []
for dt in cf.daterange(case_dt, end_dt, dt=24):
    lon, lat, slp, time = cf.load_psl(cesm_file, year, month, dt.day, daily=True, loc='n')
    
    slp = np.squeeze(slp)
    slp = np.ma.masked_where(np.logical_and(lon>-100, lon<0), slp)
    slp = np.ma.masked_where(np.logical_and(lon>=0, lon<150), slp)
    storm_slp.append(slp)

#### contours, bbox

slp0x = np.ma.masked_where(storm_slp[0]>1002, np.where(lon<0, lon+360, lon)).filled(np.nan)
slp2x = np.ma.masked_where(storm_slp[2]>1002, np.where(lon<0, lon+360, lon))
slp1y = np.ma.masked_where(storm_slp[1]>=1002.66, lat)
# slp0y = np.ma.masked_where(storm_slp[0]>1000, lat)

minlon = np.nanmin(slp0x) #167.53634557
maxlon = np.nanmax(slp2x) #232.9659133

minlat = np.nanmin(slp1y) #70.09713743
maxlat = 85 #np.nanmax(slp0y) #84.18830367

bbox_edges = cf.get_edge_lines(minlon, maxlon, minlat, maxlat, reverse=False)

in_bbox = cf.find_points_in_contour(bbox_edges, si_lon,si_lat)

#%%%% min pressure

abs_min_p = 1000

for dt in cf.daterange(case_dt, end_dt, dt=24):
    lon, lat, SLP, time = cf.load_psl(cesm_file, 2017,8,dt.day, daily=False, loc='n')
    
    for slp in SLP:
        slp = np.ma.masked_where(np.logical_and(lon>-100, lon<0), slp)
        slp = np.ma.masked_where(np.logical_and(lon>=0, lon<150), slp)
        
        
        min_p = np.nanmin(slp)
        min_lon = lon[np.where(slp==min_p)]
        min_lat = lat[np.where(slp==min_p)]
        
        if min_p < abs_min_p:
            abs_min_p = min_p
            lon_min = min_lon
            lat_min = min_lat

#%%%% colors

safe = {
        'sky':'#88CCEE',
        'salmon':'#CC6677',
        'yellow':'#DDCC77',
        'green':'#117733',
        'navy':'#332288',
        'magenta':'#AA4499',
        'turq':'#44AA99',
        'mush':'#999933',
        'purple':'#882255',
        'maroon':'#661100',
        'colonial':'#6699CC',
        'gray':'#888888'
        }

#%%%% plot

ax = ax_m
ax.set_extent([120,260, 67.5, 90], ccrs.PlateCarree())

LATS = [70,75,80,85]
LONS = [-120, -60, 0, 60, 120, 180]
gl = ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.75,zorder=-30)
gl.xlocator = mticker.FixedLocator(LONS)
gl.ylocator = mticker.FixedLocator(LATS)

ax.add_feature(cfeature.LAND, facecolor='0.75', zorder=1)
ax.add_feature(cfeature.LAKES, facecolor='0.85', zorder=2)
ax.coastlines('50m',edgecolor='black',linewidth=0.75, zorder=3)
ax.set_title(str(case_dt).split(' ')[0] + ' ('+ens_mem+')')

# colors ###!!!
c1 = '#F71735' 
c2 = '#D3D643'
cmap1 = cmo.ice
cmap1 = crop_by_percent(cmap1, 20, which='min')
bbox_color =  'k' #safe['navy'] #'k' #'#A01813' #'r'

miz_shade = (safe['salmon'], 0.75) #('#a6cee3', 0.95) #'#33BBEE',
rms_shade = (safe['yellow'],0.5) #(c2, 0.5)

# colors1 = ['#CCBB44','#EE6677','#AA3377']
# colors1 = ['#1b9e77','#d95f02','#7570b3'] # ['#1f78b4','#b2df8a','#33a02c'] # xolor brewer
# colors1 = [safe['salmon'], safe['magenta'], safe['purple']]
# colors1 = [safe['green'], safe['turq'], safe['sky']]
# colors1 = [safe['navy'],safe['navy'],safe['navy']]#['k','black','k']
# styles1 = ['solid', 'densely dashdotted', 'densely dotted']

colors1 = ['#bdbdbd','#525252','#252525']
styles1 = ['-','-','-']


# si contours
ax = cf.plot_geocontour(ax, si_lon, si_lat, si, levels=[0.15], color=c1, lw=1, ls='solid')
ax = cf.plot_geocontour(ax, si_lon, si_lat, si, levels=[0.80], color=c1, lw=1, ls='--')

ax = cf.plot_geocontour(ax, si_lon, si_lat, si, levels=[0], color=c2, lw=1.5, ls='solid')
ax = cf.plot_geocontour(ax, si_lon, si_lat, si, levels=[0.04], color=c2, lw=1.5, ls='--')

ax.pcolormesh(si_lon, si_lat, np.where(si==0,np.nan,si), 
               transform=ccrs.PlateCarree(),cmap=cmap1,alpha=0.66, 
               vmin=0, vmax=1, zorder=-100)


### storm contours    
for si, slp in enumerate(storm_slp):
    ax = cf.plot_geocontour(ax, lon, lat, slp, levels=[1002], 
                            color=colors1[si], lw=2.25, ls=styles1[si])
for col, ls, dt in zip(colors1, styles1, cf.daterange(case_dt, end_dt, dt=24)):
    ax.plot([],[], color=col, lw=2.5, ls=ls, label=dt.strftime('%b %d'))

#### plot accumulated MIZ area 
t1 = case_dt - timedelta(days=1)
t2 = end_dt + timedelta(days=1)
miz_range = cf.daterange(t1, t2, dt=24)

miz_points = np.zeros(np.shape(si_lon))
for miz_dt in miz_range:
    sicm = cf.load_seaice(si_file, miz_dt.year, miz_dt.month, miz_dt.day, grid, latlon=False)
    miz_points = np.where(((sicm>=0.15) & (sicm<=0.80)), 1, miz_points)
    
miz_points_bbox = np.ma.masked_array(miz_points, mask=in_bbox).filled(np.nan)
[mizx1, mizx2], [mizy1, mizy2], [mizz1, mizz2] = cf.geoplot_2d(si_lon, si_lat, miz_points_bbox)

# shade in MIZ
cmiz = mpl.colors.ListedColormap([miz_shade[0]])
ax.pcolormesh(si_lon,si_lat, np.where(miz_points_bbox==0, np.nan, miz_points_bbox), 
              alpha=miz_shade[1],transform=ccrs.PlateCarree(), cmap=cmiz)
 
#### plot rms MIZ 
t1 = case_dt - timedelta(days=1)
t2 = end_dt + timedelta(days=1)
miz_range = cf.daterange(t1, t2, dt=24)

miz_points = np.zeros(np.shape(si_lon))
for miz_dt in miz_range[0:1]: # first day only!
    sicm = cf.load_seaice(si_file, miz_dt.year, miz_dt.month, miz_dt.day, grid, latlon=False)
    miz_points = np.where(((sicm>0) & (sicm<=0.04)), 1, miz_points)
    
miz_points_bbox = np.ma.masked_array(miz_points, mask=in_bbox).filled(np.nan)
[mizx1, mizx2], [mizy1, mizy2], [mizz1, mizz2] = cf.geoplot_2d(si_lon, si_lat, miz_points_bbox)

# shade in MIZ
crms = mpl.colors.ListedColormap([rms_shade[0]])
ax.pcolormesh(si_lon,si_lat, np.where(miz_points_bbox==0, np.nan, miz_points_bbox), 
              alpha=rms_shade[1],transform=ccrs.PlateCarree(), cmap=crms)

#### min pressure location
ax.plot(lon_min, lat_min, marker='*', color='k', markersize=22,
        transform=ccrs.PlateCarree(), zorder=999)
ax.plot(lon_min, lat_min, marker='*', color='y', markersize=16,
        transform=ccrs.PlateCarree(), zorder=1000)

#### plot bbox, legend
bbox_ls = '--' #(0, (5, 1)) # 'solid' #
[bbox1x, bbox2x], [bbox1y, bbox2y] = cf.geoplot_2d(bbox_edges[:,0], bbox_edges[:,1])
ax.plot(bbox1x, bbox1y, color=bbox_color,lw=2.5,ls=bbox_ls,
                transform=ccrs.PlateCarree(), zorder=30)
ax.plot(bbox2x, bbox2y, color=bbox_color,lw=2.5,ls=bbox_ls,
                transform=ccrs.PlateCarree(), zorder=30)

# legend
axshade = ax.twinx()
axshade.tick_params(axis='y', left=False, right=False,labelleft=False, labelright=False)

axshade.plot([],[], color=miz_shade[0], alpha=miz_shade[1], lw=6,
             label = 'MIZ - area where\nice coverage\nis tracked')
axshade.plot([],[], color=rms_shade[0], alpha=rms_shade[1], lw=6,
             label = 'RMS MIZ - used\nto determine\ncyclone interaction\nwith the MIZ')
axshade.legend(loc='upper right', 
               handlelength=1.5, handletextpad=0.5,
               bbox_to_anchor=(-0.025,0.5), 
               facecolor='whitesmoke', framealpha=0.5)

# legend
ax.legend(loc='upper right', title='1002 hPa Daily Contours',
          handlelength=2.5, handletextpad=0.5,
           bbox_to_anchor=(-0.025,1), 
           facecolor='whitesmoke', framealpha=0.5)


# legend

axp = ax.twinx()
axp.tick_params(axis='y', left=False, right=False,labelleft=False, labelright=False)

axp.plot([],[], marker='*', color='y', markersize=12, lw=0, zorder=1000,
        label="Minimum\nPressure:\n"+str(round(abs_min_p,1))+' hPa')

legend = axp.legend(loc='lower right', facecolor=(1, 1, 1, 1))
legend.get_frame().set_alpha(1)
legend.get_frame().set_facecolor((1,1,1,1))

#%% save
gc.collect()

if SAVEFIG:
    
    savepath = root_path+ 'processed-data/supplemental_methods/'
    
    fig.savefig(savepath + 'supplemental_methods.png', bbox_inches="tight")





#%% end
