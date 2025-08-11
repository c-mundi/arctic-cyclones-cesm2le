#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 2024
Prepared for publication - 08 Aug 2025
spatial_changes.py

*** new calculations
-> adds line plots

Figure 4
Figure S3 in supplemental materials

@author: mundi
"""
root_path = '/Users/mundi/Desktop/cesm-code/'

SAVEFIG=False
savepath111 = root_path + 'processed-data/spatial/'

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
    from matplotlib.gridspec import GridSpec


import faulthandler
faulthandler.enable()

plot_total=True

months = [5,6,7,8,9] #


decades = [np.arange(1982,1992),
            np.arange(2020,2030),
            np.arange(2060,2070),
            np.arange(2090,2100)
            ]


decade_cmap = cmo.delta #cmtools.crop_by_percent(cmo.rain, 10, which='max')
decade_colors = [decade_cmap(i) for i in [0.5,0.625,0.4,0.35]]

decade_colors =['white', decade_colors[1], 'white','white']
decade_colors =['white', 'white', 'white','white']

decade_names = ['Early Satellite\nEra\n', 'Present Day\n', 'Near Future\n', 'Far Future\n']
if len(decades)!=len(decade_names):
    decade_names = [str(yy[0])+'-'+str(yy[-1])+'\n' for yy in decades]

data_root = root_path+'data/'
storm_root = root_path+'cesm_census/'
savepath = root_path+'processed-data/spatial/'
save_output = False

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

CMAP = cmo.balance_r

e_prefix = ['1231','1251','1281','1301'] 
e_range = np.arange(11,20+1,1)
# e_range = np.arange(11,15)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]

VMAX = 500#00 #7e3
VMIN = -VMAX

plot_indiv = False

plot_ice_contour = True
plot_cyc_density = True

grid = xr.open_dataset(data_root+'pop_grid.nc')
TAREA = grid.TAREA.values
grid.close()

si_lon = np.load(savepath+'si_lon.npy')
si_lat = np.load(savepath+'si_lat.npy')

#%%% functions
def LZ(day):
    ''' get leading zero string'''
    if day>=10:
        return str(day)
    elif day<10:
        return '0'+str(day)

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

def daterange(start_date, end_date, dt=6):
    alldates=[]
    delta = timedelta(hours=dt)
    while start_date <= end_date:
        alldates.append(start_date)
        start_date += delta
    return alldates

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

def get_clim_years(year):
    if year in np.arange(2010,2019+1,1):
        return np.arange(2010,2019+1,1)
    elif year in np.arange(1982,1991+1):
        return np.arange(1982,1991+1)
    elif year in np.arange(2000,2009+1,1):
        return np.arange(2000,2009+1,1)
    elif year in np.arange(1990,1999+1,1):
        return np.arange(1990,1999+1,1)
    elif year in np.arange(2020, 2030):
        return np.arange(2020, 2030)
    elif year in np.arange(2030, 2040):
        return np.arange(2030, 2040)
    elif year in np.arange(2040, 2050):
        return np.arange(2040, 2050)
    elif year in np.arange(2050, 2060):
        return np.arange(2050, 2060)
    elif year in np.arange(2060, 2070):
        return np.arange(2060, 2070)
    elif year in np.arange(2070, 2080):
        return np.arange(2070, 2080)
    elif year in np.arange(2080, 2090):
        return np.arange(2080, 2090)
    elif year in np.arange(2090, 2100):
        return np.arange(2090, 2100)
        
     
#%%%% storm area
class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_edge_lines(minlon, maxlon, minlat, maxlat, n=90, reverse=False):
    ### create new bbox edges
    #
    edge1x = np.linspace(minlon, minlon, n)
    edge1y = np.linspace(minlat, maxlat, n)
    edge2x = np.linspace(minlon, maxlon, n)
    edge2y = np.linspace(maxlat, maxlat, n)
    edge3x = np.linspace(maxlon, maxlon, n)
    edge3y = np.linspace(maxlat, minlat, n)
    edge4x = np.linspace(maxlon, minlon, n)
    edge4y = np.linspace(minlat, minlat, n)
    if reverse:
        edge2x = np.concatenate((np.linspace(minlon,0,round(n/3)),np.linspace(0,180,round(n/3))))
        edge2x = np.concatenate( ( edge2x,np.linspace(-180,maxlon,round(n/3)) ) ) 
        #
        edge4x = np.concatenate((np.linspace(maxlon,-180,round(n/3)),np.linspace(180,0,round(n/3))))
        edge4x = np.concatenate( ( edge4x,np.linspace(0,minlon,round(n/3)) ) ) 
    #
    bbox_lon = np.append(edge1x, edge2x)
    bbox_lon = np.append(bbox_lon, edge3x)
    bbox_lon = np.append(bbox_lon, edge4x)
    #
    bbox_lat = np.append(edge1y, edge2y)
    bbox_lat = np.append(bbox_lat, edge3y)
    bbox_lat = np.append(bbox_lat, edge4y)
    #
    bbox_edges = np.squeeze(np.array([[bbox_lon],[bbox_lat]])).T
    
    return bbox_edges

def get_bbox_edges(all_contours, hemisphere):
    minlon, minlat, maxlon, maxlat = 999,999,-999,-999
    
    isDivided = False
    alerted=False
    for cidx, contour in enumerate(all_contours):
        lons = contour[:,0]
        lats = contour[:,1]
        
        ### get rid of boundaries too close to pole
        for li, lat in enumerate(lats):
            if hemisphere == 'n': boo = lat > 85
            elif hemisphere == 's': boo = lat < -85
            
            if boo: 
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
        return get_edge_lines(minlon, maxlon, minlat, maxlat, reverse=False)
    else: # isDivided
        if minlon == 999:
            print('only divide: ', [westlon, eastlon, minlat, maxlat])
            bbox_edges = get_edge_lines(westlon, eastlon, minlat, maxlat, reverse=True)
            bbox_edges = np.where(bbox_edges>180, bbox_edges-360, bbox_edges)
            return bbox_edges
        else:
            print('combo!', [minlon, maxlon], [westlon, eastlon])
            bbox_edges = get_edge_lines(westlon, maxlon, minlat, maxlat, reverse=True)
            if (eastlon<maxlon) and (westlon<minlon):
                bbox_edges = get_edge_lines(westlon,eastlon, minlat, maxlat, reverse=True)
            bbox_edges = np.where(bbox_edges>180, bbox_edges-360, bbox_edges)
            return bbox_edges

def find_points_in_contour(coords, var_x, var_y, var=None):
    import matplotlib.path as mpltPath
    coords = np.where(coords<0, coords+360, coords)
    
    ### turn contour into polygon
    polygon = np.vstack((coords[:,0], coords[:,1])).T
    path = mpltPath.Path(polygon)
    
    points = np.vstack((var_x.flatten(), var_y.flatten()))
    points=points.T
    
    ### get inside points and plot
    inside = path.contains_points(points)
    inside = np.reshape(inside, np.shape(var_x))
    
    if np.nanmin(var_x) < 0:
        var_x2 = np.where(var_x < 0, var_x+360, var_x)
    else:
        var_x2 = np.where(var_x > 180, var_x-360, var_x)
    points2 = np.vstack((var_x2.flatten(), var_y.flatten())).T
    inside2  = path.contains_points(points2)
    inside2 = np.reshape(inside2, np.shape(var_x))
    
    inside_points = np.invert(np.logical_or(inside, inside2))

    return inside_points

#%%%% plot
def setup_plot(ax, extent=[-160,90,50,60], title=[], labels=True):
    ax.coastlines('50m',edgecolor='black',linewidth=0.75)
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='0.75')
    ax.add_feature(cfeature.LAKES, facecolor='0.85')
    if title:
        if type(title)!= str: title=str(title)
        ax.set_title(title)
    return ax

def seaicecontour(ax, seaice, si_lon, label=[], linestyle='-', \
                  color='k', linewidth=2, levels=[0.15], zorder=10):
    
    si_lon = np.where(si_lon>180, si_lon-360, si_lon)

    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(si_lon, -0.01)
    lon_lesser = np.ma.masked_less(si_lon, 0)
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(si_lat, mask=lon_greater.mask).filled(np.nan)
    lat_lesser = np.ma.MaskedArray(si_lat, mask=lon_lesser.mask).filled(np.nan)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(seaice, mask=lon_greater.mask).filled(np.nan)
    si_lesser = np.ma.MaskedArray(seaice, mask=lon_lesser.mask).filled(np.nan)
    
    lon_greater=lon_greater.filled(np.nan)
    lon_lesser=lon_lesser.filled(np.nan)

    # contours
    levels = levels # 15% ice extent definition
    contours1 = ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
                  linewidths = linewidth, zorder=zorder,linestyles=linestyle, 
                  transform=ccrs.PlateCarree()) 
    contours2 = ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
                  linewidths = linewidth, zorder=zorder,linestyles=linestyle, 
                  transform=ccrs.PlateCarree())
    return ax, [contours1,contours2]

def plot_geocontour(ax, lon, lat, var, levels, color='k', lw=3, ls='solid',label=False):
    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(lon, -0.01)
    lon_lesser = np.ma.masked_less(lon, 0)    
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(lat, mask=lon_greater.mask)
    lat_lesser = np.ma.MaskedArray(lat, mask=lon_lesser.mask)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(var, mask=lon_greater.mask)
    si_lesser = np.ma.MaskedArray(var, mask=lon_lesser.mask)
    
    si_greater = np.ma.masked_where(lat<58, si_greater)
    si_lesser = np.ma.masked_where(lat<58, si_lesser)

    # contours
    cs1 = ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
              linewidths = lw, zorder=10, transform=ccrs.PlateCarree(),
              linestyles=ls) 
    cs2 = ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
              linewidths = lw, zorder=10, transform=ccrs.PlateCarree(),
              linestyles=ls)
    
    if label:
        ax.clabel(cs1, inline=True, fontsize=10)
        ax.clabel(cs2, inline=True, fontsize=10)
    
    return ax

#%%%% analysis 

def get_subdir(year):
    if year >= 2050 or year in special_years:
        subdir = ''
    elif year > 2019:
        subdir = ''
    elif year in list(np.arange(1990,2000))+list(np.arange(2000,2010)):
        subdir = ''
    else: 
        subdir = ''
        
    if month==5 and (year in np.arange(1980,2050)):
        subdir = 'may2/'
        
    return subdir

def get_storm_dates(subdir, ens_dir, month):
    try:
        [startdate, enddate] = readCensus_str(storm_root+subdir+ens_dir+cens_name+str(year)+'.csv')[0]
    except ValueError:
        print('= '+subdir+ens_dir+cens_name+str(year)+'.csv')
        return [],[],[] 

    enddate = [ed for i, ed in enumerate(enddate) if startdate[i].month==month]
    startdate = [sd for sd in startdate if sd.month==month]

    try:
        [removed_start, removed_end] = readCensus_str(storm_root+subdir+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')[0]
    except FileNotFoundError:
        removed_start = []
        print('- '+subdir+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')
    
    #### get storm areas
    storm_areas = []
    stormstr_prev=''
    dupe = iter(list(string.ascii_lowercase))
    for sx, sd in enumerate(startdate):
        
        if sd in removed_start: # check miz interaction, remove storm from analysis
            startdate.pop(sx)
            enddate.pop(sx)
            continue
        
        stormstr1 = sd.strftime('%Y_%m%d')
        # duplicate storm start date?
        if stormstr1==stormstr_prev:
            stormstr = stormstr1 + next(dupe)
        else:
            stormstr=stormstr1
        stormstr_prev = stormstr1
        
        ncname = stormstr + contour_name
        try:
            cs = xr.open_dataset(storm_root+subdir+ens_dir+ncname)
        except FileNotFoundError:
            print('-- '+subdir+ens_dir+ncname)
            continue
        
        all_contours = []
        for key in list(cs.keys()):
            coord = cs[key].values
            all_contours.append(coord)
        cs.close()
        del cs; gc.collect()
        
        ### get bbox
        with HidePrint(): bbox_edges = get_bbox_edges(all_contours, hemi) 
        storm_areas.append(bbox_edges)
        
    return startdate, enddate, storm_areas

#%%% run fxns!

def get_miz_area(storm_range, seaice):
    # compute miz area
    miz_points = np.zeros(np.shape(si_lon))
    for date in storm_range:
        date = date + timedelta(days=1) # data recorded next day
        try: sic = seaice.sel(time=date.strftime('%Y-%m-%d')).squeeze().values
        except KeyError:
            print('* KeyError - New sea ice file? ...', date)
            sic = np.nan * np.ones(np.shape(si_lon))
        
        miz_points = np.where(((sic>miz[0]) & (sic<=miz[1])), 1, miz_points)
        
    return miz_points

def isolate_area(inside_points, miz_points, si):
    si_in = np.where(inside_points, np.nan, si)
    si_miz = np.where(miz_points<1, np.nan, si_in)
    grid = xr.open_dataset(data_root+'pop_grid.nc')
    TAREA = grid.TAREA.values/1e6
    grid.close()
    return si_miz*TAREA

def compute_climatology(plot_date, year, seaice, inside_points, miz_points):
    plot_date = plot_date # recorded next day
    yearly_changes = []
    clim_years = get_clim_years(year)
    for yy in clim_years:
        
        # check sea ice data file
        seaice_clim = seaice
        try:
            seaice_clim.sel(time=str(yy)+'-'+LZ(month)+'-02').squeeze()
        except KeyError:
            # load new sea ice
            si_file = glob(data_root+'aice_d/*'+ens_mem+'*.'+file_year(yy)+'*.nc')[0]
            with xr.open_dataset(si_file) as ds_si:
                ### open grid info
                ds_si = xr.merge([ds_si['aice_d'].drop_vars(['TLAT', 'TLON', 'ULAT', 'ULON']),
                               grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                              compat='identical', combine_attrs='no_conflicts')
                seaice_clim = ds_si['aice_d']
            gc.collect()
        
        # get dates
        dt = datetime(yy, plot_date.month, plot_date.day)
        
        # load yearly sea ice
        si = seaice_clim.sel(time=dt.strftime('%Y-%m-%d')).squeeze()
        
        # storm area and miz
        si_miz = isolate_area(inside_points, miz_points, si)
        
        # append
        yearly_changes.append( si_miz )
        
    with warnings.catch_warnings(action="ignore"):
        return np.nanmean(yearly_changes, axis=0)   
    

def ice_changes(plot_date, miz_points, bb, seaice):
    '''
    Parameters
    ----------
    plot_date : datetime, storm end date
    bb : 2-n aarray, storm area (bounding boxes)
    seaice : xarray dataset
            opened cesm sea ice file for searching
    Returns
    -------
    storm difference map

    '''
    # get starting and ending ice concentrations
    # si1 = seaice.sel(time=sd.strftime('%Y-%m-%d')).squeeze()
    plot_date = plot_date + timedelta(days=1)
    si = seaice.sel(time=plot_date.strftime('%Y-%m-%d')).squeeze()
    
    # isolate storm areas
    inside_points = find_points_in_contour(bb, si_lon, si_lat)
    
    si_area = isolate_area(inside_points, miz_points, si)
    
    # get climatological change
    clim_change = compute_climatology(plot_date, year, seaice, inside_points, miz_points)
   
    # storm difference
    storm_difference = si_area - clim_change
    
    return storm_difference

def indiv_plot(array, month, ens_mem, si_start, storm_areas=[], storm_count=None):
    #### plot sample areas
    fig, ax = plt.subplots(1,1, figsize=(8,7), 
                                subplot_kw={'projection':ccrs.NorthPolarStereo()})
    if storm_count:
        title_add = ', n='+str(storm_count)
    else: title_add = ''
    ax.set_title(calendar.month_name[month]+': '+ens_mem+title_add)
    ax = setup_plot(ax, extent=[-180,180,60,90], labels=False)
    ax.set_boundary(circle, transform=ax.transAxes)


    ax = seaicecontour(ax, si_start, si_lon, label=[], linestyle='-', \
                      color='k', linewidth=1, levels=[0.15], zorder=102)[0]
    ax = seaicecontour(ax, si_start, si_lon, label=[], linestyle='--', \
                      color='m', linewidth=1, levels=[0.80], zorder=102)[0]
        
    array = np.where(array==0, np.nan, array)
        
    # for diff in storm_differences:
    ax.pcolormesh(si_lon, si_lat, array, 
                  cmap=CMAP, vmin=VMIN, vmax=VMAX,
                  transform=ccrs.PlateCarree(), zorder=100)

    for bb in storm_areas:
        ax.plot(bb[:,0], bb[:,1], transform=ccrs.PlateCarree(), zorder=101, color='green')


#%% DATA AND PLOT
#### set up plot
from matplotlib.patches import Polygon
fontsize = 22
ncols = len(months)
if plot_total: ncols+=1

fig = plt.figure(figsize=(5*ncols,4*len(decades)))
gs = GridSpec(nrows=len(decades), ncols=ncols+2, width_ratios=(1,1,1,1,1,1,0.55,0.55))


axes = []
for row in np.arange(len(decades)):
    ax_row = []
    for col in np.arange(ncols):
        ax_row.append(fig.add_subplot(gs[row, col], projection=ccrs.NorthPolarStereo()))
    axes.append(ax_row)
axes=np.array(axes)


if len(decades)==1 and len(months)==1:
    axes = np.array([axes])

for ax in axes.flatten():
    ax = setup_plot(ax, extent=[-180,180,62.5,90], labels=False)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    # block off central arctic based on northern boundary
    xx = np.arange(0,360,1)
    yy = 85*np.ones(np.shape(xx))
    mycirc = np.vstack((xx,yy)).T
    plt.rcParams['hatch.linewidth'] = 0.5

    ax.add_patch(Polygon(mycirc, closed=False,
                          fill=False, hatch='xxx',lw=0.5,color='gray',
                          transform=ccrs.PlateCarree())
                 )
# background shading
if not (len(decades)==1 or len(months)==1):
    subs = axes
    m, n = subs.shape
    bbox00 = subs[0, 0].get_window_extent()
    bbox01 = subs[0, 1].get_window_extent()
    bbox10 = subs[1, 0].get_window_extent()
    pad_h = 0 if n == 1 else bbox01.x0 - bbox00.x0 - bbox00.width
    pad_v = 0 if m == 1 else bbox00.y0 - bbox10.y0 - bbox10.height
    # for _ in range(20):
        
    for row, color in zip(np.arange(0,len(decades)), decade_colors):
        bbox = subs[row, 0].get_window_extent()
        fig.patches.extend([plt.Rectangle(( bbox.x0 - pad_h*1.125 , bbox.y0 - pad_v/2 +15), # lower left corner
                                          (bbox.width + pad_h)*ncols, bbox.height + pad_v, # width, height
                                          fill=True, color=color, alpha=0.25, zorder=-1,
                                          transform=None, figure=fig)])

# labels
yiter = 1/(len(decades)+1)
ycoords = []
ystart = 1
while ystart >= yiter+0.01:
    ystart -= yiter
    ycoords.append(ystart)
    
for ax, mm in zip(axes.flatten()[0:len(months)], months):
    ax.set_title(calendar.month_name[mm]+'\n', fontsize=fontsize+2)

for di, dec, loc in zip(np.arange(0,len(decades)), decades, ycoords):
    # fig.text(0.065, loc, str(dec[0])+' -\n'+str(dec[-1]), 
    #          va='center', fontsize=fontsize+2)
    # if decade_names:
        fig.text(0.0725, loc, decade_names[di]+ '('+str(dec[0])+' - '+str(dec[-1])+')', 
                 va='center', ha='center', fontsize=fontsize+2)
    
# colorbar
expo = int( np.floor(np.log10(np.abs(VMAX))) )
if expo>2: 
    vmin1 = VMIN/(10**expo)
    vmax1 = VMAX/(10**expo)
else: 
    vmin1 = VMIN
    vmax1 = VMAX


cb = ax.pcolormesh(np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2)), 
                   cmap = CMAP,
                   vmin=vmin1, vmax=vmax1)
cax1 = fig.add_axes([0.25,0.033,0.4,0.04]) 
cbar1 = fig.colorbar(cb, cax=cax1, orientation='horizontal')
if expo>2:
    cbar1.set_label(r'Change in Sea Ice Area ($\times 10^{}$ km$^2$)'.format(expo), fontsize=fontsize)
else:
    cbar1.set_label(r'Change in Sea Ice Area (km$^2$)', fontsize=fontsize)
cax1.tick_params(labelsize=fontsize)


##################################
#### month, decade, ensemble loop
##################################

for mm, month in enumerate(months):
    print('***', month , '***')
    month_start_time = timeIN.time()
    gc.collect()
     
    total_storm_count = 0
    
    for yx, years in enumerate(decades):
        print(str(years[0])+'-'+str(years[-1]))
        
        savename = 'plot_difference2_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
        try:
            # raise FileNotFoundError()
            plot_change = np.load( savepath+savename )
            si_lon = np.load(savepath+'si_lon.npy')
            si_lat = np.load(savepath+'si_lat.npy')
            
        except FileNotFoundError:
        
            starting_ices = []
            ens_changes = []
            
            for ens_mem in ensemble_members: # whats the correct order here? vs years
                print('+ '+ens_mem)
                ens_dir = 'out_'+hemi+'_'+ens_mem+'/'
                ens_count = 0
                ens_areas = []
                         
                decade_changes = []
                storm_durations = []
                
                for year in years:
                    subdir = get_subdir(year)
                    
                    ### open sea ice file
                    si_file = glob(data_root+'aice_d/*'+ens_mem+'*.'+file_year(year)+'*.nc')[0]
                    with xr.open_dataset(si_file) as ds_si:
                        # open grid info
                        TAREA = grid.TAREA.values/1e6
                        ds_si = xr.merge([ds_si['aice_d'].drop_vars(['TLAT', 'TLON', 'ULAT', 'ULON']),
                                       grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                                      compat='identical', combine_attrs='no_conflicts')
                        
                        # get vars
                        si_lon = ds_si['TLONG'].values
                        si_lat = ds_si['TLAT'].values
                        seaice = ds_si['aice_d']
                        del ds_si
                        
                    si_start = seaice.sel(time=str(year)+'-'+LZ(month)+'-02').squeeze() # next day
                    starting_ices.append(si_start)
                    gc.collect()
                    
                    ### get storm dates and areas
                    startdate, enddate, storm_areas = get_storm_dates(subdir, ens_dir, month)
                
                    ### calculate ice changes
                    for sd, ed, bb in zip(startdate, enddate, storm_areas):
                        total_storm_count += 1
                        ens_count+=1
                        ens_areas.append(bb)
                        
                        # set up storm range for miz area determination
                        t1 = sd - timedelta(days=1)
                        t2 = ed + timedelta(days=1)
                        storm_range = daterange(t1, t2, dt=24)
                        
                        storm_duration = ed-sd
                        
                        # compute miz area
                        miz_points = get_miz_area(storm_range, seaice)
                    
                        # compute values and append for avg'ing
                        # ed += timedelta(days=14)
                        change_start = ice_changes(sd, miz_points, bb, seaice)
                        change_end = ice_changes(ed, miz_points, bb, seaice)
                        
                        storm_difference = change_end - change_start ###!!!
                    
                        decade_changes.append( storm_difference )
                        storm_durations.append(storm_duration.days * np.ones(np.shape(storm_difference)))
                        
                        gc.collect()
    
                # get net effect of all storms in this decade for this ens.mem
                # changes = (np.array(decade_changes)*np.array(storm_durations))/np.nansum(storm_durations)
                        # avg pixel change ... want sum change over all storms?
                
                changes = np.array(decade_changes)
                total_change = np.nansum(changes, axis=0)
                
                # sample plot
                if plot_indiv:
                    indiv_plot(total_change, month, ens_mem, si_start, ens_areas, ens_count)
                    gc.collect()
                
                # append to ensemble list
                if not np.all(total_change==0):
                    ens_changes.append( total_change )
                
            np.save(savepath+savename, np.nanmean(ens_changes, axis=0))
            print('*SAVED*', np.shape(np.nanmean(ens_changes, axis=0)))
    
            
        if len(months)>1 and len(decades)>1:
            plotax = axes[yx][mm]
        elif len(months)==1 and len(decades)>1:
            plotax = axes[yx]
        elif len(months)>1 and len(decades)==1:
            plotax = axes[mm]
        elif len(months)==1 and len(decades)==1:
            plotax = axes.item()
            
        plot_change = np.where(plot_change==0, np.nan, plot_change)  
        if expo>2: plot_var = plot_change/(10**expo)
        else: plot_var = plot_change
        
        if ~np.all(np.isnan(plot_var)):
            change =  plotax.pcolormesh(si_lon, si_lat, plot_var, 
                                cmap = CMAP,
                                vmin=VMIN, vmax=VMAX,
                               transform = ccrs.PlateCarree())
            
        net_change = int(round(np.nansum(plot_var),0))
        plotax.text(0.05, -0.15, f"{net_change:,}"+r' km$^2$',
                    transform=plotax.transAxes, fontsize=fontsize-2)
        
        
        #### add starting sea ice contour
        if plot_ice_contour:
            try:
                # load file
                fname = 'ice_extent_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
                starting_ice = np.load(savepath+fname)
                
            except FileNotFoundError: # calculate
                # calculate mean starting extent
                print('... calculating '+str(years[0])+'-'+str(month)+' sea ice extent')
                
                decade_changes = []
                starting_ices = []
                
                for year in years:
                    for ens_mem in ensemble_members:
                        ens_dir = 'out_'+hemi+'_'+ens_mem+'/'
                        
                        # open sea ice file
                        si_file = glob(data_root+'aice_d/*'+ens_mem+'*.'+file_year(year)+'*.nc')[0]
                        with xr.open_dataset(si_file) as ds_si:
                            # open grid info
                            grid = xr.open_dataset(data_root+'pop_grid.nc')
                            TAREA = grid.TAREA.values
                            ds_si = xr.merge([ds_si['aice_d'].drop_vars(['TLAT', 'TLON', 'ULAT', 'ULON']),
                                           grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                                          compat='identical', combine_attrs='no_conflicts')
                            
                            ### get vars
                            si_lon = ds_si['TLONG'].values
                            si_lat = ds_si['TLAT'].values
                            seaice = ds_si['aice_d']
                          
                        si_start = seaice.sel(time=str(year)+'-'+LZ(month)+'-02').squeeze() # next day
                        starting_ices.append(si_start)
                        gc.collect()
    
                # ens. mean starting sea ice extent
                with warnings.catch_warnings(action="ignore"): # empty slices
                    starting_ice = np.nanmean(starting_ices, axis=0)
                    
                # save si extent
                savename = 'ice_extent_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
                np.save(savepath+savename, starting_ice)
                
            with warnings.catch_warnings(action="ignore"):
                seaicecontour(plotax, starting_ice, si_lon, levels=[0.15], 
                                       linewidth=1.75, zorder=500)


    print('> '+str(round((timeIN.time()-month_start_time)/60,1))+' min')
    print()
    
#%% total column
if plot_total:
    print('*** Total ***')
    
    for yx, years in enumerate(decades):
        total_sum1 = []
        for mm, month in enumerate(months):
            savename = 'plot_difference2_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
            total_sum1.append( np.load(savepath+savename) )
            
        total_sum = [x for x in total_sum1 if ~np.all(np.isnan(x))]
        
        plot_change = np.nansum(total_sum,axis=0)
        plot_change = np.where(plot_change==0, np.nan, plot_change)  
        if ~np.all(np.isnan(plot_change)):
            axes[yx][-1].pcolormesh(si_lon, si_lat, plot_change, 
                                    cmap = CMAP, vmin=VMIN, vmax=VMAX,
                                    transform = ccrs.PlateCarree())
        
        net_change = int(round(np.nansum(plot_change),0))
        print(str(years[0])+'-'+str(years[-1])+':', net_change)
        axes[yx][-1].text(0.05, -0.15, f"{net_change:,}"+r' km$^2$',
                          transform=axes[yx][-1].transAxes, fontsize=fontsize-2)

    axes[0][-1].set_title('Net Change\n', fontsize=fontsize+2)     

    line = plt.Line2D((.666,.666),(.1,.875), color="k", linewidth=1.5)
    fig.add_artist(line)    
  
    print('*************')
    
    
    
#%% cyclone locations?
# contours of density???

def cyclone_density(month, years, fname):
    density_array = []
    
    for year in years:
        if year >= 2050 or year in special_years:
            subdir = ''
        elif year > 2019:
            subdir = ''
        elif year in list(np.arange(1990,2000))+list(np.arange(2000,2010)):
            subdir = ''
        else: 
            subdir = ''
            
        if month==5 and (year in np.arange(1980,2050)):
            subdir = 'may2/'
            
        ens_array = np.zeros(np.shape(si_lon))
        for ens_mem in ensemble_members:
            ens_dir = 'out_'+hemi+'_'+ens_mem+'/'
            try:
                [startdate, enddate] = readCensus_str(storm_root+subdir+ens_dir+cens_name+str(year)+'.csv')[0]
            except ValueError:
                print('= '+subdir+ens_dir+cens_name+str(year)+'.csv')
        
            enddate = [ed for i, ed in enumerate(enddate) if startdate[i].month==month]
            startdate = [sd for sd in startdate if sd.month==month]
        
            try:
                [removed_start, removed_end] = readCensus_str(storm_root+subdir+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')[0]
            except FileNotFoundError:
                removed_start = []
                print('- '+subdir+ens_dir+'census_test1_removed-icethresh_'+str(year)+'.csv')
            
            #### get storm areas
            stormstr_prev=''
            dupe = iter(list(string.ascii_lowercase))
            for sx, sd in enumerate(startdate):
                
                if sd in removed_start: # check miz interaction, remove storm from analysis
                    startdate.pop(sx)
                    enddate.pop(sx)
                    continue
                
                stormstr1 = sd.strftime('%Y_%m%d')
                # duplicate storm start date?
                if stormstr1==stormstr_prev:
                    stormstr = stormstr1 + next(dupe)
                else:
                    stormstr=stormstr1
                stormstr_prev = stormstr1
                
                ncname = stormstr + contour_name
                try:
                    cs = xr.open_dataset(storm_root+subdir+ens_dir+ncname)
                except FileNotFoundError:
                    print('-- '+subdir+ens_dir+ncname)
                    continue
                
                all_contours = np.zeros(np.shape(si_lon))
                for key in list(cs.keys()):
                    coord = cs[key].values
                    inside_pts = find_points_in_contour(coord, si_lon, si_lat)
                    all_contours[~inside_pts]+=1
                    
                cs.close()
                del cs; gc.collect()
                                                            # weight by storm duration
                ens_array[np.where(all_contours>0)] += (enddate[sx]-startdate[sx]).days
                
            density_array.append(ens_array)
        
    ## save si extent
    np.save(savepath+fname, np.nanmean(density_array,axis=0))
    
    return np.nanmean(density_array,axis=0)
    

if plot_cyc_density:
    for mm, month in enumerate(months):
        
        for yx, years in enumerate(decades):
            
            try: # loading cyc density file
                fname = 'cyc_density1_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
                density_array = np.load(savepath+fname)
            except FileNotFoundError:
                print('... calculating '+str(years[0])+'-'+str(month)+' cyclone density')
                try:
                    density_array = cyclone_density(month, years, fname)
                except Exception as EE:
                    print(years, month, EE)
                    continue
            
            if len(months)>1 and len(decades)>1:
                plotax = axes[yx][mm]
            elif len(months)==1 and len(decades)>1:
                plotax = axes[yx]
            elif len(months)>1 and len(decades)==1:
                plotax = axes[mm]
            elif len(months)==1 and len(decades)==1:
                plotax = axes
    
            xx = np.where(si_lon>180, si_lon-360, si_lon)
            density_array = np.where(si_lat>85, np.nan, density_array)
            plotax = plot_geocontour(plotax, xx, si_lat, density_array, 
                                     levels=np.arange(25,150,25), color='k', 
                                     lw=0.5, ls='solid', label=True)
            # plotax = plot_geocontour(plotax, xx, si_lat, density_array, 
            #                          levels=[100], color='lime', lw=1, ls='solid')
            
            
#%% add timeseries (line plot)
print(); print('    --------------- line plot maxima ---------------')

ax = fig.add_subplot(gs[:, -2:])
ax1=ax
ax2=ax

line_decades = [np.arange(1982,1992)]
x = 1990
while x < 2100:
    line_decades.append( np.arange(x, x+10) )
    x += 10

# YLIMS = [2095, 1978] #[2103, 1973] #
YLIMS = [2030,1978]
ax.set_ylim(YLIMS)
ax.axvline(0, color='#888888', ls='--', lw=2)
ax.yaxis.tick_right()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)

# ax.set_yticks([yy[0] for yy in line_decades])
ax.set_yticks([yy[0] for yy in line_decades], [str(yy[0])+'-\n'+str(yy[-1]) if yy[0] in [d[0] for d in decades] else '' for yy in line_decades], fontsize=fontsize-2)
scale=1e6
ax.set_xticks(np.arange(-1.5,0.5,0.5))
for item in ax.get_xticklabels(): item.set_fontsize(fontsize-2)
ax.set_xlabel('Change in Sea Ice Area\n'+r'($\times 10^6$ km$^2$)', fontsize=fontsize+2)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)

mcolors = ['#E68310', '#E73F74', '#7F3C8D', '#008695', '#4b4b8f']
mmarkers = ['<', 's', '>', 'D', 'o']
mnames = [calendar.month_abbr[mm] for mm in months]

#### calculations
annual_sum = []
month_sum = {mm:[] for mm in months}

for yx, years in enumerate(line_decades):
    year_maps = []
    for mm, month in enumerate(months):
        savename = 'plot_difference2_'+str(month)+'_'+str(years[0])+'_'+str(years[-1])+'.npy'
        grd = np.load(savepath+savename)
        if np.all(np.isnan(grd)):
            grd = np.zeros(np.shape(si_lon))
        
        year_maps.append( grd )
        month_sum[month].append( np.nansum(grd) )
        
    plot_change = np.nansum(year_maps,axis=0)
    plot_change = np.where(plot_change==0, np.nan, plot_change)  
    annual_sum.append( np.nansum(plot_change) )

#### plot
yyy = np.array([yy[0] for yy in line_decades]).T

for mm, color, mark, mne in zip(months, mcolors, mmarkers, mnames):
    ax1.plot(np.array(month_sum[mm])/scale, yyy, color=color, lw=3)
    
    print(mm, end=': ')
    print(np.array(line_decades)[np.where(np.abs(np.array(month_sum[mm])) == np.nanmax(np.abs(np.array(month_sum[mm]))))])
    
    for di, dec in enumerate(decades):
        impact = np.array(month_sum[mm])[np.where(line_decades==dec)[0][0]]
        ax1.scatter(impact/scale, dec[0], s=100, color = color, marker=mark)
        if di==0:
            if mm==7:mv=-0.6
            elif mm==6: mv= -0.6
            elif mm==5: mv=-0.45
            elif mm==8:mv=-0.15
            elif mm==9:mv=-0.16
            else: mv=0
            ax.text((impact/scale)+mv, YLIMS[-1]-1, mne, fontsize=fontsize-2, color=color)
    
# total
ax2.plot(np.array(annual_sum)/scale, yyy, color='k', lw=3.5)
print('Total', end=': ')
print(np.array(line_decades)[np.where(np.abs(np.array(annual_sum)) == np.nanmax(np.abs(np.array(annual_sum))))])

for di, dec in enumerate(decades):
    impact = np.array(annual_sum)[np.where(line_decades==dec)[0][0]]
    ax1.scatter(impact/scale, dec[0], s=300, color = 'k', marker='*')
    if di==0:
        ax.text((impact/scale)-0.25, YLIMS[-1]-1, 'Total', fontsize=fontsize-2, color='k', fontweight='bold')

# legend
for name, color, marker in zip(mnames+['Total'], mcolors+['k'], mmarkers+['*']):
    ax.plot([],[], label=name, color=color, marker=marker, lw=2.5, markersize=12)
ax.legend(loc='lower left', fontsize=fontsize-2, handlelength=1, handletextpad=0.33,
          edgecolor=(1, 1, 1, 0), facecolor=(1, 1, 1, 0), framealpha=0)


# %% end
if SAVEFIG:
    fig.savefig(savepath111+'spatial_maps5a1.png', bbox_inches = "tight", dpi=400)


#%% STORM COUNT PLOT (fig s3)
decades1 = [np.arange(1982,1992),
           np.arange(1990,2000),
           np.arange(2000,2010),
           np.arange(2010,2020),
            np.arange(2020,2030),
            np.arange(2030,2040)
            ]

mons = []
for month1 in [6,7,8,9]:

    decs1 = {}
    
    for yx, years in enumerate(decades1):
        
        scounts = []
        for ens_mem in ensemble_members: # whats the correct order here? vs years
            ens_dir = 'out_'+hemi+'_'+ens_mem+'/'
            
            sc = 0
            for year in years:
                subdir = get_subdir(year)
                
                startdate, enddate, storm_areas = get_storm_dates(subdir, ens_dir, month1)
                sc += len(startdate)
            scounts.append(sc)
            
        decs1[yx] = scounts
        
        print()
        print(str(years[0])+'-'+str(years[-1]), str(round(np.nanmean(scounts),2)))
        print()
    mons.append(decs1)
    


#%%%% plot
from scipy import stats

f1, ax1 = plt.subplots(1,1, figsize=(8,4))

month1 = 7
decs1 = mons[1]

ax1.set_title('2010-2019 has fewer July storms compared with other decades')
# ax1.set_title(calendar.month_name[month1], fontweight='bold')
ax1.set_ylabel(calendar.month_name[month1]+ ' Storm Count')

XLABELS = [str(yy[0])+'-'+str(yy[-1]) for yy in decades1]
for yx, years in enumerate(decades1):
    l1 = decs1[yx]
    l1.sort()
    
    x1 = yx+np.linspace(0,0.85,len(ensemble_members))
    ax1.bar(x1, l1, width=x1[-2]-x1[-1], alpha=0.65)
    ax1.plot(x1, np.nanmean(l1)*np.ones(np.shape(x1)), ls='--')
    ax1.text(x1[0], np.nanmean(l1)+0.5, str(round(np.nanmean(l1),1)), color = 'C'+str(yx))
    
    pval = stats.ttest_ind(decs1[yx], decs1[3]).pvalue
    if pval < 0.05:
        XLABELS[yx]+='*'
        
ax1.set_xticks(np.arange(len(decades1))+0.425) 
ax1.set_xticklabels(labels=XLABELS)
for yx, years in enumerate(decades1):
    ax1.get_xticklabels()[yx].set_color('C'+str(yx))
    
if False:
    savepath111 =  '/Users/mundi/Desktop/out_figs/'
    f1.savefig(savepath111+'july_counts.png', bbox_inches = "tight", dpi=400)

#%% end2


