#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 2024
Prepared for publication - 08 Aug 2025
cesm_functions.py

common functions for data analysis

@author: mundi
"""
import os, sys
import numpy as np
import xarray as xr
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cmocean.cm as cmo
        
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
    
class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#%% starting info, plotting
def LZ(day):
    ''' get leading zero string'''
    if day>=10:
        return str(day)
    elif day<10:
        return '0'+str(day)
    
def rstr(val, n=3):
    return str(round(val, n))

def daterange(start_date, end_date, dt=6):
    alldates=[]
    delta = timedelta(hours=dt)
    while start_date <= end_date:
        alldates.append(start_date)
        start_date += delta
    return alldates


def background_plot(extent=[-160,90,50,60], returnfig=False, title=[], labels=True):
    
    if type(extent)==str:
        if extent=='north' or extent=='n':
            extent = [-160,90,50,60]
            myproj = ccrs.NorthPolarStereo(central_longitude=0)
        elif extent=='south' or extent=='s':
            extent=[-180,180, -53,-90]
            myproj = ccrs.SouthPolarStereo()
    elif type(extent)==list:
        if extent[-1]>0: myproj = ccrs.NorthPolarStereo(central_longitude=0)
        else: myproj = ccrs.SouthPolarStereo()
     
    fig=plt.figure(figsize=[15,15]) 
    ax = plt.axes(projection=myproj)
    ax.coastlines('50m',edgecolor='black',linewidth=0.75, zorder=-25)
    ax.set_extent(extent, ccrs.PlateCarree())
    try:
        ax.gridlines(draw_labels=labels)
    except:
        print('Unable to create grid lines on map')
    ax.add_feature(cfeature.LAND, facecolor='0.75', zorder=-26)
    ax.add_feature(cfeature.LAKES, facecolor='0.85', zorder=-25)
    
    if title:
        if type(title)!= str: title=str(title)
        ax.set_title(title, fontsize=22)
    
    if not returnfig: return ax
    if returnfig: return fig, ax
    
def setup_plot(ax, extent=[-160,90,50,60], title=[], labels=True):
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
    
def plot_geocontour(ax, lon, lat, var, levels, color='k', lw=3, ls='solid'):
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
    ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
              linewidths = lw, zorder=10, transform=ccrs.PlateCarree(),
              linestyles=ls) 
    ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
              linewidths = lw, zorder=10, transform=ccrs.PlateCarree(),
              linestyles=ls)
    return ax

#%% sea ice

def load_seaice(ice_fname, year, month, day, pop, latlon=True):
    in_date = datetime(year, month, day)
    si_date = in_date + timedelta(days=1) # si records on next daily timestep
    
    if month==2 and (day==28 or day==29): si_date = datetime(year, 3, 1)
    
    datestr = si_date.strftime('%Y-%m-%d')

    ### open file
    ds = xr.open_dataset(ice_fname)

    var_in = 'aice_d'     
    var_to_keep = ds[var_in]

    ds = xr.merge([var_to_keep.drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                   pop[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                  compat='identical', combine_attrs='no_conflicts')
    ds.close()
    gb=gc.collect()
    del gb
    
    ### get vars #.sel(ni=slice(50,90))
    si = ds[var_in].sel(time=datestr).squeeze()
    if latlon:
        si_lon = ds['TLONG'].values
        si_lon = np.where(si_lon>180, si_lon-360, si_lon)
        si_lat = ds['TLAT'].values
        return si, si_lon, si_lat
    else:
        return si
    
def seaicecontour(ice_fname, year, month, day, ax, pop, label=[], \
                       color='k', linewidth=4, levels=[0.15]):
    import calendar
    if type(year) == str: year = int(year)
    
    if day == []:  # monthly average
        all_si_month = []
        num_days = calendar.monthrange(int(year), month)[1]
        days = [datetime(int(year), month, day) for day in range(1, num_days+1)]
        for thisday in days:
            #latlon=False if day.day != 1 else True
            si_day, si_lon, si_lat = load_seaice(ice_fname, year, month, thisday.day, pop, latlon=True)
            all_si_month.append(si_day)
        si_day = np.ma.mean(all_si_month, axis=0)
    else:    
        # load ice data
        si_day, si_lon, si_lat = load_seaice(ice_fname, year, month, day, pop, latlon=True)

    si_lon = np.where(si_lon>180, si_lon-360, si_lon)

    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(si_lon, -0.01)
    lon_lesser = np.ma.masked_less(si_lon, 0)
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(si_lat, mask=lon_greater.mask).filled(np.nan)
    lat_lesser = np.ma.MaskedArray(si_lat, mask=lon_lesser.mask).filled(np.nan)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(si_day, mask=lon_greater.mask).filled(np.nan)
    si_lesser = np.ma.MaskedArray(si_day, mask=lon_lesser.mask).filled(np.nan)
    
    lon_greater=lon_greater.filled(np.nan)
    lon_lesser=lon_lesser.filled(np.nan)

    # contours
    levels = levels # 15% ice extent definition
    contours1 = ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
                  linewidths = linewidth, zorder=10,
                  transform=ccrs.PlateCarree()) 
    contours2 = ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
                  linewidths = linewidth, zorder=10,
                  transform=ccrs.PlateCarree())
    return ax, [contours1,contours2]

def plot_seaicecontour(seaice_ds, year, month, day, ax, label=[], \
                       color='k', linewidth=4, linestyle='-', levels=[0.15], zorder=10):
    if type(year) == str: year = int(year)
    
    datestr = str(year)+'-'+LZ(month)+'-'+LZ(day)
    si_day = seaice_ds['aice_d'].sel(time=datestr).squeeze()
    si_lon = seaice_ds['TLONG'].values
    si_lat = seaice_ds['TLAT'].values
    si_lon = np.where(si_lon>180, si_lon-360, si_lon)

    #do masked-array on the lon
    lon_greater = np.ma.masked_greater(si_lon, -0.01)
    lon_lesser = np.ma.masked_less(si_lon, 0)
    # apply masks to other associate arrays: lat
    lat_greater = np.ma.MaskedArray(si_lat, mask=lon_greater.mask).filled(np.nan)
    lat_lesser = np.ma.MaskedArray(si_lat, mask=lon_lesser.mask).filled(np.nan)
    # apply masks to other associate arrays: daily ice
    si_greater = np.ma.MaskedArray(si_day, mask=lon_greater.mask).filled(np.nan)
    si_lesser = np.ma.MaskedArray(si_day, mask=lon_lesser.mask).filled(np.nan)
    
    lon_greater=lon_greater.filled(np.nan)
    lon_lesser=lon_lesser.filled(np.nan)

    # contours
    levels = levels # 15% ice extent definition
    ax.contour(lon_greater, lat_greater, si_greater, colors=color, levels=levels, 
                linewidths = linewidth, zorder=zorder,linestyles=linestyle,
                transform=ccrs.PlateCarree()) 
    ax.contour(lon_lesser, lat_lesser, si_lesser, colors=color, levels=levels, 
                linewidths = linewidth, zorder=zorder,linestyles=linestyle,
                transform=ccrs.PlateCarree())
    return ax
    
def calc_seaice_area(si_grid, pop):
    mysum = np.nansum(pop*si_grid)
    return mysum

def get_total_area(si_grid, pop):
    si_grid = np.where(si_grid>=0, 1, np.nan)
    return np.nansum(pop*si_grid)

#%% cyclone information

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

def get_storm_distance(ice_fname, current_date, storm_lat, pop, bbox_edges=[]):
    # load 15% sea ice contour
    fig, ax = background_plot(returnfig=True)
    ax, conts = seaicecontour(ice_fname, current_date.year, current_date.month, 
                            current_date.day,ax=ax,pop=pop, levels=[0.15],color='black', 
                            linewidth=2.5)
    contlon, contlat = [],[]
    for CS in conts:
       for lst in CS.allsegs: #get contour coord arrays
           for ix, pts in enumerate(lst): #
               if ix==0: 
                   all_pts = pts
               else:
                   all_pts = np.concatenate((all_pts, pts), axis=0)
                   
               pts=np.array(pts)
               for yi, y in enumerate(pts[:,0]):
                    if y<0:
                        pts=np.concatenate((pts, np.array([[y+360],[pts[yi,1]]]).T ))

               ### average latitudes of contour (entire globe or latitudes within bbox?])
               if bbox_edges != []:
                   # get contour within box
                   cont_in = find_points_in_contour(bbox_edges, pts[:,0], pts[:,1])
                   contlat1 = np.ma.masked_array(pts[:,1],mask=cont_in)
                   contlon1 = np.ma.masked_array(pts[:,0],mask=cont_in)
                   
                   if len(np.unique(contlat1.mask)) > 1:
                       for j, jj in enumerate(contlat1.data[~contlat1.mask]):
                           contlat.append([contlat1.data[~contlat1.mask][j]])
                           contlon.append([contlon1.data[~contlon1.mask][j]])
               elif bbox_edges == []:
                   contlat = pts[:,1]
                   
    ### average latitudes of contour (entire globe or latitudes within bbox?])
    avglat = np.mean(contlat)

    ### calculate distance
    # subtract latitudes * some distance (111 km or so)
    km = 111
    dist = (avglat-storm_lat)*km
   
    plt.close(fig)
    return dist

#%% contours

def load_psl(file, year, month, days, daily=False, loc='n'):
    if type(days)==int:
        days = [days]
    
    start_date= str(year)+'-'+LZ(month)+'-'+LZ(days[0])
    end_date =  str(year)+'-'+LZ(month)+'-'+LZ(days[-1])
    
    if loc=='n' or loc=='north': myslice = slice(55,90)
    elif loc == 's' or loc=='south': myslice = slice(-90,-55)
    
    with xr.open_dataset(file) as ds:
        
        lon = ds['lon'].values
        lon = np.where(lon>180, lon-360, lon)
        lon, lat = np.meshgrid(lon, ds['lat'].sel(lat=myslice).values)
        
        if daily:
            for day in days:
                slp= ds['PSL'].sel(time=slice(start_date, end_date)).resample(time='1D').mean(dim='time')/100
                time = slp.time
        else:
            slp = ds['PSL'].sel(time=slice(start_date, end_date))/100
            time = ds['time'].sel(time=slice(start_date, end_date))
        slp = slp.sel(lat=myslice)
            
    return lon, lat, slp, time


def get_contour_points2(year, month, day, level, storm_info, file, lon_thresh=10, lat_thresh=5, loc='n'):
    if type(level) != list: level=[level]
    
    print(''); print(year, month, day); print('---------')
    

    x, y, pressure, time = load_psl(file, year, month, day, daily=True, loc=loc)
    if 0 in list(np.shape(pressure)):
        raise NameError('get_contour_points: empty pressure ' + str((year, month, day)))
    elif 1 in list(np.shape(pressure)):
        pressure=np.squeeze(pressure)
        
    xv = x[0,:]
    yv = y[:,0]
    
    x, y = np.meshgrid(xv,yv)
    
    ### find minimum pressures and compare to storm_info 
    storm_x = storm_info[0]
    storm_y = storm_info[1]
    nearby_min = False
    
    while nearby_min==False:
        indp = np.where(pressure == np.nanmin(pressure))
        minlon = x[indp]
        minlat = y[indp]
        
        for pt_id, lo in enumerate(minlon):
            
            if minlon[pt_id]<0: minlon[pt_id] =minlon[pt_id]+360
            if abs(minlon[pt_id] - storm_x) < lon_thresh and abs(minlat[pt_id] - storm_y) < lat_thresh:
                min_x = float(minlon[0])
                min_y = float(minlat[0])
                min_p = np.nanmin(pressure)
                nearby_min = True
                break
            else:
                pressure[indp] = np.nan #[pt_id]
                break
    fig, ax =  background_plot(returnfig=True)
    ax.set_title(str(year)+' '+str(month)+' '+str(day) + ': ' +str(round(min_p,1))+' hPa' +',  '+str(round(min_x,2)), 
                 fontsize=26)
    
    contours_all = ax.contour(x, y, pressure, levels=level,
                      linewidths = 1.5, zorder=10, colors='k',
                      transform=ccrs.PlateCarree())
    
    
    ax.plot(min_x, min_y,'y*',transform=ccrs.PlateCarree(), markersize=40)    
    
    coords=[]
    for lvl, lev in enumerate(level):
        try:
            coords = coords + contours_all.allsegs[lvl]
        except IndexError:
            print(str(lev) +': ' + str(month)+', '+ str(day))
        
    min_x = np.squeeze(min_x)
    if min_x > 180:
        min_x = min_x - 360
    
    plt.close(fig)
    return coords, min_x, np.squeeze(min_y), ax


def detect_contours2(storm_daymonth_grid, daymonth_grid, year, storm_info, intervals=[990,1000], local_file='', loc='n'):
    import matplotlib.path as mpltPath
    
    # return lists of selcted contours and their respective date
    all_cont_dt1, all_conts1 = [], []
    
    snum=-1
    for month, day in storm_daymonth_grid:
        snum+=1
        ### get pressure contours      
        intervals = [990,1000] #[980,985,990,995,1000,1005,1010] #
        current_date = datetime(int(year), int(month), int(day))
        
        if snum >= len(storm_info):break
        
        coords, min_x, min_y, ax = get_contour_points2(year, month, day, intervals, 
                                                       storm_info[snum], 
                                                       local_file, loc=loc)
        
        points = np.array([ [min_x, min_y] ])
        for cc1 in coords :
            cc = np.array(cc1)
            
            if len(cc) > 2 : path = mpltPath.Path(cc)
            else: continue
            pts_inside = path.contains_points(points, radius=15)
            
            if any(pts_inside):
                all_conts1.append(cc)
                all_cont_dt1.append(current_date)
                ax.plot(cc[:,0], cc[:,1], 'r', linewidth=3, transform=ccrs.PlateCarree())
                continue
            
            if abs(abs(min_x) - 180) < 20:
                second_x = 175
                second_pts = path.contains_points(np.array([ [second_x, min_y] ]))
                ax.plot(second_x, min_y,'c*',transform=ccrs.PlateCarree(), markersize=30) 
                if any(second_pts):
                    all_conts1.append(cc)
                    all_cont_dt1.append(current_date)
                    [x1,x2], [y1,y2] = geoplot_2d(cc[:,0], cc[:,1])
                    ax.plot(x1, y1, 'b', linewidth=5, transform=ccrs.PlateCarree())
                    ax.plot(x2, y2, 'b', linewidth=5, transform=ccrs.PlateCarree())
                continue
    
    print()
    return all_cont_dt1, all_conts1   

#%%% bbox

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
            
            if boo: ###!!!
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
                if lons1[li+1] - ll > 20: ###!!! threshold here
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

#%% processing
def readCensus(file, convertDT=False):
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
        startdate.append(a) 
        startlat.append(float(b))
        startlon.append(float(c))
        pressure.append(float(d))
        enddate.append(e)
        endlat.append(float(f))
        endlon.append(float(g))
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

def readCensus_str(file, convertDT=False):
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

