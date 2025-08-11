#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26  2024
Prepared for publication - 08 Aug 2025
cyctrack_test_fxns.py

@author: mundi
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.crs as ccrs

import cesm_functions as cf

#%% GET POINTS

def get_contour_points(year, month, day, level, storm_info, file, lon_thresh=10, lat_thresh=5, loc='n'):
    if type(level) != list: level=[level]
    
    print(''); print(year, month, day); print('---------')
    
    x, y, pressure, time = cf.load_psl(file, year, month, day, daily=True, loc=loc)
    
    if 0 in list(np.shape(pressure)):
        raise NameError('get_contour_points: empty pressure ' + str((year, month, day)))
    elif 1 in list(np.shape(pressure)):
        pressure=np.squeeze(pressure)
        
    x = np.where(x<0, x+360, x)
        
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
    fig, ax =  cf.background_plot(returnfig=True)
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
    # if min_x > 180:
    #     min_x = min_x - 360
    
    plt.close(fig)
    return coords, min_x, np.squeeze(min_y), ax



#%% DETECT

def detect_contours2(storm_daymonth_grid, daymonth_grid, year, storm_info, intervals=[990,1000], local_file='', loc='n'):
    import matplotlib.path as mpltPath
    
    # return lists of selcted contours and their respective date
    all_cont_dt1, all_conts1 = [], []
    
    snum=-1
    for month, day in storm_daymonth_grid:
        snum+=1
        ### get pressure contours      
        current_date = datetime(int(year), int(month), int(day))
        
        if snum >= len(storm_info):break
        
        coords, min_x, min_y, ax = get_contour_points(year, month, day, intervals, 
                                                       storm_info[snum], 
                                                       local_file, loc=loc)
        
        points = np.array([ [min_x, min_y] ])
        for cc1 in coords :
            cc = np.array(cc1)
            
            if len(cc) > 2 : path = mpltPath.Path(cc)
            else: continue
            pts_inside = path.contains_points(points)
            
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
                    [x1,x2], [y1,y2] = cf.geoplot_2d(cc[:,0], cc[:,1])
                    ax.plot(x1, y1, 'b', linewidth=5, transform=ccrs.PlateCarree())
                    ax.plot(x2, y2, 'b', linewidth=5, transform=ccrs.PlateCarree())
                continue
    
    print()
    return all_cont_dt1, all_conts1   
