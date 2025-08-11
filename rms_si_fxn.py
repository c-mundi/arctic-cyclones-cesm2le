#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 2024
cesm/rms_si.py

calculate rms values for comparative ice values

@author: mundi
"""
#%% imports
import numpy as np
import xarray as xr
import glob
import calendar
from datetime import datetime, timedelta
import gc

#%% functions

def get_total_area(si_grid, area):
    si_grid = np.where(si_grid>0, 1, np.nan)
    return np.nansum(si_grid*area)

def load_seaice(root_dir, year, month, day, hemisphere, latlon=True):
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

def load_cesm(file_path, year, hemisphere, lat_bound, ens_mem, grid_path):
    try: ice_fname = glob.glob(file_path+'aice_d/'+'*'+ens_mem+'*'+'h1.aice_d'+'*.'+file_year(year)+'*')[0]
    except:
        print(file_path+'aice_d/'+'*'+ens_mem+'*'+'h1.aice_d'+'*.'+file_year(year)+'*')
        raise NameError
    
    ds = xr.open_dataset(ice_fname)
    ds.close()

    # si_lon = ds['TLON'].values
    # si_lat = ds['TLAT'].values

    #### grid and final

    grid = xr.open_dataset(grid_path+'pop_grid.nc')
    # TAREA = grid.TAREA.values

    var_in = 'aice_d'     
    var_to_keep = ds[var_in]

    ds = xr.merge([var_to_keep.drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                   grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                  compat='identical', combine_attrs='no_conflicts')
    grid.close()
    gc.collect()
    
    # get regional ice
    if hemisphere == 'n':
        return ds.where(ds.TLAT>lat_bound, drop=True)
    elif hemisphere == 's':
        return ds.where(ds.TLAT<-lat_bound, drop=True)
    
#%% starting info

def rms_si(concentration, ens_mem, hemisphere, ice_fname, file_path, grid_path, conc_spacing=1, years=[]):
    
    if len(years) == 0:
        years = np.arange(1982, 2019+1) ###!!!

    print('RMS analysis: '+str(concentration), end=' ')
    # testing values (0% -> observations)
    model_conc = np.arange(0, concentration, conc_spacing)    

    if hemisphere =='n': 
        months = [6,7,8,9]
        lat_bound = 50
    elif hemisphere=='s': 
        months = [12,1,2,3]
        lat_bound = -50
    
    #### grid info
    # nsidc grid
    _, si_lon, si_lat = load_seaice(ice_fname, 2010,1,1, hemisphere, latlon=True)
    obs_area = 25*25*np.ones(np.shape(si_lon))
    # cesm -> TAREA
    with xr.open_dataset(grid_path+'pop_grid.nc') as grid:
        if hemisphere == 'n':
            grid = grid.where(grid.TLAT>lat_bound, drop=True)
        elif hemisphere == 's':
            grid = grid.where(grid.TLAT<lat_bound, drop=True)
        TAREA = grid.TAREA.values


    #### set up analysis variable
    AREAS = {}
    AREAS[str(concentration)+'_obs'] = []
    for cc in model_conc:
        AREAS[str(cc)] = []

    #### daily analysis
    for year in years:
        ds = load_cesm(file_path, year, hemisphere, lat_bound, ens_mem, grid_path)
        for month in months:
            days = list(np.arange(1,calendar.monthrange(year, month)[1]+1))
            for day in days: # for every day
                # load sea ice observations
                si = load_seaice(ice_fname, year, month, day, hemisphere, latlon=False)*100
                # calc total ice area
                area_total_obs = get_total_area(si, obs_area)
                # calculate area for given concentration
                si_grid = np.where(si > concentration, si, np.nan)
                area_conc_obs = get_total_area(si_grid, obs_area)
                # add to export variable
                if area_total_obs==0: # or area_conc_obs==0 
                    if year!=1984: print('~ 0 area ~', year, month, day)
                    continue
                else: # only compare good dates
                    AREAS[str(concentration)+'_obs'].append(area_conc_obs/area_total_obs)
                
                # load daily model sea ice
                date = datetime(year,month,day) + timedelta(days=1) # data recorded next day
                try: sic = ds['aice_d'].sel(time=date.strftime('%Y-%m-%d')).squeeze().values*100
                except KeyError as ke:
                    print('* New sea ice file:', date)
                    print(ke)
                    sic = np.nan * np.ones(np.shape(si_lon))
                    continue
                # calc total ice area
                area_total_obs = get_total_area(sic, TAREA)
                
                for conc in model_conc:
                    # get area for each concentration
                    sic_grid = np.where(sic > conc, sic, np.nan)
                    area_conc_model = get_total_area(sic_grid, TAREA)
                    AREAS[str(conc)].append(area_conc_model/area_total_obs)
                
    #### calculate rms for each model value (return vector)
    rms_list = []
    min_rms, min_conc = 999,999
    for cc in model_conc:
        diff_sq = (np.array(AREAS[str(cc)])-np.array(AREAS[str(concentration)+'_obs']))**2
        rms = np.sqrt(np.sum(diff_sq)/len(diff_sq))
        rms_list.append( rms )
        
        if rms < min_rms:
            min_rms = rms
            min_conc = cc
            
    print('- '+str(min_conc))
    
    if 1984 in years and hemisphere =='n': print('--> 1984 ice area')
    
    return min_conc, min_rms
    




#%% end