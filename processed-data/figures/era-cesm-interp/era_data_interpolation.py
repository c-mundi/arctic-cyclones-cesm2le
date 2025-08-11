#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 2024
Prepared for publication - 08 Aug 2025
era_data_interpolation.py

interpolate era5/sea ice data to cesm resolution

@author: mundi
"""

# DATA PATHS
era_path = '/Users/mundi/Desktop/era_data/arctic/'
cesm_path = '/Users/mundi/Desktop/cesm_data/'
cesm_psl = cesm_path + 'psl/north/'
output_path = '/Users/mundi/Desktop/era_data/arctic/cesm-interp/'
ice_fname = '/Users/mundi/Desktop/seaice/'

years = list(np.arange(2010,2019+1)) + list(np.arange(1982, 1991+1))

#%% imports, functions
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import calendar
import time as timeIN
import glob
import gc

def daterange(start_date, end_date, dt=6):
    alldates=[]
    delta = timedelta(hours=dt)
    while start_date <= end_date:
        alldates.append(start_date)
        start_date += delta
    return alldates

def LZ(day):
    ''' get leading zero string'''
    if day>=10:
        return str(day)
    elif day<10:
        return '0'+str(day)

from cartopy.util import add_cyclic_point

def local_era5(slp_file, year, month, days, idx=6):
    if type(days)==int: days=[days]
    startdate = str(year)+'-'+LZ(month)+'-'+days[0]
    enddate = str(year)+'-'+LZ(month)+'-'+days[-1]
    
    ds = xr.open_dataset(slp_file)
    
    msl = ds['msl'].sel(time=slice(startdate, enddate))/100
    msl, cyc_x = add_cyclic_point(msl, coord=ds['longitude'].values)
    
    lon, lat = np.meshgrid(cyc_x, ds['latitude'].values)
    time = ds['time']
    
    msl_out, msl6, time6 = [],[],[]
    count = 0
    for xi, msl1 in enumerate(msl):
        if count == 0: 
            timestr = str(time[xi].values).split('.')[0]
            time6.append( datetime.strptime(timestr,'%Y-%m-%dT%H:%M:%S') )
        msl6.append(msl1) #.values)
        count += 1
        if count ==idx:
            msl_out.append(np.nanmean(msl6,axis=0))
            count=0; msl6 = []
    time6 = np.array(time6)
    ds.close()
    return time6, lon, lat, msl_out

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


#%% start
months = [6,7,8,9]
ens_mem = '1251.012'

#%% cesm grid
time = daterange(datetime(2010,6,1), datetime(2010,9,30,23), dt=6)

cesm_file = glob.glob(cesm_psl +'*'+ ens_mem +'*h2.PSL*'+ file_year(2010)+'*.nc')

# pressure
cesm = xr.open_dataset(cesm_file[0]).sel(lat=slice(60,90))
cesm = cesm.sel(time = slice(time[0].strftime('%Y-%m-%d'),time[-1].strftime('%Y-%m-%d')))
clon = cesm['lon'].values
clon = np.where(clon>180, clon-360, clon)
clat = cesm['lat'].values
cesm_lon, cesm_lat = np.meshgrid(clon, clat)
cesm['PSL'] = cesm['PSL']/100
cesm.close()
print('Collected cesm grid...')
gc.collect()


#%% era
for year in years:
    
    time = daterange(datetime(year,6,1), datetime(year,9,30,23), dt=6)

    era_msl_file = era_path + 'msl_'+ str(year)+'-' 
   
    cesm_file = glob.glob(cesm_psl +'*'+ ens_mem +'*h2.PSL*'+ file_year(year)+'*.nc')

    print('Starting '+str(year)+' ERA interpolation: ', end='')
    interp_start = timeIN.time()
    
    ### get time vector
    time = daterange(datetime(year,6,1), datetime(year,9,30,23), dt=6)
    
    ### load pressure data
    interp_msl = []
    for month in months:
        print(month, end=' ')
        calendar_days = list(np.arange(1,calendar.monthrange(year, month)[1]+1))
        alldays = [str(d) if d>=10 else '0'+str(d) for d in calendar_days]
        
        TIME, lon, lat, MSL_ERA = local_era5(era_msl_file+str(month)+'.nc', 
                                             year, month, alldays, idx=6)
        
        
        for msl_era in MSL_ERA:
            new_grid = griddata(np.array([lon.flatten(), lat.flatten()]).T, msl_era.flatten(),
                                np.array([cesm_lon.flatten(), cesm_lat.flatten()]).T)
            interp_msl.append( np.reshape(new_grid, np.shape(cesm_lon)) )
        
    print();print('... Done: ' + str(round((timeIN.time() - interp_start)/60, 2))+' min')
    
    ### make data array for export
    da = xr.DataArray(
        np.array(interp_msl),
        dims=("time", "y", "x"),
        coords=[
            ("time", time),
            ("y", clat),
            ("x", clon)
        ],
    )
    da.to_netcdf(output_path+str(year)+'_arctic.nc')
    
    
#%% sea ice
import netCDF4

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


for year in years:

    time = daterange(datetime(year,6,1), datetime(year,9,30,23), dt=6)

    cesm_file = glob.glob(cesm_psl +'*'+ ens_mem +'*h2.PSL*'+ file_year(year)+'*.nc')

    print('Starting '+str(year)+' SEA ICE interpolation: ', end='')
    interp_start = timeIN.time()
    
    ### get time vector
    time = daterange(datetime(year,6,1), datetime(year,9,30,0), dt=24)
    
    ### load pressure data
    interp_msl = []
    for month in months:
        print(month, end=' ')
        calendar_days = list(np.arange(1,calendar.monthrange(year, month)[1]+1))

        for day in calendar_days:        
            si, si_lon, si_lat = load_seaice(ice_fname, year, month, day)
        
            new_grid = griddata(np.array([si_lon.flatten(), si_lat.flatten()]).T, si.flatten(),
                                np.array([cesm_lon.flatten(), cesm_lat.flatten()]).T)
            interp_msl.append( np.reshape(new_grid, np.shape(cesm_lon)) )
        
    print();print('... Done: ' + str(round((timeIN.time() - interp_start)/60, 2))+' min')
    
    ### make data array for export
    da = xr.DataArray(
        np.array(interp_msl),
        dims=("time", "y", "x"),
        coords=[
            ("time", time),
            ("y", clat),
            ("x", clon)
        ],
    )
    da.to_netcdf(output_path+str(year)+'_seaice.nc')
