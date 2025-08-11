#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 2024
Prepared for publication - 08 Aug 2025
storm_tseries.py
- sea ice
- winds/sst

calculate 3-week timeseries of sea ice, wind/sst

seaice runtime: ~2min/yr

@author: mundi
"""

root_path = '/Users/mundi/Desktop/cesm-code/'

#%% imports
import os
import numpy as np
import glob

import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import griddata

import traceback
import time as timeIN
import string

import cesm_functions as cf
import gc

def get_miz_values(census_path):
    with open(census_path+'readme.txt') as f:
            lines = f.readlines()
            for line in lines:
                if line.split(':')[0] != 'New MIZ': continue
                vals = line.split(':')[-1][1:-1]
                break
    vals1 = vals.split(',')[0].replace('[','')
    vals2 = vals.split(',')[1].replace(']','').replace(' ','')
    return [float(vals1), float(vals2)]

#%% STARTING INFO
run_sample = False
savenc = True

years = list(np.arange(2010, 2019+1))+list(np.arange(1982,1991+1))

e_prefix = ['1231','1251','1281','1301'] 
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]

### SAMPLE
# run_sample = True
# years=[2015]
# ensemble_members = ['1301.020'] 

### SELECT VARIABLES
run_seaice = True
run_sort = not run_seaice
run = {}

if run_seaice:
    ncnameadd= '_seaice'     
    savedir = 'seaice/'
    run['seaice'] = True

if run_sort:
    ncnameadd= '_sort'   
    savedir = 'winds_sst/'
    run['winds'] = True
    run['sst'] = True

### SELECT STORM AREAS
run1000 = True

hemisphere='n'

ice_name = 'h1.aice_d'
var_in = 'aice_d'   

if hemisphere=='n' or hemisphere=='north':
    hemi_path = 'north/'
if hemisphere=='s' or hemisphere=='south':
    hemi_path = 'south/'


original_runkeys = ['seaice', 'winds', 'sst']
for ogk in original_runkeys:
    try:
        x = run[ogk]
    except KeyError:
        run[ogk] = False

print('- filepaths')
    

#%% functions

def get_starting_info(ens_mem, ice_name, year, grid_path, var_in):
    
    ice_fname = glob.glob(file_path+'aice_d/'+'*'+ens_mem+'*'+ice_name+'*.'+cf.file_year(year)+'*')[0]
    ds = xr.open_dataset(ice_fname)
    ds.close()

    #### grid and final
    grid = xr.open_dataset(grid_path+'pop_grid.nc')
    TAREA = grid.TAREA.values

    var_to_keep = ds[var_in]

    ds = xr.merge([var_to_keep.drop(['TLAT', 'TLON', 'ULAT', 'ULON']),
                   grid[['TLAT', 'TLONG', 'TAREA']].rename_dims({'nlat':'nj','nlon':'ni'})],
                  compat='identical', combine_attrs='no_conflicts')
    
    si_lon = ds['TLONG'].values
    si_lat = ds['TLAT'].values
    
    grid.close()
    gc.collect()
    print('- starting sea ice info')
    
    return ds, si_lon, si_lat, TAREA, grid


def get_yearly_storm_info(startdate, enddate):
    """
    timing_grid = list of tuples: (start_date, end_date)
    storm_ranges = list of DT lists: [storm_day1, storm_day2, ... storm_day_f]
    analysis_ranges = list of DT lists: [storm_day1-one week, ... storm_day_f + two weeks]
    """
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
    
    new_storm_list = []
    for stormi in storm_ranges:
        if stormi[0].month<=10:
            new_storm_list.append(stormi)
        else:
            continue
        
    print('- census acquired')
    
    return timing_grid, analysis_ranges, storm_ranges, new_storm_list

#%%% variables
def get_seaice(year, miz_mask_list, miz_clim_list, og_miz_mask_list, og_miz_clim_list, 
               keys, daymonth_grid, ds, miz_points, miz_points_og, inside_points1000,TAREA,
               nan_list, storm_area_list, grid, stormstr, savenc):
    
    miz_mask_series = {}
    miz_mask_clim = {}
    og_mask_series = {}
    og_mask_clim = {}
    for key in keys:
        miz_mask_series[key] = []
        miz_mask_clim [key] = []
        og_mask_series[key] = []
        og_mask_clim [key] = []
    try:
       for month, day in daymonth_grid: 
           datestr = str(year)+'-'+cf.LZ(month)+'-'+cf.LZ(day)
           try:
               # get daily sea ice
               sic = ds[var_in].sel(time=datestr).squeeze()
               
               # all marginal points
               miz_masked = np.ma.masked_where(miz_points==0, sic).filled(np.nan)
               og_masked = np.ma.masked_where(miz_points_og==0, sic).filled(np.nan)
               
               ### area list
               miz_areas = []
               og_areas = []
               
               # restrict area - 1000 box  & sum
               if run1000:
                   miz_masked_1000 = np.nansum((np.ma.masked_array(miz_masked, mask=inside_points1000).filled(np.nan))*TAREA)
                   miz_areas.append(miz_masked_1000)
                   
                   og_masked_1000 = np.nansum((np.ma.masked_array(og_masked, mask=inside_points1000).filled(np.nan))*TAREA)
                   og_areas.append(og_masked_1000)
                    
                # append to timeseries
               for cc, carea in enumerate(miz_areas):
                   miz_mask_series[keys[cc]].append(carea)
                   og_mask_series[keys[cc]].append(og_areas[cc])
                   
           except Exception as e:
                # append nans to timeseries if error
                print(datestr)
                print(e) 
                for cc, carea in enumerate(nan_list): 
                    miz_mask_series[keys[cc]].append(carea)
                    og_mask_series[keys[cc]].append(carea)
           
           # calculate climatology
           for cc, inside_points in enumerate(storm_area_list):
               all_yr = []
               ogall_yr = []
               for yr in clim_years:
                   climname = glob.glob(file_path+'aice_d/'+'*'+ens_mem+'*'+ice_name+'*.'+cf.file_year(yr)+'*')[0]
                   sict = cf.load_seaice(climname, yr, month, day, pop=grid, latlon=False)
                   # all marginal points
                   try:
                       miz_mask = np.ma.masked_where(miz_points==0, sict).filled(np.nan)
                       all_yr.append( np.nansum(np.ma.masked_array(miz_mask, mask=inside_points).filled(np.nan)*TAREA) )
                       
                       og_mask = np.ma.masked_where(miz_points_og==0, sict).filled(np.nan)
                       ogall_yr.append( np.nansum(np.ma.masked_array(og_mask, mask=inside_points).filled(np.nan)*TAREA) )
                   except:
                       all_yr.append(np.nan)
                       ogall_yr.append(np.nan)
               miz_mask_clim[keys[cc]].append(np.nanmean(all_yr))
               og_mask_clim[keys[cc]].append(np.nanmean(ogall_yr))
         
       if savenc:  
            miz_mask_list.append(miz_mask_series)
            miz_clim_list.append(miz_mask_clim)
            
            og_miz_mask_list.append(og_mask_series)
            og_miz_clim_list.append(og_mask_clim)
       print('... sea ice area completed')
        
    except Exception as e:
        print('')
        print('*ERROR* ... SEAICEAREA ... ' + stormstr)
        print(e)
        print(traceback.format_exc())
        print('')
        
    return miz_mask_series, og_mask_series, miz_mask_list, miz_clim_list, og_miz_mask_list, og_miz_clim_list



def get_winds(year, total_wind_list, total_u_list, total_v_list, 
              keys, savenc, miz_list, inside_points, daymonth_grid,
              TAREA, si_lon, si_lat, storm_area_list):
    
    [miz_points,miz_points_og] = miz_list
    
    # prep vars
    TOTAL_AREA = np.ma.masked_array(TAREA, mask=inside_points)
    total_area = np.ma.masked_where(miz_points<1, TOTAL_AREA).filled(np.nan)
    total_area = np.nansum(total_area)
    
    total_area_og = np.nansum( np.ma.masked_where(miz_points_og<1, TOTAL_AREA).filled(np.nan) )
    
                    # u, v, total
    total_series = [ [], [], [] ]
    
    total_series_og = [ [], [], [] ]
    
    ### load winds
    wfiles = []
    for wind_name in ['UBOT', 'VBOT']: 
        wfilename = file_path+'winds/'+hemi_path+wind_name+'/'+'*'+ens_mem+'*'+wind_name+'*.'+cf.file_year(year)+'*'
        try:
            wfiles.append(glob.glob(wfilename)[0])
        except IndexError:
            wfiles.append('')
    
    # calcualte mean wind each day
    for month, day in daymonth_grid:
        datestr = str(year)+'-'+cf.LZ(month)+'-'+cf.LZ(day)
        
        windy_list = []
        for wfile, wind_name in zip(wfiles, ['UBOT', 'VBOT']) :
            try:
                with xr.open_dataset(wfile) as wds:
                    wind = wds[wind_name].sel(time=datestr).mean(dim='time')
                    windy_list.append(wind.values)
            except:
                windy_list.append( np.nan*np.ones((38,288)) )
            
        # total
        windy_list.append( np.sqrt( (windy_list[0]**2)+(windy_list[1]**2) ) )
        
        # restrict area    # miz points
        for tw, thiswind in enumerate(windy_list):
            # regrid to si grid
            wind_grd = griddata((lon.flatten(), lat.flatten()), thiswind.flatten(),
                                    (si_lon.flatten(), si_lat.flatten()))
            wind_grd = wind_grd.reshape(np.shape(si_lon))
            
            wind_in = np.ma.masked_array(wind_grd, mask=inside_points).filled(np.nan)
            wind_miz = np.ma.masked_where(miz_points<1, wind_in).filled(np.nan)
            wind_miz_og = np.ma.masked_where(miz_points_og<1, wind_in).filled(np.nan)

            # wind timeseries
            total_series[tw].append(np.nansum(wind_miz*TAREA)/total_area)
            total_series_og[tw].append(np.nansum(wind_miz_og*TAREA)/total_area_og)
     
    if savenc:
         try:     
            total_u_list.append([total_series_og[0], total_series[0]])
            total_v_list.append([total_series_og[1], total_series[1]])
            total_wind_list.append([total_series_og[-1], total_series[-1]])
            print('... winds completed')
         except:
            print(year, daymonth_grid[7], '... issue with appending winds to nc')

    return total_u_list, total_v_list, total_wind_list



def get_sst(sst_list, year, ens_mem, ds1, daymonth_grid,
            keys, savenc, nan_list
            ):
    
    miz_points = ds1['miz_points'].values
    miz_points_og = ds1['miz_points_og'].values
    inside_points = ds1['inside_points'].values
    TAREA = ds1['TAREA'].values
    
    # [miz_points,miz_points_og] = miz_list
    sst_series = {}
    for key in keys:
        sst_series[key] = []
    
    try: # hemi_path (!)
        sst_file = glob.glob(file_path+'sst/'+hemi_path+'*'+ens_mem+'*'+'SST'+'*.'+cf.file_year(year)+'*')[0]
        with xr.open_dataset(sst_file) as sst_ds:
            SST = sst_ds['SST']
            
    except IndexError:
        print('> no SST file: '+str(year))
        # append nan
        for month, day in daymonth_grid:
            for cc, carea in enumerate(nan_list): 
                sst_series[keys[cc]].append([carea,carea])
        if savenc:
            sst_list.append(sst_series)
        return sst_list
        
    TOTAL_AREA = np.ma.masked_array(TAREA, mask=inside_points)
    total_area = np.ma.masked_where(miz_points<1, TOTAL_AREA).filled(np.nan)
    total_area = np.nansum(total_area)
    total_area_og = np.nansum( np.ma.masked_where(miz_points_og<1, TOTAL_AREA).filled(np.nan) )
    
        
    for month, day in daymonth_grid: 
        dt = datetime(year, month, day)
        dt = dt+ timedelta(days=1) # recorded on next day
        datestr = dt.strftime('%Y-%m-%d')
        str(year)+'-'+cf.LZ(month)+'-'+cf.LZ(day)
        try:
            # get daily sea ice
            sst = SST.sel(time=datestr).squeeze()
            
            # all marginal points
            sst_miz = np.ma.masked_where(miz_points==0, sst).filled(np.nan)
            sst_miz_og = np.ma.masked_where(miz_points_og==0, sst).filled(np.nan)
            
            ### area list
            sst_group = []
            # restrict area - 1000 box  & sum
            if run1000:
                sst_inside = np.ma.masked_array(sst_miz, mask=inside_points).filled(np.nan)
                sst_inside_og = np.ma.masked_array(sst_miz_og, mask=inside_points).filled(np.nan)
                
                sst_group.append( [np.nansum(sst_inside_og*TAREA)/total_area_og,
                                   np.nansum(sst_inside*TAREA)/total_area]
                                 )
                
             # append to timeseries
            for cc, csst in enumerate(sst_group):
                sst_series[keys[cc]].append(csst)
                
        except Exception as e:
             # append nans to timeseries if error
             print(datestr)
             print(e) 
             for cc, carea in enumerate(nan_list): 
                 sst_series[keys[cc]].append([carea,carea])
    if savenc:
         try:
             sst_list.append(sst_series)
             print('... sst completed')
         except:
             print(year, daymonth_grid[7], '... issue with appending sst to nc')
             
    return sst_list


''' ???
>> area tendency (dyn/thermo)
- upper ocean 
''' 


#%% main function
# * START YEAR LOOP *
    
def main_run(year, startdate, enddate, savenc):
    
    TIMESTART = timeIN.time()
    
    
    original_runkeys = ['seaice', 'winds', 'sst','sst_daily', 'glorys', 'air_temp']
    for ogk in original_runkeys:
        try:
            run[ogk]
        except KeyError:
            run[ogk] = False
            
    print(''); 
    print('********')
    print('* '+str(year)+' *'+' ('+ens_mem+')') 
    print('********')   
    print(run); print('')
    
    ### starting sea ice info
        
    ds, si_lon, si_lat, TAREA, grid = get_starting_info(ens_mem, ice_name, year, grid_path, var_in)
            
    #### get storm info for this year
    
    timing_grid, analysis_ranges, storm_ranges, new_storm_list = get_yearly_storm_info(startdate, enddate)
    
    #### set up export nc
    
    analyrange_save, stormrange_save = [],[]
    
    miz_mask_list = []
    miz_clim_list = []
    
    og_miz_mask_list = []
    og_miz_clim_list = []
    
    total_wind_list = []
    total_u_list = []
    total_v_list = []
    
    sst_list = []
    
    ### initial xarray
    # define coordinates
    time_list = np.arange(-7,14+1,1)
    coords = {'time': (['time'], time_list),
              'nstorms':(['nstorms'], np.arange(1,len(storm_ranges)+1)),
              'miz_type':(['miz_type'], [0,1])
              }
    
    # define data with variable attributes
    data_vars = {}
    attrs = {'creation_date':datetime.now().strftime('%Y-%m-%d'), 
             'title':'Exported timeseries and storm info for ' + str(year),
             'miz_types':'0 = original (15/80), 1 = rms-miz'
             }
    
    print('- data export set up')
    
    #### * START STORM LOOP *
    print(' ')
    stormstr_prev=''
    dupe = iter(list(string.ascii_lowercase))
    for storm_num, storm_event in enumerate(storm_ranges): 
        print(str(year)+': Storm ' + str(storm_num+1) + '/' + str(len(new_storm_list)), flush = True)
        
        #### get starting storm info
        stormstr1 = storm_event[0].strftime('%Y_%m%d')
        
        # duplicate storm start date?
        if stormstr1==stormstr_prev:
            stormstr = stormstr1 + next(dupe)
            print('duplicate storm start')
        else:
            stormstr=stormstr1
        stormstr_prev = stormstr1
        
        daymonth_grid = [(dt.month, dt.day) for dt in analysis_ranges[storm_num]]
        
        storm_daymonth_grid = []
        for storm_day in storm_event:
            storm_daymonth_grid.append((storm_day.month, storm_day.day))
            
        # Save storm times
        if savenc:
            analyrange_save.append([analysis_ranges[storm_num][0].day,
                                    analysis_ranges[storm_num][-1].day])
            stormrange_save.append([storm_event[0].day, storm_event[-1].day])
            
        #### get all MIZ points
        
        t1 = storm_event[0] - timedelta(days=1)
        t2 = storm_event[-1] + timedelta(days=1)
        storm_range = cf.daterange(t1, t2, dt=24)
        
        miz_points = np.zeros(np.shape(si_lon))
        miz_points_og = np.zeros(np.shape(si_lon))
        
        for date in storm_range:
            date = date + timedelta(days=1) # data recorded next day
            try: sic = ds[var_in].sel(time=date.strftime('%Y-%m-%d')).squeeze().values
            except KeyError: 
                print('* KeyError - New sea ice file? ...', date)
                sic = np.nan * np.ones(np.shape(si_lon))
            
            miz_points = np.where(((sic>miz[0]) & (sic<=miz[1])), 1, miz_points)
            
            miz_points_og = np.where(((sic>miz_og[0]) & (sic<=miz_og[1])), 1, miz_points_og)

        #### load storm areas
        ncname = stormstr + contour_name
        cs = xr.open_dataset(nc_path+ncname)
        all_contours = []
        for key in list(cs.keys()):
            coord = cs[key].values
            all_contours.append(coord)
        cs.close()
        del cs; gc.collect()
        
        ### get bbox
        if run1000:
            with cf.HidePrint(): bbox_edges = cf.get_bbox_edges(all_contours, hemisphere) 
            inside_points1000 = cf.find_points_in_contour(bbox_edges, si_lon, si_lat)
            
        ### append masks for slicing (?)
        ds1 = ds.copy()
        ds1['inside_points']=(['nj', 'ni'], inside_points1000)
        ds1['miz_points']=(['nj', 'ni'], miz_points)
        ds1['miz_points_og']=(['nj', 'ni'], miz_points_og)
        
        ds1 = ds1.assign_coords({"ni": ds1.ni})
        ds1 = ds1.assign_coords({"nj": ds1.nj})

        ds1 = ds1.isel(nj=slice(310,400))
        
        #### * SELECT STORM AREAS
        storm_area_list = []
        keys = []
        short_keys = []
        nan_list = []
        
        if run1000: 
            storm_area_list.append(inside_points1000)
            keys.append('1000 hPa Box')
            short_keys.append('1000')
            nan_list.append(np.nan)
            
        #### --------------------------- get data
        
        
        #### SEA ICE AREA
        ### rms miz and 15/80 miz
        if run['seaice']:
            miz_mask_series, og_mask_series, miz_mask_list, miz_clim_list, \
                og_miz_mask_list, og_miz_clim_list = \
                    get_seaice(year, miz_mask_list, miz_clim_list, og_miz_mask_list, og_miz_clim_list, 
                                   keys, daymonth_grid, ds, miz_points, miz_points_og, inside_points1000,TAREA,
                                   nan_list, storm_area_list, grid, stormstr, savenc
                               )
        
        #### WINDS
        if run['winds']:
            total_u_list, total_v_list, total_wind_list = \
                get_winds(year, total_wind_list, total_u_list, total_v_list, 
                              keys, savenc, [miz_points, miz_points_og], inside_points1000, daymonth_grid,
                              TAREA, si_lon, si_lat, storm_area_list)
            
        #### SST
        if run['sst']:
            sst_list = get_sst(sst_list, year, ens_mem, ds1, daymonth_grid,
                        keys, savenc, nan_list)
            
    
    del dupe
    #### ------------------------- end indiv storm
    
    # print output for comparison
    print('')
    print('--------------------------')
    print('DONE CALCULATING:')
    print('nstorms, ', np.shape(np.arange(1,len(storm_ranges)+1)))
    print('analysis ranges, ',np.shape(analysis_ranges[storm_num]))
    if run['seaice']: print('sea ice, ', np.shape(miz_mask_series[keys[0]]))
    if run['seaice']: print('sea ice 2, ', np.shape(og_mask_series[keys[0]]))
    # if run['winds']: print('winds, ',np.shape(total_wind_series[keys[0]]))
    # if run['air_temp']: print('air temp, ',np.shape(total_temp_series[keys[0]]))
    # if run['sst_daily']: print('daily sst, ',np.shape(sst_series_totalmiz[keys[0]]))
    print('--------------------------')
    print('')
        
    #### export all storms to nc
    print('--------------------------')
    print('      EXPORT TO NC        ')
    print('--------------------------') 
    print('')
    if savenc:
        
        if run['winds']:
            def use_wind_coords(var, storm_ranges, wind_name):
                print('* using wind coords'+' ('+wind_name+'): ', np.shape(var), len(storm_ranges))
                coords = {'wind_x': (['wind_x'], np.arange(1,np.shape(var)[0]+1)),
                          'wind_y':(['wind_y'], np.arange(1,np.shape(var)[1]+1))}
                coord1, coord2 = 'wind_x', 'wind_y'
                return coords, coord1, coord2
        
        
        try:
            try:
                analysis_str = [[dt.strftime('%Y%m%d') for dt in range1] for range1 in analysis_ranges]
                stormranges_convert = [[dt.strftime('%Y%m%d') for dt in range1] for range1 in storm_ranges]
                stormrange_str = []
                for stormstrs in stormranges_convert:
                    appendstr = list(stormstrs)
                    while len(appendstr) < 22: 
                        appendstr.append('-')
                        
                    if len(appendstr)>22:
                        appendstr = appendstr[:22]
                        
                    stormrange_str.append(appendstr)
                    
                data_vars['analysis_ranges'] = (['nstorms','time'], analysis_str, 
                              {'long_name':'range of dates before/after storm that were analyzed'})
                data_vars['storm_ranges'] = (['nstorms','time'], stormrange_str, 
                              {'long_name':'storm dates'})
            except Exception as ee:
                print('')
                print('*ERROR*  START... exporting variables to nc')
                print(ee)
                print(traceback.format_exc())
                print('')
            
            if run['seaice']:
                try:
                    for kk, key in enumerate(keys):
                        sic1, clim1 = [],[]
                        short = short_keys[kk]
                        for list1, list2 in zip(miz_mask_list, miz_clim_list): 
                                sic1.append(list1[key]) 
                                clim1.append(list2[key])
                            
                        if np.shape(sic1) == (len(storm_ranges), len(time_list)):
                            coord1, coord2 = 'nstorms','time'    
                        else:
                            print('* using seaice coords (1): ', np.shape(sic1), len(storm_ranges))
                            coords = {'si_x': (['si_x'], np.arange(1,np.shape(sic1)[0]+1)),
                                      'si_y':(['si_y'], np.arange(1,np.shape(sic1)[1]+1))
                                      }
                            coord1, coord2 = 'si_x','si_y'
                            
                        data_vars['sia_miz2_'+short] = ([coord1, coord2], sic1, 
                                 {'units': 'm^2', 
                                  'long_name':'sea ice area timeseries,rms, '+key+str(miz)})
                        
                        data_vars['sia_clim_miz2_'+short] = ([coord1, coord2], clim1, 
                                 {'units': 'm^2', 
                                  'long_name':'sea ice area climatology,rms, '+key+str(miz)})
                        
                        
                        sic0, clim0 = [],[]
                        for list1, list2 in zip(og_miz_mask_list, og_miz_clim_list): 
                                sic0.append(list1[key]) 
                                clim0.append(list2[key])
                            
                        if np.shape(sic0) == (len(storm_ranges), len(time_list)):
                            coord1, coord2 = 'nstorms','time'    
                        else:
                            print('* using seaice coords (0): ', np.shape(sic0), len(storm_ranges))
                            coords = {'si_x': (['si_x'], np.arange(1,np.shape(sic0)[0]+1)),
                                      'si_y':(['si_y'], np.arange(1,np.shape(sic0)[1]+1))
                                      }
                            coord1, coord2 = 'si_x','si_y'
                            
                        data_vars['sia_miz_'+short] = ([coord1, coord2], sic0, 
                                 {'units': 'm^2', 
                                  'long_name':'sea ice area timeseries, miz, '+key,
                                 'sea_ice_limits': str(miz_og)})
                        
                        data_vars['sia_clim_miz_'+short] = ([coord1, coord2], clim0, 
                                 {'units': 'm^2', 
                                  'long_name':'sea ice area climatology, miz, '+key,
                                  'sea_ice_limits': str(miz_og)})
                        
                except Exception as ee:
                    print('')
                    print('*ERROR* SEAICE AREA... exporting variables to nc')
                    print(ee)
                    print(traceback.format_exc())
                    print('') 
                    
                 
            if run['winds']:
                try:
                    wind_list = [total_u_list, total_v_list, total_wind_list, ]
                    wind_names = ['u', 'v','total']

                    for wind_type, wind_name in zip(wind_list, wind_names):
                        wind1 = wind_type
                    
                        if np.shape(wind1) == (len(storm_ranges), 2, len(time_list)):
                            coord1, coordx, coord2 = 'nstorms','miz_type','time'
                        else:
                            coords, coord1, coord2 = use_wind_coords(wind1, storm_ranges, wind_name)
                            coordx = 'miz_type'
                            
                        data_vars[wind_name+'_winds'] = ([coord1, coordx, coord2],  wind1, 
                                 {'units': 'm/s', 
                                  'long_name':wind_name+'-wind timeseries, total miz area'})
                        
                except Exception as ee:
                    print('')
                    print('*ERROR* WINDS... exporting variables to nc')
                    print(ee)
                    print(traceback.format_exc())
                    print('')
                    
            if run['sst']:
                try:
                    for kk, key in enumerate(keys):
                        short = short_keys[kk]
                        sst = []
                        
                        for list1 in sst_list:
                            sst.append(list1[key]) 
                        
                        if np.shape(sst) == (len(storm_ranges), len(time_list), 2):
                            data_vars['sst_'+short] = (['nstorms','time','miz_type'], sst, 
                                     {'units': 'deg C', 
                                      'long_name':'sea surface temperature, original miz, '+key})
                        else:
                            print('* using sst coords: ', np.shape(sst), len(storm_ranges))
                            coords = {'sst_x': (['sst_x'], np.arange(1,np.shape(sst)[0]+1)),
                                      'sst_y':(['sst_y'], np.arange(1,np.shape(sst)[1]+1)),
                                      'sst_z':(['sst_z'], np.arange(1,np.shape(sst)[-1]+1))
                                      }
                            data_vars['sst_'+short] = (['sst_x','sst_y','sst_z'], sst, 
                                     {'long_name':'sea surface temperature, '+key})

                except Exception as ee:
                    print('')
                    print('*ERROR* SST... exporting variables to nc')
                    print(ee)
                    print(traceback.format_exc())
                    print('')
                    
                    
        except Exception as ee:
            print('')
            print('*ERROR* ... exporting variables to nc')
            print(ee)
            print(traceback.format_exc())
            print('')
            
    #### create dataset -> nc
    try:
        ds = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)
        
        ds.to_netcdf(savepath+ str(year) + ncnameadd+ '.nc')
        
        print('\n netcdf saved!')
        print(savepath+ str(year) + ncnameadd+ '.nc')

    except Exception as e:
        print('')
        print('*ERROR* ... saving nc file')
        print('--------------------------')
        
        print('analysis ranges, ',np.shape(analysis_str))
        print('storm ranges, ',np.shape(stormrange_str))
        if run['seaice']: print('sea ice, ', np.shape(sic1), np.shape(clim1))
        if run['seaice']: print('sea ice-miz, ', np.shape(sic0), np.shape(clim0))
        
        print(''); print('')
        print(e)
        print(traceback.format_exc())
        print('')
    
            
    #### end
    print(' '); print(' ')
    print('------------------')
    print('       done!      ')
    print('------------------')
    print(str(year)+' elapsed time: ')
    print(cf.rstr((timeIN.time() - TIMESTART)/60,1), 'minutes')  
    print(); print()
    

#%% run year loop

timer = timeIN.time()

for ens_mem in ensemble_members:
    ens_time = timeIN.time()
    print(); print(ens_mem); print()
    
    # path to cesm data files
    file_path = root_path + 'data/'
    grid_path = file_path
    # path to census files and storm information
    nc_path = root_path +'out_'+str(hemisphere)+'_'+str(ens_mem)+'/'
    contour_name = '_contours.nc'
    census_path = nc_path
    census_name = 'census_test1_all_'
    # output path
    savepath = nc_path+savedir
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    ###!!! check which miz in winds/sst
    miz = get_miz_values(census_path) # check readme (rms)
    miz_og = [0.15,0.80] 
    
    if run['winds']:
        samplewindname = 'b.e21.BHISTsmbb.f09_g17.LE2-1231.011.cam.h2.VBOT.2010010100-2014123100.nc'
        wind = xr.open_dataset(file_path+'winds/'+hemi_path+'VBOT/'+samplewindname)
        lon, lat = np.meshgrid(wind['lon'].values, wind['lat'].values)

    for year in years:
        if year in np.arange(2010,2019+1,1):
            clim_years = np.arange(2010,2019+1,1)
        elif year in np.arange(1982,1991+1):
            clim_years = np.arange(1982,1991+1)
        elif year in [1980, 1981]:
            clim_years = np.arange(1980,1990)
        elif year in np.arange(2000,2009+1,1):
            clim_years = np.arange(2000,2009+1,1)
        elif year in np.arange(1990,1999+1,1):
            clim_years = np.arange(1990,1999+1,1)
        elif year in np.arange(2020, 2030):
            clim_years = np.arange(2020, 2030)
        elif year in np.arange(2030, 2040):
            clim_years = np.arange(2030, 2040)
        elif year in np.arange(2040, 2050):
            clim_years = np.arange(2040, 2050)
        elif year in np.arange(2050, 2060):
            clim_years = np.arange(2050, 2060)
        elif year in np.arange(2060, 2070):
            clim_years = np.arange(2060, 2070)
        elif year in np.arange(2070, 2080):
            clim_years = np.arange(2070, 2080)
        elif year in np.arange(2080, 2090):
            clim_years = np.arange(2080, 2090)
        elif year in np.arange(2090, 2100):
            clim_years = np.arange(2090, 2100)
            
        if run_sample: # run with sample code (one sample data file)
            clim_years = [2015,2016,2017]
        
        ### get start/end dates + add'l time
        try:[startdate, enddate] = cf.readCensus(census_path+census_name + str(year)+'.csv' , convertDT=True)[0]
        except ValueError as ve:
            print(ve); print('Empty Census: '+str(year))
            continue
        
        main_run(year, startdate, enddate, savenc)
    
    print(ens_mem+': '+ str(round((timeIN.time()-ens_time)/60, 1))+' minutes')

print();print(); print('--> '+str(round((timeIN.time()-timer)/60, 1))+' minutes')


#%% end





