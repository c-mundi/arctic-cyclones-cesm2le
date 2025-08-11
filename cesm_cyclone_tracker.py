# -*- coding: utf-8 -*-
"""
Feb, July 2024 - Update August 2024
Prepared for publication - 08 Aug 2025

cesm_cyclone_tracker3.py
-- better contour detection
-- add breakdown of storm thresholds

turn on/off thresholds for comparison: miz ice, duration
> min slp threshold
> ice criteria

~ 2 min per run year

@author: mundi
"""
import time as timeIN
TIMESTART = timeIN.time()

root_path = '/Users/mundi/Desktop/cesm-code/'

run_rms_analysis = True

#%% inputs
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import gc, os, sys
import string

import warnings
warnings.simplefilter('error', UserWarning)

years = list(np.arange(1982 ,1991+1)) + list(np.arange(2010,2019+1) )

e_prefix = ['1231','1251','1281','1301'] 
e_range = np.arange(11,20+1,1)
ensemble_members = [str(ep)+'.0'+str(er) for ep in e_prefix for er in e_range]

hemisphere = 'n' 

min_pres = 986

months = [5,6,7,8,9]
contour_intervals = [990+2, 1000+2]
data_dir = 'north/'

ice_name = 'h1.aice_d'
psl_name = 'h2.PSL' # 6 hr
census_name = 'census_test1'

### SAMPLE
# years=[2015]
# ensemble_members = ['1301.020']


#%% pressure threshold for storm lifetime
# weaker min pressure at storm start and storm final (984)
pres_thresh = 6

### how far apart (degrees) two points below pressure threshold need to be to count as different storms
## used for comparing many points below threshold for single slp profile (15)
grouping_thresh = 15

### how far apart (degrees) two absolute minima need to be to count as different storms
## used for comparing single points from different slp profiles (50)
sorting_thresh = 40

### time between storms
timediff_thresh = timedelta(hours=12)

### minimum storm duration
min_duration = timedelta(days=2)

### distance between storm min and ice edge (km)
icedist_thresh = 1250 

### ice area percentages
ice_lims = [20,80] ############ #[0,100] #[10,90] #
ice_criteria = [80,'cell'] ##########[15, 'cell'] # #15/80, 'cell'/'ice'

if (ice_criteria[0] not in [15,80]) or (ice_criteria[1] not in ['ice', 'cell']):
    raise ValueError('ice_criteria = '+str(ice_criteria))


#%% imports
import calendar
import matplotlib.pyplot as plt
import glob
import csv
    
import cesm_functions as cfx
import cyctrack_functions as ctrack

header = ['start', 'start_lat', 'start_lon', 'minimum', 'finish', 'finish_lat', 'finish_lon']

#%% fxns
class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def export2nc(nc_in, all_contours, all_cont_dts, intervals = [990, 1000]):
    import netCDF4
    import traceback
    
    ncfile = netCDF4.Dataset(nc_in,mode='w',format='NETCDF4_CLASSIC')
    try:
        titlestr=''
        for interval in intervals:
            titlestr += str(interval)
            titlestr += ', '
        ncfile.title= titlestr[:-2] + ' mb pressure over storm days'
        
        ncfile.createDimension('num_coords', 2)
        prev_stormstr=''
        num=0
        
        for c, cont in enumerate(all_contours):
            stormstr = all_cont_dts[c].strftime('%Y_%m%d')
            
            if c == len(all_cont_dts)-1:
                if all_cont_dts[c] == all_cont_dts[c-1]:
                    pstr=str(intervals[1])
                else: pstr=str(intervals[0])
            elif (all_cont_dts[c+1] != all_cont_dts[c]):
                pstr = str(intervals[1])
            else:
                pstr = str(intervals[0])
                
            if stormstr+'_'+pstr == prev_stormstr+'_'+pstr:
                num+=1
            else:
                num=0
            
            ## create dimensions
            ncfile.createDimension('points'+str(c+1), len(cont))
            
            try:
                pres_edges = ncfile.createVariable('coords_' +stormstr+'_'+pstr+'_'+str(num),
                                               np.float64, ('points'+str(c+1),'num_coords'))
            except RuntimeError as rte:
                print(rte)
                print(traceback.format_exc())
                print('~~~  '+'coords_' +stormstr+'_'+pstr+'_'+str(num)+ '  ~~~')
            
            prev_stormstr = stormstr
                
            print('coords_' +stormstr+'_'+str(intervals[c%2]) )
            pres_edges.long_name = 'Coordinates for ' + str(intervals[c%2]) +'mb on ' + stormstr
            pres_edges[:,:] = cont # load data to nc
            
            
    except Exception as e:
        print(''); print('* ERROR: exporting to nc')
        print(e)
        print(traceback.format_exc())
    finally:
        ncfile.close()
        print("*exported*")


def get_storm_area(year, all_contours, storm_event, ice_fname, ice_criteria, miz1 = [0.15,0.80]):
    ### get bbox_edges
    with HidePrint():
        bbox_edges = cfx.get_bbox_edges(all_contours, hemisphere) 
        
    for miz_idx in [0,1]:
        if miz1[miz_idx] > 1: 
            print('MIZ conversion: '+ str(miz1[miz_idx])+' -> '+str(miz1[miz_idx]/100))
            miz1[miz_idx] = miz1[miz_idx]/100
        
    out = {}
        
    print('Retrieved bbox')
    
    ### area of bbox
    # get si lon/lat grid
    si_grid, si_lon, si_lat = cfx.load_seaice(ice_fname, year, storm_event[0].month, storm_event[0].day, pop=grid, latlon=True)
    si_grid = si_grid.values
    collected=gc.collect()
    print('Loaded sea ice: '+str(collected))# get box mask
    # get box mask
    in_mask = cfx.find_points_in_contour(bbox_edges, si_lon, si_lat)
    # calculate area
    si_in = np.ma.masked_array(si_grid, mask=in_mask).filled(np.nan)
    
    
    box_area = cfx.get_total_area(si_in, pop=TAREA)
    ice_area15 = cfx.calc_seaice_area(np.where(si_in<=miz1[0], 0, si_in), pop=TAREA) 
    ice_area80 = cfx.calc_seaice_area(np.where(si_in<miz1[1], 0, si_in), pop=TAREA)
    
    ### export info
    out['box_area'] = box_area
    out['ice_area15'] = ice_area15
    out['ice_area80'] = ice_area80
    
    ### calculate icearea fraction 
    if ice_criteria[0] == 80:
        ice_frac = ice_area80*100/box_area 
    elif ice_criteria[0] == 15:
        ice_frac = ice_area15*100/box_area 
    else:
        print('> get_storm_area(): ice_criteria ['+str(ice_criteria[0])+']')
        raise(ValueError)        

    return ice_frac, out, bbox_edges


def diagnostic_print(dates, duration, pressures, ice_distance):
    print('Removing Storm: '+ str(dates[0])+'-'+str(dates[-1]))
    if (duration < min_duration): print('Duration: '+str(duration))
    if (np.min(pressures) > min_pres): print('Pressure: '+str(np.min(pressures)))
    if (ice_distance > icedist_thresh): print('Ice Distance: '+str(ice_distance))
    print('-')

def final_storm_check(starting_list, stormtime, all_storms, ice_fname):
    
    starting_array = np.array(starting_list)
    # check final pressure of previous grouping
    pressures = starting_array[:,2]
    # and that storm duration is long enough
    dates = starting_array[:,-1]
    duration = dates[-1] - dates[0]
    # and close enough to ice to be considered
    storm_lat = np.mean(starting_array[:,1]) # average storm latitude
    ice_distance = cfx.get_storm_distance(ice_fname, dates[0], storm_lat, pop=grid, bbox_edges=[])
    
    if (duration < min_duration) or (np.min(pressures) > min_pres) or (ice_distance > icedist_thresh):
        # diagnostic_print(dates, duration, pressures, ice_distance)
        starting_list = [stormtime] # throw out, reset list
        return [stormtime], all_storms #continue
    # append old list; make new
    all_storms.append(starting_list)
    starting_list = [stormtime]
    
    return starting_list, all_storms

def final_storm_check_and_export(starting_list, stormtime, all_storms, ice_fname, CENSUS_INFO):
    
    starting_array = np.array(starting_list)
    # check final pressure of previous grouping
    pressures = starting_array[:,2]
    # and that storm duration is long enough
    dates = starting_array[:,-1]
    duration = dates[-1] - dates[0]
    # and close enough to ice to be considered
    storm_lat = np.mean(starting_array[:,1]) # average storm latitude
    ice_distance = cfx.get_storm_distance(ice_fname, dates[0], storm_lat, pop=grid, bbox_edges=[])
    
    if (duration < min_duration) or (np.min(pressures) > min_pres) or (ice_distance > icedist_thresh):
        
        if (duration < min_duration):
            filename = nc_save_path+census_name +'_duration_'+str(year)+'.csv'
        if (np.min(pressures) > min_pres):
            filename = nc_save_path+census_name +'_pressure_'+str(year)+'.csv'
        if (ice_distance > icedist_thresh):
            filename = nc_save_path+census_name +'_ice_distance_'+str(year)+'.csv'
        
        with open(filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CENSUS_INFO)
        
        starting_list = [stormtime] # throw out, reset list
        return [stormtime], all_storms #continue
    # append old list; make new
    all_storms.append(starting_list)
    starting_list = [stormtime]
    
    return starting_list, all_storms

def LZ(day):
    if day>=10:
        return str(day)
    elif day<10:
        return '0'+str(day)

#%% find locations of minimum pressures
# returns all_storms: lon/lat/minpressure each day

def main(year, ens_mem):
    
    # print('................................................... HEADER! ! !')
    for census_type in ['duration', 'pressure', 'ice_distance']:
        with open(nc_save_path+census_name +'_'+census_type+'_'+str(year)+'.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header) # write the header
    
    collected=gc.collect()
    
    storm_sorter = {}
    sort_count = 0
    
    ice_fname = glob.glob(file_path+'aice_d/'+'*'+ens_mem+'*'+ice_name+'*.'+cfx.file_year(year)+'*')[0]
    slp_fname = glob.glob(file_path+'psl/'+data_dir+'*'+ens_mem+'*'+psl_name+'*.'+cfx.file_year(year)+'*')[0]
    
    for month in months: 
        collected=gc.collect()
        plt.close('all')
        day_counter = -1
        my_storms_daily = []
        
        # load monthly data 
        calendar_days = list(np.arange(1,calendar.monthrange(year, month)[1]+1))
        
        if month==2 and calendar_days[-1]==29: calendar_days = calendar_days[:-1] # 365 days in model always
        
        lon, lat, MSL, TIME = cfx.load_psl(slp_fname, year, month, calendar_days, loc=hemisphere)
        
        lon = np.where(lon<0, lon+360, lon)
        
        for day in calendar_days: 
            day_counter += 1    
            current_date = datetime(year, month, day)
            # datestr = current_date.strftime('%Y%m%d')
    
            # get daily value
            msl = MSL.sel(time=str(year)+'-'+LZ(month)+'-'+LZ(day)).values
            time = TIME.sel(time=str(year)+'-'+LZ(month)+'-'+LZ(day)).values
    
            ### get storm locations
            hour_counter = -1
            my_storms_hourly = {}
            for tind, slp in enumerate(msl): # loop through each hour
                hour_counter += 1 
                my_storms_hourly[hour_counter] = []
                
                # find indices of points of low pressure
                indp = np.where(slp < min_pres + pres_thresh) 
                if slp[indp].size == 0: 
                    continue
    
                # sort by longitudes to group different storm events
                sorted_lons, sorted_lats, sorted_slp = zip(*sorted(zip(lon[indp], lat[indp], slp[indp])))
                sorted_lons = np.array(sorted_lons)
                sorted_lons = np.where(sorted_lons<0, sorted_lons+360, sorted_lons)
                
                ### group detected low pressure points by location
                ### find abs minimum for each pressure grouping
                # --- check lon for comparison
                for lx, ll in enumerate(sorted_lons):
                    l1=ll
                    l2 = l2 = sorted_lons[lx-1]  
            
                    # --------------------------
                    if lx==0: 
                        starting_list = [[ll, sorted_lats[lx], sorted_slp[lx]]]
                        continue
                    if abs(l1-l2) <= grouping_thresh: # same storm grouping if pressure mins are within this many degrees
                        starting_list.append([ll, sorted_lats[lx], sorted_slp[lx]])
     
                    if abs(l1-l2) > grouping_thresh or lx == len(sorted_lons)-1: 
                        # if greater than that difference, get absolute min of previous grouping then make new group
                        # OR get minimum of last grouping
                        starting_array = np.array(starting_list)
                        maxslp, maxlon, maxlat = zip(*sorted(zip(starting_array[:,2], starting_array[:,0], starting_array[:,1])))
                        my_storms_hourly[hour_counter].append([maxlon[0],maxlat[0],maxslp[0], time[tind]])
                        # start new grouping
                        starting_list = [[ll, sorted_lats[lx], sorted_slp[lx]]] 
                starting_list= None
            # append mins detected each hour
            my_storms_daily.append(my_storms_hourly)
           
        ### sort storm locations for the day
        sortnum=0
        for hourlyinfo in my_storms_daily:
            for hr in hourlyinfo:
                for si, storm_list in enumerate(hourlyinfo[hr]):
                    if sort_count != 0:
                        sortnum = 0
                        while sortnum < sort_count: #"loops" thru identified storms and matches up lons
                            
                            l1 = storm_list[0]
                            l2 = storm_sorter[sortnum][-1][0]
                            
                            if abs(l1-l2) <= sorting_thresh:
                                storm_sorter[sortnum].append(storm_list)
                                added=True
                                break
                            else:
                                sortnum+=1
                                if sortnum >= sort_count:
                                    added=False
                                    break
                        if not added:
                            storm_sorter[sort_count] = [storm_list]
                            added = False
                            sort_count += 1
                    elif sort_count == 0:
                        storm_sorter[sort_count] = [storm_list]
                        sort_count += 1
    
    ### storm_sorter --> all_storms
    
    #%% chunk storms and organize
    all_storms = []
    
    startdates, enddates = [], []
    censusinfo = []
    storm_info = []
    
    for stormkey in storm_sorter:
        min_p = 99999
        
        event = np.array(storm_sorter[stormkey])
        startdates.append(event[0][-1])
        enddates.append(event[-1][-1])
        # find minimum pressure
        pressures = event[:,2]
        minimum = np.min(pressures)
        
        # start, start_lat, start_lon, minimum, 
        # finish, finish_lat, finish_lon
        censusinfo.append([event[0][-1], event[0][1], event[0][0], minimum, 
                           event[-1][-1], event[-1][1], event[-1][0] ])
        
        starting_list = []
        # loop through each individual point identified in each preliminary grouping
        for sidx, stormtime in enumerate(storm_sorter[stormkey]):
            if sidx == 0: 
                starting_list = [stormtime]
                continue
            # compare timestamp with previous timestamp (identify same storm grouping based on time)
            # if abs(stormtime[-1] - storm_sorter[stormkey][sidx-1][-1])<timediff_thresh:
            if abs(stormtime[-1] - starting_list[-1][-1])<timediff_thresh:
                starting_list.append(stormtime)
                if stormtime[2] < min_p: min_p = stormtime[2]
                
                # if final
                if sidx==len(storm_sorter[stormkey])-1:
                   # starting_list, all_storms = final_storm_check(starting_list, stormtime, all_storms, ice_fname)
                   
                   starting_list = np.array(starting_list)
                   if min_p>2000: min_p = starting_list[0][2]
                   this_census = [starting_list[0][-1], starting_list[0][1], starting_list[0][0],
                                  min_p,
                                  starting_list[-1][-1], starting_list[-1][1], starting_list[-1][0]]
                   starting_list, all_storms = final_storm_check_and_export(starting_list, stormtime, all_storms, ice_fname, this_census)
                   min_p = 99999
                   
                   if len(starting_list) == 1: continue
    
            else: # split storm grouping
                if len(starting_list) == 0: 
                    continue
                # starting_list, all_storms = final_storm_check(starting_list, stormtime, all_storms, ice_fname)
                
                starting_list = np.array(starting_list)
                if min_p>2000: min_p = starting_list[0][2]
                this_census = [starting_list[0][-1], starting_list[0][1], starting_list[0][0],
                               min_p,
                               starting_list[-1][-1], starting_list[-1][1], starting_list[-1][0]]
                starting_list, all_storms = final_storm_check_and_export(starting_list, stormtime, all_storms, ice_fname, this_census)
                min_p = 99999
                
                if len(starting_list) == 1: continue
            
            
    #%% check for duplicate times - remove adjacent mins within the same storm
    all_storms1 = []
    stormnum=0
    dupes=None
    still_duplicates, first_time, checked_storm = None, None, None
    for storm in all_storms:
        stormnum+=1
        checked_storm=[]
        first_time=True
        still_duplicates=True
        while still_duplicates:
            still_duplicates=False
            if first_time:
                working_storm = storm
                first_time=False
            else:
                working_storm = checked_storm
                checked_storm=[]
            # psuedo for loop
            ip = 0 
            while ip < len(working_storm)-1:
                if ip==0:
                    prev_date = working_storm[ip][-1]
                    ip+=1
                if working_storm[ip][-1] == prev_date:
                    still_duplicates=True
                    dupes=[working_storm[ip-1]]
                    # print('dupe found!')
                    # get true min
                    while working_storm[ip][-1] == prev_date:
                        # print('dupppees')
                        dupes.append(working_storm[ip])
                        ip+=1
                        if ip == len(working_storm)-1: break
                    dupes = np.array(dupes)
                    # sort by pressure and select min
                    dupes = dupes[dupes[:, 2].argsort()]
                    checked_storm.append(list(dupes[0]))
                else:
                    checked_storm.append(working_storm[ip-1])
                prev_date = working_storm[ip][-1]
                ip+=1
           
        working_storm = checked_storm
        all_storms1.append(working_storm)
        
    all_storms = all_storms1.copy()
    all_storms = sorted(all_storms, key=lambda t: t[0][-1]) # sort by starting date
    
    
    #%% clear out vars
    try:
        del added, all_storms1, current_date, day_counter, dupes # storm_sorter
        del hour_counter, indp, l1, l2, ll, lx, maxlat, maxlon, maxslp
        del my_storms_daily, my_storms_hourly, si, sidx, slp, sort_count # prev_date
        del sorted_lats, sorted_lons, sorted_slp, sortnum, starting_array
        del starting_list, still_duplicates, storm, storm_list, checked_storm
        del stormkey, stormnum, tind, working_storm, first_time
    except (UnboundLocalError, NameError) as EE:
        print('Could not delete variables')
        print(EE)
        
    collected = gc.collect()
    print() 
    print(":: "+str(year)+" Variable clearout: collected", "%d objects." % collected)
    print() 
    
    #%% get census info for export
    
    startdates, enddates = [], []
    censusinfo = []
    storm_info = []
    for storm_event in all_storms:
        event = np.array(storm_event)
        startdates.append(event[0][-1])
        enddates.append(event[-1][-1])
        
        # find minimum pressure
        pressures = event[:,2]
        minimum = np.min(pressures)
        
        # start, start_lat, start_lon, minimum, 
        # finish, finish_lat, finish_lon
        censusinfo.append([event[0][-1], event[0][1], event[0][0], minimum, 
                           event[-1][-1], event[-1][1], event[-1][0] ])
        
        
        info1 = [] # all hours of the day, to be averaged
        info2 = [] # average of each day for every storm 
        for tidx, timing in enumerate(event):
            if tidx==0: 
                info1 = [timing[0:3]]
                continue
            if timing[-1].day == event[tidx-1][-1].day: #if same day as previous entry
                info1.append(timing[0:3])
                if tidx == len(event)-1:
                    info2.append(np.mean(info1,axis=0))
            else:
                info2.append(np.mean(info1,axis=0))
                info1 = [timing[0:3]]
        
        storm_info.append(info2)
        info2 = []
        
    #%% begin storm_id
    timing_grid = []
    for xx in range(0,len(startdates)):
        timing_grid.append((startdates[xx], enddates[xx]))
    
    storm_ranges = []
    analysis_ranges = []
    for startdt, enddt in timing_grid:
        week_ago = startdt - timedelta(days=7)
        two_week = startdt + timedelta(days=14) # startdt vs end (same duration for all storms, 3wks)
        analysis_ranges.append(cfx.daterange(week_ago, two_week, dt=24))
        storm_ranges.append(cfx.daterange(startdt, enddt, dt=24))
    
    try: del event, info1, info2, tidx, timing, startdates, enddates, week_ago, two_week
    except (UnboundLocalError, NameError) as EE:
        print('Could not delete variables')
        print(EE)
    collected = gc.collect()
    
    #%%% * START STORM LOOP * get contours
    ### set up ice area export
    myvars= ['box_area', 'ice_area15', 'ice_area80']
    datavars, outvars = {}, {}
    for var in myvars:
          outvars[var]=[]
    
    print(' '); print(' ')
    stormstr_prev = ''
    new_storm_info = []
    new_all_storms = []
    
    
    dupe = iter(list(string.ascii_lowercase))
    for storm_num, storm_event in enumerate(storm_ranges):        
        print();print('-');
        print('Storm ' + str(storm_num+1) + '/' + str(len(storm_ranges)))
        print('-')
        print("- Garbage collector: collected", "%d objects." % collected)
        
        ### get needed contours and create box
        stormstr1 = storm_event[0].strftime('%Y_%m%d')
        daymonth_grid = [(dt.month, dt.day) for dt in analysis_ranges[storm_num]]
            
        print('-- contours')
        
        storm_daymonth_grid = []
        for storm_day in storm_event:
            storm_daymonth_grid.append((storm_day.month, storm_day.day))
    
        try:
            all_cont_dts, all_contours = ctrack.detect_contours2(storm_daymonth_grid, 
                                                              daymonth_grid, year, 
                                                              storm_info=storm_info[storm_num],
                                                              intervals=contour_intervals,
                                                              local_file=slp_fname, loc=hemisphere) 
        except:
            all_cont_dts, all_contours = [],[]
            print('$$$ ERROR = detect_contours, ', storm_daymonth_grid[0])
            print('- len storm_info: '+str(len(storm_info)))
            print('- storm_num: ' +str(storm_num))
            
        #### garbage?
        collected = gc.collect()
        print("- Garbage collector: collected", "%d objects." % collected)
        
        #### export
        if stormstr1==stormstr_prev:
            try:
                stormstr = stormstr1 + next(dupe)
                print('Dupe: '+stormstr)
            except StopIteration:
                continue
        else:
            stormstr=stormstr1
        stormstr_prev = stormstr1
    
        ### export *.nc!
        nc_name = stormstr+'_contours.nc'
        nc_in = nc_save_path+ nc_name
        export2nc(nc_in, all_contours, all_cont_dts, intervals=contour_intervals)
        
        
        ice_frac, out, bbox_edges = get_storm_area(year, all_contours, storm_event, ice_fname, ice_criteria, miz)
        new_storm_info.append(storm_info[storm_num])
        new_all_storms.append(all_storms[storm_num])
        for var in myvars:
            outvars[var].append(out[var])
        collected=gc.collect()
        
    del dupe
    
    ### update storm list
    storm_info = new_storm_info 
    all_storms = new_all_storms
    coords = {'nstorms':(['nstorms'], np.arange(1,len(storm_ranges)+1))
              }
    
    #### export ice frac info to nc
    for var in myvars:
        datavars[var] = (['nstorms'], outvars[var])
    ds = xr.Dataset(data_vars=datavars, 
                    coords=coords)
    
    ds.to_netcdf(nc_save_path+str(year)+'_area.nc')
    
    #%% export storm_info to new census database
    data = []
    for storm_num, storm_row in enumerate(storm_info):
        
        storm_data = [storm_ranges[storm_num][0]] # start date
        
        storm_data.append(round(storm_row[0][1],4)) # start lat
        storm_data.append(round(storm_row[0][0],4)) # start lon
        
        # find storm minimum pressure
        min_p = 9999
        for day_entry in storm_row:
            if day_entry[-1] < min_p:
                min_p = day_entry[-1]
        storm_data.append(round(min_p,3)) 
        
        storm_data.append(storm_ranges[storm_num][-1]) # finish date
        storm_data.append(round(storm_row[-1][1],4)) # finish lat
        storm_data.append(round(storm_row[-1][0],4)) # finsih lon
        
        # add data to full list for export
        data.append(storm_data)
    
    ### create csv and export
    with open(nc_save_path+census_name +'_all_'+str(year)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
    
        # write multiple rows
        writer.writerows(data)
        
    #%% filter storms 

    storm_counts_early = {'all':0, 'good_ice':0, 'bad_ice':0}
    storm_counts_late = {'all':0, 'good_ice':0, 'bad_ice':0}
    ice_header_tf = False
    stormstr_prev = ''
    dupe = iter(list(string.ascii_lowercase))
    for storm_num, storm_event in enumerate(storm_ranges):
        stormstr1 = storm_event[0].strftime('%Y_%m%d')
        if storm_event[0].month in months[0:2]: storm_counts_early['all']+=1
        elif storm_event[0].month in months[-2:]: storm_counts_late['all']+=1
        
        ### open storm contours
        if stormstr1==stormstr_prev:
            stormstr = stormstr1 + next(dupe)
        else:
            stormstr=stormstr1
        stormstr_prev = stormstr1
    
        nc_name = stormstr+'_contours.nc'
        cs = xr.open_dataset(nc_save_path+nc_name)
        all_contours = []
        for key in list(cs.keys()):
            coord = cs[key].values
            all_contours.append(coord)
        cs.close()
        
        #### add in interaction with sea ice [20,80]
        ice_frac, out, bbox_edges = get_storm_area(year, all_contours, storm_event, ice_fname, ice_criteria, miz)
        print(ice_frac)
        
        ice_fname = glob.glob(file_path+'aice_d/''*'+ens_mem+'*'+'aice_d'+'*.'+cfx.file_year(year)+'*')[0]
        ices, si_lon, si_lat = cfx.load_seaice(ice_fname, int(year),
                                             storm_event[0].month,storm_event[0].day, 
                                             pop=grid,latlon=True)
        collected = gc.collect()
        print("- Garbage collector: collected", "%d objects." % collected)
        
        if (ice_frac<np.min(ice_lims) or ice_frac>np.max(ice_lims)):
            print('--> ice interaction: '+str(round(ice_frac,2))+'%'); print();print()
            if storm_event[0].month in months[0:2]: storm_counts_early['bad_ice']+=1
            elif storm_event[0].month in months[-2:]: storm_counts_late['bad_ice']+=1
            
            with open(nc_save_path+census_name +'_removed-icethresh_'+str(year)+'.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                if not ice_header_tf:
                    writer.writerow(header) # write the header
                    ice_header_tf = True
                writer.writerow(censusinfo[storm_num])  # write multiple rows
            continue
        else: 
            new_storm_info.append(storm_info[storm_num])
            new_all_storms.append(all_storms[storm_num])
            if storm_event[0].month in months[0:2]: storm_counts_early['good_ice']+=1
            elif storm_event[0].month in months[-2:]: storm_counts_late['good_ice']+=1
            for var in myvars:
                outvars[var].append(out[var])
        collected=gc.collect()
        
    # record storm ice ratios
    with open(nc_save_path+'readme.txt', 'a') as f:
        f.write(str(year)+' early-summer ice fraction: '+str(storm_counts_early)+'\n')
        f.write(str(year)+' late-summer ice fraction: '+str(storm_counts_late)+'\n')
        f.write(str(year)+' detected storms: '+str(len(storm_ranges))+'\n')
        f.write('\n')
    
    #%% pickle some variables for later
    print(''); print('FINISHED STORM DATA COLLECTION: '+str(year)); print('')
    
    ### save important stuff for later
    import pickle
    
    output = open(nc_save_path + str(year)+'_all_storms.pkl', 'wb')
    pickle.dump(all_storms, output)
    output.close()
    
    collected = gc.collect()


#%% RUN 
# !!!!

for ens_mem in ensemble_members: 
    # path to cesm data files
    file_path = root_path+'data/'
    grid_path = file_path
    # path to satellite sea ice observations (for RMS calculations)
    # si_fname = '/boltzmann/data5/arctic/NOAA_CDR_SIC_V4/daily/'
    si_fname = root_path+'seaice/'
    # output path
    nc_save_path = root_path+'out_'+str(hemisphere)+'_'+ens_mem+'/'
    
    # make save directory
    if not os.path.exists(nc_save_path):
           os.makedirs(nc_save_path)

    #### load pop grid
    grid = xr.open_dataset(grid_path+'pop_grid.nc')
    TAREA = grid.TAREA.values
    grid.close()
    garbage = gc.collect()
    
    #### get comparative MIZ
    if run_rms_analysis:
        from rms_si_fxn import rms_si
        conc15, rms15 = rms_si(15, ens_mem, hemisphere, si_fname, file_path, grid_path, conc_spacing=1, years=[])
        conc80, rms80 = rms_si(80, ens_mem, hemisphere, si_fname, file_path, grid_path, conc_spacing=1, years=[])
        gc.collect(); print('... RMS completed')
    else:
        print('Skipping RMS analysis: 0,4%')
        conc15, conc80 = 0,4
        rms15, rms80 = None, None
    
    miz = [conc15/100, conc80/100] 
    
    #### read me
    text = []
    text.append('\nSTART: ' + str(years) + ', ' + str(months)+', '+str(hemisphere)+'-hemisphere')
    text.append('Ensemble member: '+ str(ens_mem))
    text.append('pressure: ' + str(min_pres) +' +- '+ str(pres_thresh))
    text.append('grouping = '+str(grouping_thresh)+' | ' +'sorting = '+ str(sorting_thresh))
    text.append('timing: '+str(timediff_thresh)+' hrs')
    text.append('duration: '+str(min_duration))
    text.append('ice distance: '+ str(icedist_thresh)+ ' km')
    text.append('ice lims: '+ str(ice_lims)+ ' - '+str(ice_criteria[0])+'% SIC '+str(ice_criteria[1])+' area')
    text.append(' ')
    text.append( 'RMS analysis: [' + str(conc15)+','+str(conc80)+']%'+' - '+str(rms15)+','+str(rms80) )
    text.append('New MIZ: '+str(miz))
    text.append(' ')
    text.append('out: '+ nc_save_path)
    text.append('census: '+ census_name)
    text.append('\n\n')
    text.append('*\n')
    with open(nc_save_path+'readme.txt', 'w') as f:
        for txt in text:
            print(txt)
            f.write(txt)
            f.write('\n')
        f.write('\n')

    #### year loop
    for year in years:
        print('*** '+str(year)+' ***')
        main(year, ens_mem)


#%% ---
#%% end
print(' '); print(' ')
print('------------------')
print('       done!      ')
print('------------------')
print('------------------')
print('elapsed time: ', end='')
print((timeIN.time() - TIMESTART)/60, 'minutes')

print(); print()
print(years[0],'-', years[-1])
print(ensemble_members[0],'-->', ensemble_members[-1])
print()
raise NameError('Done!')


