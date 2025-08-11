# arctic-cyclones-cesm2le
Software files for Mundi et al., 2025: python scripts used for data analysis and plotting  
Have Impacts of Intense Arctic Cyclones on Summer Sea Ice Reached a Maximum? (Geophysical Research Letters)  
Contact: Claire Mundi, mundi@wisc.edu

CONTAINS:

./environment.yml
+ python environment

./data/
+ example data structure for cesm data files

./era5_data/
+ output from ERA5 census of storms
* ./era5_data/areas = storm area fractions (miz vs pack ice)
* ./era5_data/cyclone_tracker_out_may = areas, census for may storms
* ./era5_data/decades = areas, census for 1990s and 2000s
* ./era5_data/original_census = storm lists for 1980s, 2010s
* ./era5_data/out_n_era3 = storm census broke down by filters (see Fig. 1)
* ./era5_data/seaice = storm sea ice time series for 1980s,2010s

./cesm_census/
+ storm lists (JJAS) for each ensemble member used in this study, filters
+ ./cesm_census/may2 = may storm lists, broken down by filters

./processed_data/
+ intermediate saved data
* ./processed_data/figures = saved data for figures.py
* ./processed_data/miz = saved data for miz_storms.py
* ./processed_data/spatial = saved data for spatial.py
* ./processed_data/supplemental_methods = saved data for supplemental_methods.py

./cesm_functions.py  
./cyctrack_functions.py
+ python files containing functions used in data analysis

./rms_si_fxn.py
+ code for calculating root-mean-square equivalent areas

./cesm_cyclone_tracker.py
+ script for identifying cyclones, full descritpion in Mundi & Lecuyer (2025), https://doi.org/10.1175/JCLI-D-24-0026.1
+ output: csv files listing cyclone info, SLP contours to define storm area, storm area fractions (miz vs pack)

./storm_tseries.py
+ calculate 3-week time series of sea ice, wind/sst
+ output: netcdf files containing time series for each storm

./figures.py
+ code for plotting Figs 1,2 from paper; supplemental figures

./miz_storms.py
+ code for plotting Figure 3 from paper

./spatial_changes.py
+ code for plotting Figure 4 from paper, Fig S3 (supplemental material)

./supplemental_methods_figure.py
+ code for plotting Figure S1 in Supplemental Material
 
