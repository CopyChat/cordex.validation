#!/usr/bin/env python
"""
========
ctang, a map of geba stations in southern africa
========
"""
import math
import pdb
from scipy import stats
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
from textwrap import wrap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap , addcyclic
from matplotlib.dates import YearLocator,MonthLocator,DateFormatter,drange
import sys 
sys.path.append('/Users/ctang/Code/Python/')
import ctang

DIR='/Users/ctang/climate/GLOBALDATA/OBSDATA/CM_SAF/validation.CM_SAF.SA/'
#=================================================== plot

STATION_idfile = 'cmsaf.vld.sta.mab.csv'
name=[ 'Sta_NO','MAB','Sta_NO','sta_ID','lat','lon','N_month','Sta_NO','sta_ID,lat','lon','N_month','sta_ID','altitude']
#Sta_NO,MAB,Sta_NO,sta_ID,lat,lon,N_month,Sta_NO,sta_ID,lat,lon,N_month,sta_ID,altitude
# 0    ,1,  2,    3,    , 4, , 5,   6,      7,     8,   9,  10, 11,     12,    13



N_month = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,6]
lats = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,4]
lons = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,5]
station_id = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,5]
alt = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,13]
mab = np.array(pd.read_csv(DIR+STATION_idfile,header=None,skiprows=1))[:,1]

#=================================================== 
alt_mab = np.array(np.column_stack((alt,mab,N_month)))
alt_mab=alt_mab[np.argsort(alt_mab[:,0])]

alt_mab = alt_mab[alt_mab[:,2] > 10]
alt_mab = alt_mab[alt_mab[:,1] < 50]# rm 2 cites in Madagascar

alt=alt_mab[:,0]
mab1=alt_mab[:,1]

#=================================================== k

lat_mab = np.array(np.column_stack((lats,mab,N_month)))
lat_mab=lat_mab[np.argsort(lat_mab[:,0])]

lat_mab = lat_mab[lat_mab[:,2] > 10]
lat_mab = lat_mab[lat_mab[:,1] < 50] # rm 2 cites in Madagascar

lat=lat_mab[:,0]
mab2=lat_mab[:,1]

#=================================================== plotting
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
#fig.subplots_adjust(left=0.04,bottom=0.15,right=0.98,\
        # hspace=0.15,top=0.8,wspace=0.43)

plt.sca(axes) # active shis subplot 
axx=axes

axx.plot(alt,mab1,linestyle='--',marker='s',markersize=2,lw=1,zorder=2,color='blue')

axx.set_xlabel("station elevation (m)")
axx.set_ylabel("MAB (W/m2)")

title1='Mean Absolute Bias and station altitude in southern Africa 1983-2005'
fig.suptitle(title1,fontsize=12)

# plot record num of each station
# for sta in range(len(station_id)):
    # if N_month[sta] > 5 and mab[sta] > 15:
        # plt.annotate( format(mab[sta],'.1f'),xy=(lons[sta], lats[sta]), xytext=(4, 1),\
            # textcoords='offset points', ha='left', va='top',color='blue',\
            # # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),\
            # # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')\
            # )

plt.savefig('cmsaf.vld.atl.mab.eps',format='eps')
#===================================================  end of subplot 3
fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
#fig.subplots_adjust(left=0.04,bottom=0.15,right=0.98,\
        # hspace=0.15,top=0.8,wspace=0.43)

#=================================================== plot 2
plt.sca(axes) # active shis subplot 
axx=axes

# linear plot:
axx.plot(lat,mab2,linestyle='-',marker='o',markersize=10,lw=2,zorder=2,color='blue')

# linear fitting
slope, intercept, r_value, p_value, std_err = stats.linregress(lat,mab2)
print slope,intercept,r_value,p_value,std_err

yy=[t*slope+intercept for t in range(-40,10,1)]
axx.plot(range(-40,10,1),yy,'--',color='red',zorder=10,label='fitting')
legend = axx.legend(loc='upper left', shadow=False,prop={'size':8})

if p_value < 0.01:
    axx.text( -30,35,'r='+str(format(r_value,'.2f'))+'(p<0.01)',ha='left', fontsize=8, rotation=0)   
else:
    axx.text( -30,35,'r='+str(format(r_value,'.2f'))+'(p='+str(format(p_value,'.2f'))+')',ha='left', fontsize=8, rotation=0)   

# hist plot:
bin=np.linspace(-40, 0, 10)

# bin=
axx.set_xlabel("lat")
axx.set_ylabel("MAB (W/m2)")

title2='Mean Absolute Bias and station latitude in southern Africa 1983-2005'
fig1.suptitle(title2,fontsize=12)
#=================================================== end of plot 2
# plot record num of each station
# for sta in range(len(station_id)):
    # if N_month[sta] > 5 and mab[sta] > 15:
        # plt.annotate( format(mab[sta],'.1f'),xy=(lons[sta], lats[sta]), xytext=(4, 1),\
            # textcoords='offset points', ha='left', va='top',color='blue',\
            # # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),\
            # # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')\
            # )

plt.savefig('cmsaf.vld.lat.mab.eps',format='eps')
print "done"
plt.show()

quit()

