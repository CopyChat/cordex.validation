#!/usr/bin/env python
"""
========
ctang, a map of geba stations in southern africa
========
"""
import math
import pdb
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
# Sta_i, mab, no, sta_id, lat, lon, n_of_month
# 0    , 1  , 2,  3,    , 4, , 5,   6
lats = np.array(pd.read_csv(DIR+STATION_idfile,header=None))[:,4]
lons = np.array(pd.read_csv(DIR+STATION_idfile,header=None))[:,5]
station_id = np.array(pd.read_csv(DIR+STATION_idfile,header=None))[:,5]
mab = np.array(pd.read_csv(DIR+STATION_idfile,header=None))[:,1]
N_month = np.array(pd.read_csv(DIR+STATION_idfile,header=None))[:,6]

#=================================================== plotting
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
#fig.subplots_adjust(left=0.04,bottom=0.15,right=0.98,\
        # hspace=0.15,top=0.8,wspace=0.43)

plt.sca(axes) # active shis subplot 
axx=axes

vmin = 10
vmax = 50 

cmap = plt.cm.GnBu
cmaplist = [cmap(i) for i in range(cmap.N)]
bounds = np.linspace(vmin,vmax,9)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

map=Basemap(projection='cyl',area_thresh=10000.,llcrnrlat=-45,urcrnrlat=1,llcrnrlon=0,urcrnrlon=60,resolution='h')

map.drawcoastlines(linewidth=1)
map.drawparallels(np.arange(-90.,91.,10.),labels=[1,0,0,0],linewidth=0.5)
map.drawmeridians(np.arange(-180.,181.,20.),labels=[0,0,0,1],linewidth=0.5)
map.drawmapboundary(linewidth=1)
map.drawcountries(linewidth=1)




for sta in range(len(station_id)):
    if N_month[sta] > 12:
        cc=int(mab[sta])
        print cc,lons[sta],lats[sta]
        sc=plt.scatter(\
            # lons[sta], lats[sta], c=cc,\
            -22,22,c=cc,\
            zorder=2,norm=norm,vmin=vmin,vmax=vmax,s=55, cmap=cmap)

cb=plt.colorbar(sc,orientation='horizontal',shrink=0.6)
cb.ax.tick_params(labelsize=9) 
cb.ax.set_title("Mean Absolute Bias (W/m2)")



title='Mean Absolute Bias of CM SAF vs GEBA monthly SSR in southern Africa 1983-2005'
fig.suptitle(title,fontsize=12)

# plot record num of each station
# for sta in range(len(station_id)):
    # if N_month[sta] > 5 and mab[sta] > 15:
        # plt.annotate( format(mab[sta],'.1f'),xy=(lons[sta], lats[sta]), xytext=(4, 1),\
            # textcoords='offset points', ha='left', va='top',color='blue',\
            # # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),\
            # # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')\
            # )
plt.savefig('mon.flg.sta.SA.8305.eps',format='eps')
#===================================================  end of subplot 3
print "done"
plt.show()
quit()
