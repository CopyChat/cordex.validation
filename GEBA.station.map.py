#!/usr/bin/env python
"""
========
ctang, a map of geba stations in southern africa
========
"""
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
from textwrap import wrap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap , addcyclic
import sys 
sys.path.append('/Users/ctang/Code/Python/')
import ctang



DIR='/Users/ctang/Code/CORDEX_AFR_studies/cordex.validation/'

 
#=================================================== titles
title2='NO. of monthly SSR in 33 GEBA stations during 1990-2005'

#=================================================== plot
# map empty
vmin=0
vmax=200
#print np.max(station_all.year.max()) #25

cmap = plt.cm.YlOrRd
cmaplist = [cmap(i) for i in range(cmap.N)]
bounds = np.linspace(vmin,vmax,11)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#=================================================== 

STATION_sel='GEBA.allAfrica.station.1990-2005.csv'
station_sel = pd.read_csv(DIR+STATION_sel, index_col='STA',\
        names=['STA', 'lat', 'lon', 'year'])

fig2, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
fig2.subplots_adjust(left=0.1,bottom=0.1,right=0.99,hspace=0.1,top=0.8,wspace=0.43)

map=Basemap(projection='cyl',llcrnrlat=-40,urcrnrlat=1,llcrnrlon=0,urcrnrlon=60,resolution='h',area_thresh=1000)
ctang.setMap(map)

sc=plt.scatter(\
    station_sel.lon, station_sel.lat, c=station_sel.year,edgecolor='black',\
    zorder=2,norm=norm,vmin=vmin,vmax=vmax,s=85, cmap=cmap)

cb=plt.colorbar(sc,orientation='horizontal',shrink=0.5, pad = 0.05)
cb.ax.tick_params(labelsize=9) 
axes.set_title("\n".join(wrap(title2)))

plt.savefig('GEBA.station.southern.africa.1990-2005.map.png')

#===================================================  end of subplot 3
plt.show()

quit()

