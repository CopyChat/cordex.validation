#!/usr/bin/env python
"""
========
ctang, code to vld RSDS in SA from GCMs RCMs OBSs to GEBA
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



#=================================================== input
DIR='/Users/ctang/Code/CORDEX_AFR_studies/cordex.validation/'

GEBA_file='rsds.GEBA.allAfrica.csv.1990-2005.csv.anomaly.csv'
    # ID StaID, lat lon altitude country year, month, value
    # 0     1    2   3    4        5      6      7      8

    # missing and low qaulity values are removed.


StaID = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,1]
lat = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,2]
lon = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,3]
altitude = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,4]
country = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,5]
Year = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,6]
Month = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,7]
OBS = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,8]
Anomaly_OBS = np.array(pd.read_csv(DIR+GEBA_file,header=None,skiprows=0))[:,9]

print Month, Year, StaID, lon,lat,country

print Month.shape # 2065 months

GRID=(\
        'SARAH2',\
        'ERAINT',\
        'SRB',\
        'CFSR',\
        # RCM evaluation:
        # 'CCLM',\
        # 'RCA4',\
        # 'HIRHAM5',\
        # 'RACMO22T',\
        # 'REMO2009',\
        # GCM evaluation:
        # 'CNRM-CM5',\
        # 'CSIRO-Mk3-6-0',\
        # 'CanESM2',\
        # 'EC-EARTH',\
        # 'GFDL-ESM2M',\
        # 'HadGEM2-ES',\
        # 'IPSL-CM5A-LR',\
        # 'IPSL-CM5A-MR',\
        # 'MIROC5',\
        # 'MPI-ESM-LR',\
        # 'NorESM1-M',\
        # GCM-RCM:
        # 'CNRM-CM5_CLMcom-CCLM4',\
        # 'CNRM-CM5_RCA4',\
        # 'CSIRO-Mk3-6-0_RCA4',\
        # 'CanESM2_RCA4',\
        # 'EC-EARTH_CLMcom-CCLM4',\
        # 'EC-EARTH_DMI-HIRHAM5_v2',\
        # 'EC-EARTH_KNMI-RACMO22T',\
        # 'EC-EARTH_MPI-CSC-REMO2009',\
        # 'EC-EARTH_RCA4',\
        # 'GFDL-ESM2M_RCA4',\
        # 'HadGEM2-ES_CLMcom-CCLM4',\
        # 'HadGEM2-ES_KNMI-RACMO22T_v2',\
        # 'HadGEM2-ES_RCA4',\
        # 'IPSL-CM5A-LR_GERICS-REMO2009',\
        # 'IPSL-CM5A-MR_RCA4',\
        # 'MIROC5_RCA4',\
        # 'MPI-ESM-LR_CLMcom-CCLM4',\
        # 'MPI-ESM-LR_MPI-CSC-REMO2009',\
        # 'MPI-ESM-LR_RCA4',\
        # 'NorESM1-M_DMI-HIRHAM5',\
        # 'NorESM1-M_RCA4',\
        )




#=================================================== validation:
def Validation(inputarray):
    print "====================================="
    print "data      R              bias    RMSE     Sigma"
    for i in range(len(inputarray)):

        inputfile='rsds.'+inputarray[i]+'.allAfrica.csv.1990-2005.csv.anomaly.csv'

        # 'rsds.SARAH2.allAfrica.csv.1990-2005.csv',\
        InputValue = np.array(pd.read_csv(DIR+inputfile,\
                header=None,skiprows=0))[:,8]
        Anomaly = np.array(pd.read_csv(DIR+inputfile,\
                header=None,skiprows=0))[:,9]



        # if(i==23):

            # for ff in range(len(InputValue)):
                # print i,InputValue[ff]

            # print inputarray[i]
            # quit()

        # correlation
        cof= "%.2f" % float(np.ma.corrcoef(InputValue,OBS)[0,1])

        # cof after removing seasonal cycle
        cof_anomaly = "%.2f" % float(np.ma.corrcoef(Anomaly,Anomaly_OBS)[0,1])


        # Bias
        bias= "%.2f" % np.array([(InputValue[l]-OBS[l]) for l in range(len(OBS))]).mean()

        # Absolute Mean Bias
        MAB="%.2f" % np.array([np.abs(InputValue[l]-OBS[l]) for l in range(len(InputValue))]).mean()

        # RMSE
        error=InputValue-OBS
        square=error**2
        mean=np.mean(square,dtype=np.float64)
        RMSE = "%.2f" %  np.sqrt(mean)

        # Sigma
        Sigma = np.std(InputValue,dtype=np.float64)/np.std(OBS,dtype=np.float64)
        Sigma = "%.2f" % Sigma

        print inputarray[i],cof,"("+cof_anomaly+")",bias,RMSE,Sigma, MAB


    print "====================================="

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


def BiasDistribution(inputarray):

    # def of intervals:
    Intervals=np.array([0,-5,-10,-15,-20,-25,-30,-35])
    print type(Intervals)
    # Intervals=[0,-5,-10,-15,-20,-25,-30,-35]

    # create a 2D array for the biases in (Model x Interval) dim.
    InterBias=np.zeros((len(GRID),len(Intervals)-1))

    No_sta_Interval=np.array([9,7,8,5,4,1])# from counting 

    for i in range(len(inputarray)):

        inputfile='rsds.'+inputarray[i]+'.allAfrica.csv.1990-2005.csv.anomaly.csv'

        InputValue = np.array(pd.read_csv(DIR+inputfile,\
                header=None,skiprows=0))[:,8]


        # no. of values in each interval
        NO_value=[]

        for x in range(len(Intervals)-1):
            InterValue = InputValue[(lat < Intervals[x]) &\
                    (lat >= Intervals[x+1])]

            InterValue_OBS = OBS[(lat < Intervals[x]) &\
                    (lat >= Intervals[x+1])]

            NO_value.append(InterValue.shape)

            # print InterValue.shape, InterValue_OBS.shape

            # Bias
            InterBias[i,x]="%.2f" % np.array([np.abs(InterValue[l]-InterValue_OBS[l]) for l in range(len(InterValue))]).mean()

            # print("---------------"+str(x))

    print InterBias.shape
    print InterBias
    print NO_value


    #=================================================== 
    # to plot this InterBias
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
    fig.subplots_adjust(left=0.1,bottom=0.1,right=0.9,\
        hspace=0.15,top=0.93,wspace=0.43)

    # for more colors
    cm = plt.get_cmap('gist_rainbow')

    for m in range(len(GRID)):
        # nan in lat: 5-10,so mask it:
        Mask = np.isfinite(InterBias[m])

        InterBias_plot = InterBias[m][Mask]

        Intervals_plot = Intervals[Mask]

        # for more colors
        ax.set_color_cycle([cm(1.*m/len(GRID)) \
                for i in range(len(GRID))])
        ax.plot(\
                [(jk-2.5)*(-1) for jk in Intervals_plot],\
                InterBias_plot,\
                marker='o',markersize=6,label=GRID[m])

    rects1 = ax.bar(\
                [(jk-2.5)*(-1) for jk in Intervals_plot],\
                No_sta_Interval, 2, color='gray',label='No. of station')
    autolabel(rects1,ax)
    # ax.legend((rects1[0]), loc=4)
    # # plt.gca().add_artist(legend1)
    legend1 = ax.legend(loc='upper right',shadow=False ,prop={'size':12})


    title='monthly RSDS Mean Absolute Bias at GEBA stations (1990-2005)'
    fig.suptitle(title,fontsize=12)

    ax.set_xlabel('latitude', fontsize=14)
    ax.set_ylabel('Mean Absolute Bias ($\mathregular{W/m^{2}}$)', fontsize=14)
    # Unit=( '($\mathregular{W/m^{2}}$)','($\mathregular{W/m^{2}}$)')
    ax.set_xlim([0, 41])
    plt.xticks(Intervals[0:8]*(-1),Intervals[0:8])

    plt.show()


Validation(GRID)

BiasDistribution(GRID)

#===================================================  end 
print "done"
quit()
