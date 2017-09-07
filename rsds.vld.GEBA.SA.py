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
        'ERA_Interim',\
        'CCLM',\
        'RCA4',\
        'HIRHAM5',\
        'RACMO22T',\
        'REMO2009',\
        # 'SRB',\
        # 'NCEP-CFSR',\
        )

RCM=(\
        'CCCma-CanESM2_SMHI-RCA4_v1',\
        'CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_v1',\
        'CSIRO-QCCCE-CSIRO-Mk3-6-0_SMHI-RCA4_v1',\
        'ICHEC-EC-EARTH_SMHI-RCA4_v1',\
        'IPSL-IPSL-CM5A-MR_SMHI-RCA4_v1',\
        'MIROC-MIROC5_SMHI-RCA4_v1',\
        'MOHC-HadGEM2-ES_SMHI-RCA4_v1',\
        'MPI-M-MPI-ESM-LR_SMHI-RCA4_v1',\
        'NCC-NorESM1-M_SMHI-RCA4_v1',\
        'NOAA-GFDL-GFDL-ESM2M_SMHI-RCA4_v1',\
        \
        'CNRM-CERFACS-CNRM-CM5_CLMcom-CCLM4-8-17_v1',\
        'ICHEC-EC-EARTH_CLMcom-CCLM4-8-17_v1',\
        'MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17_v1',\
        'MPI-M-MPI-ESM-LR_CLMcom-CCLM4-8-17_v1',\
        'ICHEC-EC-EARTH_DMI-HIRHAM5_v2',\
        'NCC-NorESM1-M_DMI-HIRHAM5_v1',\
        'ICHEC-EC-EARTH_KNMI-RACMO22T_v1',\
        'MOHC-HadGEM2-ES_KNMI-RACMO22T_v2',\
        'ICHEC-EC-EARTH_MPI-CSC-REMO2009_v1',\
        'IPSL-IPSL-CM5A-LR_GERICS-REMO2009_v1',\
        'MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009_v1',\
        )

GCM=(\
    'rsds_Amon_CNRM-CM5_historical-rcp85_r1i1p1',\
    'rsds_Amon_CSIRO-Mk3-6-0_historical-rcp85_r1i1p1',\
    'rsds_Amon_CanESM2_historical-rcp85_r1i1p1',\
    'rsds_Amon_EC-EARTH_historical-rcp85_r1i1p1',\
    'rsds_Amon_GFDL-ESM2M_historical-rcp85_r1i1p1',\
    'rsds_Amon_HadGEM2-ES_historical-rcp85_r1i1p1',\
    'rsds_Amon_IPSL-CM5A-LR_historical-rcp85_r1i1p1',\
    'rsds_Amon_IPSL-CM5A-MR_historical-rcp85_r1i1p1',\
    'rsds_Amon_MIROC5_historical-rcp85_r1i1p1',\
    'rsds_Amon_MPI-ESM-LR_historical-rcp85_r1i1p1',\
    'rsds_Amon_NorESM1-M_historical-rcp85_r1i1p1',\
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

        # correlation
        cof= "%.2f" % float(np.ma.corrcoef(InputValue,OBS)[0,1])

        # cof after removing seasonal cycle
        cof_anomaly = "%.2f" % float(np.ma.corrcoef(Anomaly,Anomaly_OBS)[0,1])


        # Bias
        bias= "%.2f" % np.array([np.abs(InputValue[l]-OBS[l]) for l in range(len(OBS))]).mean()

        # RMSE
        error=InputValue-OBS
        square=error**2
        mean=np.mean(square,dtype=np.float64)
        RMSE = "%.2f" %  np.sqrt(mean)

        # Sigma
        Sigma = np.std(InputValue,dtype=np.float64)/np.std(OBS,dtype=np.float64)
        Sigma = "%.2f" % Sigma

        print inputarray[i],cof,"("+cof_anomaly+")",bias,RMSE,Sigma

    print "====================================="




def BiasDistribution(inputarray):

    # def of intervals:
    Intervals=[0,-5,-10,-15,-20,-25,-30,-35]

    # create a 2D array for the biases in (Model x Interval) dim.
    InterBias=np.zeros((len(GRID),len(Intervals)-1))

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

    print InterBias.shape
    print InterBias


    # to plot this InterBias
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,9),\
        facecolor='w', edgecolor='k') # figsize=(w,h)
    # fig.subplots_adjust(left=0.3,bottom=0.1,right=0.9,\
        # hspace=0.15,top=0.9,wspace=0.43)

    for m in range(len(GRID)):
        ax.plot([jk+2.5 for jk in Intervals[0:7]],InterBias[m][::-1],marker='o',markersize=6,label=GRID[m])

    legend = ax.legend(loc='upper right',shadow=False ,prop={'size':12})

    
    title='monthly RSDS Mean Absolute Bias at GEBA stations'
    fig.suptitle(title,fontsize=12)

    ax.set_xlabel('latitude', fontsize=14)
    ax.set_ylabel('Mean Absolute Bias (W/m2)', fontsize=14)
    ax.set_xticks([ 1,2,3,4,5,6,7,8])
    plt.xticks(Intervals[0:7],Intervals[0:7][::-1])
    # print (Intervals[0:8].reverse())

    plt.show()


# Validation(GRID)

BiasDistribution(GRID)

#===================================================  end 
print "done"
quit()
