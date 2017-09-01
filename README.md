# validation.CM_SAF.SA

## cmsaf.rsds.csv :

    * ID is the line number of file: FLAG.mon.GEBA all the 11o stations in GEBA
    over SA.
    * this ID is the same as rsds.mon.GEBA.csv and flag.mon.GEBA.csv

## flag(/rsds).mon.GEBA.csv:
    
    * 44 stations with at least one good records in 1983 - 2005
    * the same as RSDS.mon.GEBA

## GEBA.station.gt.1mon.SA.csv:

    * generate by mon.flg.sta.SA.py in $OBS/GEBAdata/validation.cm_saf/


## validation.mon.sh:

    * titan file to select lonlatbox ( 1 point ) from CORDEX output

## cmsaf.validation.station.py:

    * plot the GEBA vs CM_SAF scatter plot
    * output: validation.CM_SAF.GEBA.eps

## cmsaf.vld.sta.series.py:

    * plot the GEBA vs CM_SAF monthly series
    * output: validation.sta.series.CM_SAF.GEBA.eps

## WRDC.series.flg.8305.png

    * plot of WRDC series, all stations

## GEBA.series.flg.8305.png

    * plot of GEBA series, all stations

