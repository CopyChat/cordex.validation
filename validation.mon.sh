#!/bin/bash - 
#===============================================================================
#
#          FILE: validation.mon.sh
# 
#         USAGE: ./validation.mon.sh 
# 
#   DESCRIPTION: to prepare the validation against GEBA monthly data 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Tang (Tang), tangchao90908@sina.com
#  ORGANIZATION: KLA
#       CREATED: 03/19/17 11:59:37 RET
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

#station='rsds.southern.africa.mon.1970-1999.gt.5year'
station='rsds.southern.africa.gt.5year.1970-1999.csv'

line=$(wc -l $station | awk '{print $1}')
for i in $(seq 1 $line)
do
	sta=$(awk 'NR=='$i'{print $1}' $station)
	lat1=$(awk 'NR=='$i'{print $2-0.22}' $station)
	lat2=$(awk 'NR=='$i'{print $2+0.22}' $station)
	lon1=$(awk 'NR=='$i'{print $3-0.22}' $station)
	lon2=$(awk 'NR=='$i'{print $3+0.22}' $station)
	year=$(awk 'NR=='$i'{print $7}' $station)

	cdo2 -b 64 -s sellonlatbox,$lon1,$lon2,$lat1,$lat2 1999.nc test.nc 2>&1-
	nx=$(cdo -s sinfo test.nc | grep curvilinear | awk '{print $12}' )
	ny=$(cdo -s sinfo test.nc | grep curvilinear | awk '{print $15}' )
	echo ----- $i --station=$sta, year=$year ----test: box: $lon1,$lon2,$lat1,$lat1,nx=$nx,ny=$ny

	for f in $(ls *monmean.nc)
	do
		echo year=$year, station=$sta
		cdo2 -b 64 -s sellonlatbox,$lon1,$lon2,$lat1,$lat2 $f ${f%.nc}.$i.$sta.nc
		cdo2 -b 64 -s selyear,$(eval echo $year) ${f%.nc}.$i.$sta.nc ${f%.nc}.$i.$sta.$year.nc
	done
done
