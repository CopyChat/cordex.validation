#!/bin/bash
#
#          FILE: merge.GEBA.WRDC.BSRN.sh
# 
USAGE="./merge.GEBA.WRDC.BSRN.sh"
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 04/03/17 14:37
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Shell/functions.sh   # ctang's functions


#=================================================== copy data

#---  FUNCTION  ----------------------------------------------------------------
#          NAME:  copyData
#   DESCRIPTION:  copy data
#    PARAMETERS:  
#       RETURNS:  
#-------------------------------------------------------------------------------

function copyData()
{

WRDC=/Users/ctang/climate/GLOBALDATA/OBSDATA/WRDC
# 1,angola,dundo,-7.4,20.8167,775,1964,0,0,0,0,0,0,0,0,0,0,0,0,

GEBA=/Users/ctang/climate/GLOBALDATA/OBSDATA/GEBAdata/validation.cm_saf
# 148,1,1983,0,0,0,0,0,0,0,0,0,0,0,0,0,Luanda/Angola
# change '/' to ',' in vim

cp $WRDC/Southern.Africa/flag.MonMean ./flag.mon.WRDC
awk -F "," '{ gsub(/\(/,""); gsub(/\)/,""); print $0}' \
    $GEBA/flag.rsds.monthly.1983-2005.csv > ./flag.mon.GEBA

awk -F "," '{ gsub(/\(/,""); gsub(/\)/,""); print $0}' \
    $GEBA/station.south.africa > ./GEBA.station 

}
#copyData
#=================================================== 

awk -F ',' '$7 >= 1983 && $7 <= 2005' flag.mon.WRDC > flag.mon.WRDC.csv

rm flag.mon.WRDC


#---  FUNCTION  ----------------------------------------------------------------
#          NAME:  arrange
#   DESCRIPTION:  formate: index,country,station,lat,lot,year,mon1,..,mon12
#    PARAMETERS:  
#       RETURNS:  
#-------------------------------------------------------------------------------


function arrange()
{
    awk -F "," '{print "WRDC",$1,$2,$3,$4,$5,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17i,$18,$19}' flag.mon.WRDC.csv > flag.mon.WRDC.reform.csv

    rm flag.mon.WRDC.csv
    
    Tline=$(wc -l flag.mon.GEBA | awk '{print $1}')
    echo $Tline

    rm flag.mon.GEBA.reform.csv
    for i in $(seq -s " " 1 $Tline)
    do
        country=$(awk -F "," 'NR=='$i'{print $17}' flag.mon.GEBA) 

        lat=$(awk -v val=$country -F ',' '{if($4 ~ val) print $2}' GEBA.station)
        lon=$(awk -v val=$country -F ',' '{if($4 ~ val) print $3}' GEBA.station)
        
        echo $i,$country,$lat,$lon

        echo -n "GEBA" $i $country $lat $lon" "  >> flag.mon.GEBA.reform.csv
        awk -F "," 'NR=='$i'{print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15}' flag.mon.GEBA >> flag.mon.GEBA.reform.csv
    done

awk -F "," '{ gsub(/\//," "); print $0}' \
    flag.mon.GEBA.reform.csv > jj
mv jj flag.mon.GEBA.reform.csv

    rm GEBA.station flag.mon.GEBA.reform flag.mon.GEBA.csv

}

#arrange


#---  FUNCTION  ----------------------------------------------------------------
#          NAME:  merge
#   DESCRIPTION:  merge GEBA and WRDC
#    PARAMETERS:  
#       RETURNS:  GEBA.WRDC.csv
#-------------------------------------------------------------------------------
function merge()
{
    cp flag.mon.GEBA.reform.csv GEBA.WRDC.csv
    cat flag.mon.WRDC.reform.csv >> GEBA.WRDC.csv

    # sort by lat
    sort -n -k 5 GEBA.WRDC.csv > GEBA.WRDC.lat.csv
    sort -n -k 5 flag.mon.WRDC.reform.csv > flag.mon.WRDC.lat.csv
    sort -n -k 5 flag.mon.GEBA.reform.csv > flag.mon.GEBA.lat.csv
}

merge

