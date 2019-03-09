#!/bin/bash

#the HINODE archive from HAO is structured as follows
# level1: level1/2019/01/01/SP3D/20190101_000004/*.fits
# level2: level2/2019/01/01/SP3D/20190101_000004/20190101_000004.fits
# to ingest:
#   d = YYYYMMDD_SEQ (the path part after SP3D)
#   symbolic link level1/*/*/*/SP3D/$d to $DATA/$d/level1
#   symbolic link level2/*/*/*/SP3D/$d to $DATA/$d/level2

BASE=/hinode/level2
DATA=/d/hinode/data

find $BASE -name '*.fits' | while read lv2
do
  d="$(cut -d'/' -f8 <<<"$lv2")"
  lv1=`echo $lv2|sed -e 's/level2/level1/'`
  lv1=`dirname $lv1`
  lv2=`dirname $lv2`
  if test -d $lv1
  then
  if ! test -d $DATA/$d
    then
      mkdir $DATA/$d
      ln -s $lv1 $DATA/$d/level1
      ln -s $lv2 $DATA/$d/level2
      echo $d ingested
    fi
  fi
done

#for t in $BASE/*.tar
#do
#  basename $t .tar | sed -e's/^sp_//' | while read d
#  do
#    if ! test -d $DATA/$d
#    then
#      echo mkdir $DATA/$d
#      mkdir $DATA/$d/level1
#      mkdir $DATA/$d/level2
#      tar xvf $t
#      find ./hao -name '*.fits' -exec mv {} $DATA/$d/level1 \;
#      rm -rf ./hao
#      mv $BASE/$d.fits $DATA/$d/level2
#      echo $d ingested
#    fi
#  done
#done
