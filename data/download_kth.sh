#!/bin/bash
mkdir ./kth
cd ./kth

wget http://www.nada.kth.se/cvap/actions/walking.zip
wget http://www.nada.kth.se/cvap/actions/jogging.zip
wget http://www.nada.kth.se/cvap/actions/running.zip
wget http://www.nada.kth.se/cvap/actions/boxing.zip
wget http://www.nada.kth.se/cvap/actions/handwaving.zip
wget http://www.nada.kth.se/cvap/actions/handclapping.zip

unzip walking.zip -d walking
unzip jogging.zip -d jogging
unzip running.zip -d running
unzip boxing.zip -d boxing
unzip handwaving.zip -d handwaving
unzip handclapping.zip -d handclapping

find . -name '*.zip' -exec rm {} \;
