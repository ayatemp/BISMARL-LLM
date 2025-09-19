#!/usr/bi/env bash

# put in your doc2json directory here
DIR=$PWD

# # Download Grobid
cd $DIR
wget https://github.com/kermitt2/grobid/archive/0.7.3.zip
unzip 0.7.3.zip
rm 0.7.3.zip
cd $DIR/grobid-0.7.3
./gradlew clean install




