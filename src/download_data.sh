#!/bin/sh 
mkdir -p ../data/raw_data/wikipedia/
wget -c http://www.derczynski.com/resources/nordicdsl/dk.txt -O ../data/raw_data/wikipedia/dk.txt
wget -c http://www.derczynski.com/resources/nordicdsl/nb.txt -O ../data/raw_data/wikipedia/nb.txt