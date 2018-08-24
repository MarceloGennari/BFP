#!/bin/bash

m=0
x=1
while [ $x -le 8 ]
do
	m=0
	while [ $m -le 12 ]
	do
		python gentest.py -m $m -x $x -b 2500 -e 10 >> /mnt/d/results.txt
		((m++))
	done
	((x++))
	echo "Latest update" | mutt -a "/mnt/d/results.txt" -s "Results of the thing" -- marcelo.gennaridonascimento@wadham.ox.ac.uk
done
