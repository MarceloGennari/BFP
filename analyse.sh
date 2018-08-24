#!/bin/bash

m=0
x=0
while [ $m -le 10 ]
do
	x=1
	while [ $x -le 8 ]
	do
		python gentest.py -d -x $x -m $m -e 1 -b 1 &> tmp.txt
		python analyse.py -m $m -x $x
		((x++))
	done
	((m++))
done
