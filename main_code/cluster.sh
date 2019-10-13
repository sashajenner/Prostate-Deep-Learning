#!/bin/bash

password="ExMachinaGen"

#sshpass -p $password scp -r ../data/X.npy ../data/Y.npy ../data/models sjen6644@headnode.physics.usyd.edu.au:~/data
#sshpass -p $password scp -r models sjen6644@headnode.physics.usyd.edu.au:~/main_code

sshpass -p $password ssh sjen6644@headnode.physics.usyd.edu.au
