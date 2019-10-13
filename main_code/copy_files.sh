#!/bin/bash

password="ExMachinaGen"

sshpass -p $password scp -r sjen6644@headnode.physics.usyd.edu.au:~/data/models ../data
sshpass -p $password scp -r sjen6644@headnode.physics.usyd.edu.au:~/main_code/models ./
