! wget "https://www.dropbox.com/s/v56dirv1z14fy4j/dalyell_2019_s2_train.zip?dl=0#" -O data.zip
! ls -la
! mkdir data
! unzip data.zip -d data/
! ls data
! ls data/mri
! ls data/mri/pt_0000/
import csv

with open('data/info.csv') as f:

  reader = csv.reader(f)

  content = list(reader)
