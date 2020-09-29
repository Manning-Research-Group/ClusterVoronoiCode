#!/home/aparke05/miniconda2/bin/python

import numpy as np
import os
from time import sleep

## Set the number of simulations per set of input parameters
num_runs_per_param_set = 1

## Set the other important parameters
initSteps = int(1)
ncells = int(200)
nclust_vals = (np.linspace(1, 13, 3)).round(1)
T_vals= np.logspace(-3,0,num=8,base=10)
trial_amount = np.linspace(0,29,30)
#trial_amount = np.linspace(0,1,1)
sbar = 4.0;
dels = 0.38;

## Set the executable
runfile = '/home/elawsonk/cellGPU-master/voronoi_cluster.out'

## Make the director for saving the nc files. 
## IMPORTANT: This must match the path for the nc file in the .cpp file.
os.system('mkdir ./data/cluster')


## Loop through all parameters and runs. 
## For each run, 
## 1. Define a bash file and a submit file.
## 2. Open and write to the bash file and to the submit file, given the current parameter values.
## 3. Make the files executable with "chmod".
## 4. Submit the submit file to condor.
## 5. Wait some amount of time before submitting the next simulation, using sleep(# seconds to wait).

for trial in trial_amount:
    for nclust in nclust_vals:
        for T in T_vals:
            for run in range(num_runs_per_param_set):
            
                bashfilename = 'cluster_' + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial)  + '.sh'
                newsubfilename = 'cluster_' + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial)  + '.sub'
                
                with open(bashfilename, 'w') as shfile:
                    shfile.write("""# !/bin/bash                                                                                  
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/elawsonk/cudalibs:/usr/local/cuda/lib64:/usr/local/lib64

        """ + str(runfile) + """ -n """ + str(ncells) + """ -v """ + str(T) + """ -w """ + str(nclust) + """ -h """ + str(trial) 
                         )

                os.system('chmod +x ' +  bashfilename)
            
                with open(newsubfilename, 'w') as nsubfile:
                    nsubfile.write("""executable = /usr/local/bin/singularity
        arguments  = exec /home/elawsonk/sl7.img /home/elawsonk/cellGPU-master/""" + bashfilename + """

        output = cluster_""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + """.out
        error = cluster_""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + """.err
        log = cluster_""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_'  + """.log

        priority = 1
        queue
                            """)
                
                os.system('condor_submit ' + newsubfilename)
                sleep(1.0)
