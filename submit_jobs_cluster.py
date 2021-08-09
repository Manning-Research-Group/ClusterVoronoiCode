#!/home/aparke05/miniconda2/bin/python

import numpy as np
import os
from time import sleep

## Set the number of simulations per set of input parameters
num_runs_per_param_set = 1

## Set the other important parameters
#initSteps = int(1)
ncells = int(500)
#nclust_vals = (np.linspace(1, 13, 3)).round(1)
#nclust_vals = [100,200,500,1000,5000]
#nclust = 7
#nclust_vals = [15]
#nclust_vals = [10,15,20]
nclust_vals = [1,20]
#nclust_vals =[30,35,40,45,50]
#nclust_vals = [1,5,10,15,20,25,30,35,40,45,50]
#T_vals= np.logspace(-3,0,num=8,base=10)
#T_vals= [0.01,0.05,5]
T_vals = [1]
#T_vals= [0.1,0.5,1,2,10]
#T_vals = [0.03, 0.075, 0.15, 0.3, 0.75]
#T_vals = [1.05, 1.35, 1.65, 1.95]
#trial_amount = np.linspace(0,29,30)
#trial_amount = [1]
#trial_amount = np.linspace(65,79,15)
trial_amount = np.linspace(0,4,5)
sbar = 1;
dels = 3;
loc = 1;
#dt_amount = [0.0005]
dt_amount=[0.001];

#Init
#str(5000*0.01/time) 
#-t """ + str(2e5*0.001/time)
#-t """ + str(5*1903*1.5*np.sqrt(ncells))
#-t """ + str(5e5)
#+ """ -i """ + str(1000*0.001/time)

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

for time in dt_amount:
    for trial in trial_amount:
        for nclust in nclust_vals:
            for T in T_vals:
                for run in range(num_runs_per_param_set):
                
                    bashfilename = 'cluster_advect' + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + str(loc) + '_' + str(time)  + '.sh'
                    newsubfilename = 'cluster_advect' + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + str(loc) + '_' + str(time)  + '.sub'
                    
                    with open(bashfilename, 'w') as shfile:
                        shfile.write("""# !/bin/bash                                                                                  
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/elawsonk/cudalibs:/usr/local/cuda/lib64:/usr/local/lib64

            """ + str(runfile) + """ -n """ + str(ncells) + """ -v """ + str(T) + """ -w """ + str(nclust) + """ -h """ + str(trial) + """ -e """ + str(time) + """ -t """ + str(5e4*0.001/time) + """ -i """ + str(100*0.001/time) 
                             )

                    os.system('chmod +x ' +  bashfilename)
                
                    with open(newsubfilename, 'w') as nsubfile:
                        nsubfile.write("""executable = /usr/local/bin/singularity
            arguments  = exec /home/elawsonk/sl7.img /home/elawsonk/cellGPU-master/""" + bashfilename + """

            output = cluster_advect""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + str(loc) + '_' + str(time) + """.out
            error = cluster_advect""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_' + str(loc) + '_' + str(time) +  """.err
            log = cluster_advect""" + str(ncells) + '_' + str(nclust) + '_'  + str(T) + '_' + str(sbar) + '_' + str(dels) + '_'  + str(trial) + '_'  + str(loc) + '_' + str(time) +  """.log

            priority = 1
            queue
                                """)
                    
                    os.system('condor_submit ' + newsubfilename)
                    sleep(1.0)
