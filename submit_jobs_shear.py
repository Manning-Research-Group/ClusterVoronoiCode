#!/home/aparke05/miniconda2/bin/python

import numpy as np
import os
from time import sleep

## Set the number of simulations per set of input parameters
num_runs_per_param_set = 1

## Set the series of p0 values to run
p0_vals = (np.linspace(3.5, 4.1, 14)).round(4)
##p0_vals = (np.linspace(0.80, 1.0, 21)).round(4)

## Set the other important parameters
initSteps = int(1)
dt = 0.0001
ncells = int(600)

## Set the executable
runfile = '/home/elawsonk/cellGPU-master/dynamicalMatrix.out'

## Make the director for saving the nc files. 
## IMPORTANT: This must match the path for the nc file in the .cpp file.
os.system('mkdir ./data/gapFIRE_set01')


## Loop through all parameters and runs. 
## For each run, 
## 1. Define a bash file and a submit file.
## 2. Open and write to the bash file and to the submit file, given the current parameter values.
## 3. Make the files executable with "chmod".
## 4. Submit the submit file to condor.
## 5. Wait some amount of time before submitting the next simulation, using sleep(# seconds to wait).

for p in p0_vals:
    for run in range(num_runs_per_param_set):
    
        bashfilename = 'sheargap_' + str(ncells) + '_' + str(p).replace('.', 'p')  + '.sh'
        newsubfilename = 'sheargap_' + str(ncells) + '_' + str(p).replace('.', 'p')  + '.sub'
        
        with open(bashfilename, 'w') as shfile:
            shfile.write("""# !/bin/bash                                                                                  
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/elawsonk/cudalibs:/usr/local/cuda/lib64:/usr/local/lib64

""" + str(runfile) + """ -e """ + str(dt) + """ -n """ + str(ncells) + """ -p """ + str(p) 
                 )

        os.system('chmod +x ' +  bashfilename)
    
        with open(newsubfilename, 'w') as nsubfile:
            nsubfile.write("""executable = /usr/local/bin/singularity
arguments  = exec /home/elawsonk/sl7.img /home/elawsonk/cellGPU-master/""" + bashfilename + """

output = sheargap_""" + str(ncells) + '_'  + str(p) + """.out
error = sheargap_""" + str(ncells) + '_' + str(p)  + """.err
log = sheargap_""" + str(ncells) + '_' + str(p)  +""".log

priority = 1
queue
                    """)
        
        os.system('condor_submit ' + newsubfilename)
        sleep(1.0)
