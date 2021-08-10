# !/bin/bash                                                                                  
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/elawsonk/cudalibs:/usr/local/cuda/lib64:/usr/local/lib64

            /home/elawsonk/cellGPU-master/voronoi_cluster.out -n 500 -v 1 -w 4 -h 36.0 -e 0.001 -t 100000.0 -i 1000.0