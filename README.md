# Concentration gradient inside cellGPU

This work adds a dynamically updating biochemical signaling gradient to the cellGPU code
which was created and mantained by Daniel Sussman.The original open-source code is found 
at https://github.com/sussmanLab/cellGPU. The paper describing this code in more detail 
can currently be found on the arXiv (https://arxiv.org/abs/1702.02939), or in print 
(http://www.sciencedirect.com/science/article/pii/S0010465517301832). While this addition
uses the cellGPU code as a base the new additions are not currently able to be used on a GPU.

The signal gradient is a scalar field superimposed on top of the Voronoi model. This field
is divded into a grid which evolves according the advection-diffusion equation using 
the central finite difference method. At each time step, the cells will calculate a signal 
strength by taking the average concentration of each gridpoint within their cell walls. Then
this can be coupled to any of the cell mechanical properties. The paper describing the gradient
in more details can be found on arXiv at (arxiv link once we have it). 

## Additions and alterations to the cellGPU code
The primary additions to the code are the following:

Main: voronoi_cluster.cpp
Model: voronoiQuadraticEnergyWithConc.cpp
       voronoiQuadraticEnergyWithConc.h
Updater:  gradientinteractions.cpp
       gradientinteractions.h
Database: DatabaseNetCDFSPVConc.cpp
       DatabaseNetCDFSPVConc.h
                
With minor changes to the following:
Simple2DActiveCell.cpp
Simple2DCell.cpp
voronoiModelBase.cpp
Simple2DCell.h
Simple2DModel.h
voronoiModelBase.h


