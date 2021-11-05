# Concentration gradient inside cellGPU

This work adds a dynamically updating biochemical signaling gradient to the cellGPU code which was created and maintained by Daniel Sussman. The original open-source code is found at https://github.com/sussmanLab/cellGPU. The paper describing this code in more detail can currently be found on the arXiv (https://arxiv.org/abs/1702.02939), or in print (http://www.sciencedirect.com/science/article/pii/S0010465517301832). While this addition uses the cellGPU code as a base the new additions are not currently able to be used on a GPU.

The signal gradient is a scalar field superimposed on top of the Voronoi model. This field is divided into a grid which evolves according to the advection-diffusion equation using the central finite difference method. At each time step, the cells will calculate a signal strength by taking the average concentration of each gridpoint within their cell walls. Then this can be coupled to any of the cell mechanical properties. The paper describing the gradient in more detail can be found on arXiv at (arxiv link once we have it). 

## Additions and alterations to the cellGPU code
The primary additions to the code are the following:

Main: voronoi_cluster.cpp  <br />
Model: voronoiQuadraticEnergyWithConc.cpp / voronoiQuadraticEnergyWithConc.h  <br />
Updater:  gradientinteractions.cpp / gradientinteractions.h  <br />
Database: DatabaseNetCDFSPVConc.cpp / DatabaseNetCDFSPVConc.h
                
<br />                
With minor changes to the following:  <br />
Simple2DModel.h  <br />
Simple2DActiveCell.cpp  <br />
Simple2DCell.cpp /  Simple2DCell.h  <br /> 
voronoiModelBase.cpp / voronoiModelBase.h  <br />


 


