#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergyWithConc.h"
#include "selfPropelledParticleDynamics.h"
#include "gradientinteractions.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFSPVConc.h"
#include "analysisPackage.h"

#include <iostream>
#include <chrono>  // for high_resolution_clock

/*!
This file compiles to produce an executable that can be used to reproduce the timing information
in the main cellGPU paper. It sets up a simulation that takes control of a voronoi model and a simple
model of active motility
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
// Record start time
auto start = std::chrono::high_resolution_clock::now();

    //...some default parameters
    int numpts = 500.0; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 1e5; //number of time steps to run after initialization 1e6
    int initSteps = 200; //number of initialization steps
    int Nclust = 7; //number of cells in the cluster

    Dscalar KA = 500.0;
    Dscalar dt = 0.001; //the time step size
    Dscalar p0 = 3.7;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 1.0;  // the self-propulsion
    Dscalar gamma = 1.0;
    Dscalar Lx=10; // The length of the box in the x-direction
    Dscalar Ly=numpts/Lx; // The length of box in the y-direction
    Dscalar deltac = 2; // the magnitude of the effect of the gradient on the interaction
    Dscalar degradTau = 0; // Set the time constant for degradation of biochemical. If 0 it will not degrade
    int CILswitch = 0; // Set if the gradient will couple to CIL or HIT. CIL = 1 and HIT = 0
    Dscalar cellTau = 2.0; //Set the persistence time of cells for the CIL updater
    
    int current_run = 0; // the current run. useful for saving many runs of save params.

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:w:h:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 's': gamma = atof(optarg); break;
            case 'w': Nclust = atof(optarg); break;
            case 'h': current_run  = atoi(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    clock_t t1,t2; //clocks for timing informatio
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
    else
        initializeGPU = false;

    //define an equation of motion object
    shared_ptr<gradientinteractions> spp = make_shared<gradientinteractions>(numpts);
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(numpts);

    //if(CILswitch!=0)
    //    {shared_ptr<gradientinteractions> spp = make_shared<gradientinteractions>(numpts);}
    //shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(numpts);
    if(CILswitch==0)
        {bd->setT(v0);}   

    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergyWithConc> spv  = make_shared<VoronoiQuadraticEnergyWithConc>(numpts,1.0,4.0,Lx,Ly,reproducible);
 
    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(a0,p0);
    spv->setCellTypeUniform(0);  
    
    //set the cell activity to have D_r = 1. and a given v_0
    spv->setv0Dr(v0,1.0);
    spv->setModuliUniform(KA,1.0);
    //This activates tension
    spv->setSurfaceTension(gamma);
    spv->setUseSurfaceTension(true);
 
    spv->setGradvariables(deltac, dt, degradTau, CILswitch, cellTau); //Initialize the gradient variables

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    if(CILswitch!=0)
        {sim->addUpdater(spp,spv);}
    if(CILswitch==0)
        {sim->addUpdater(bd,spv);} 
    //set the time step size
    sim->setIntegrationTimestep(dt); 

    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);


    //Set box to be a rectangle                                                              
    Dscalar squareInitialSize=sqrt(numpts);
    BoxPtr square=make_shared<gpubox>(squareInitialSize,squareInitialSize);
    BoxPtr rect=make_shared<gpubox>(Lx,Ly);                                                                           
    sim->setBoxAndScale(square,rect);

    char dataname[1000];
    sprintf(dataname, "./data/cluster/cluster_N_%i_Nclust_%i_Temp_%.5f_deltac_%.2f_CILswitcht_%i_cellTau_%.2f_run_%i_dt_%.5f.nc", numpts, Nclust, v0, deltac,CILswitch, cellTau, current_run,dt);
    SPVDatabaseNetCDFConc ncdat(numpts, dataname, NcFile::Replace);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    printf("Finished with initialization\n");
    cout << "current q = " << spv->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    spv->reportMeanCellForce(false);

    spv->setCluster(Nclust); //Initialize the cluster


    int intervals = 100*(0.001/dt);

    dynamicalFeatures dynFeat(spv->returnPositions(),spv->Box);
    logSpacedIntegers logInts(0,0.05);
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%intervals==0)
            {
            ncdat.WriteState(spv);
            printf("timestep %i \n",ii);    
            logInts.update();        
            };
        sim->performTimestep();
        };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;
    cout << spv->reportq() << endl;
    cout << "number of local topology updates per cell per tau = " << spv->localTopologyUpdates*(1.0/numpts)*(1.0/tSteps/dt) << endl;

auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = finish - start;
std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
