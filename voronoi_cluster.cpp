#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergyWithConc.h"
#include "selfPropelledParticleDynamics.h"
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
    int numpts = 200.0; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 1e6; //number of time steps to run after initialization 1e6
    int initSteps = 1000; //number of initialization steps
    int Nclust = 1; //number of cells in the cluster

    Dscalar KA = 100.0;
    Dscalar dt = 0.001; //the time step size
    Dscalar p0 = 4.00;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 0.0;  // the self-propulsion
    Dscalar gamma = 1.0;
    Dscalar sbar = 4.0;
    Dscalar dels = 0.38;

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

    clock_t t1,t2; //clocks for timing information
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

    //define an equation of motion object...here for self-propelled cells
    //EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(numpts);
    bd->setT(v0);   

    //shared_ptr<voronoiQuadraticEnergy> spv  = make_shared<voronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);
    shared_ptr<VoronoiQuadraticEnergyWithConc> spv  = make_shared<VoronoiQuadraticEnergyWithConc>(numpts,1.0,4.0,reproducible);
    //shared_ptr<VoronoiQuadraticEnergyWithTension> spv = make_shared<VoronoiQuadraticEnergyWithTension>(numpts,1.0,4.0,reproducible);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(1.0,p0);
    spv->setCellTypeUniform(0);  
    
    //set the cell activity to have D_r = 1. and a given v_0
    //spv->setv0Dr(v0,1.0);

    //"setSurfaceTension" with a single Dscalar just declares a single tension value to apply whenever cells of different type are in contact
    spv->setSurfaceTension(gamma);
    spv->setUseSurfaceTension(true);
    spv->setModuliUniform(KA,1.0);
    //in contrast, setSurfaceTension with a vector allows an entire matrix of type-type interactions to be specified
    //vector<Dscalar> gams(numpts*numpts,gamma);
    //spv->setSurfaceTension(gams);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(bd,spv);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

  
    char dataname[1000];
    sprintf(dataname, "./data/cluster/vortest_cluster_gamma_solid_N_%i_Nclust_%i_Temp_%.5f_sbar_%.2f_dels_%.2f_run_%i.nc", numpts, Nclust, v0, sbar, dels, current_run);
    //sprintf(dataname,"./vortest_200_cluster_9.nc");
    SPVDatabaseNetCDFConc ncdat(numpts, dataname, NcFile::Replace);
    // ncdat.WriteState(spv);

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
    spv->setCluster(Nclust);
    
   dynamicalFeatures dynFeat(spv->returnPositions(),spv->Box);
   logSpacedIntegers logInts(0,0.05);
    t1=clock();
    for(int ii = 0; ii < tSteps; ++ii)
        {

        if(ii%100==0)
        //if(ii == logInts.nextSave)
            {
            ncdat.WriteState(spv);
            //printf("timestep %i\t\t energy %f \t msd %f \t overlap %f topoUpdates %i \n",ii,spv->computeEnergy(),dynFeat.computeMSD(spv->returnPositions()),dynFeat.computeOverlapFunction(spv->returnPositions()),spv->localTopologyUpdates);
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

