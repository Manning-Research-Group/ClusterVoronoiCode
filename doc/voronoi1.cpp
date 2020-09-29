#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
//#include "voronoiQuadraticEnergy.h"
#include "voronoiQuadraticEnergyWithConc.h"
//#include "voronoiQuadraticEnergyWithTension.h"
#include "selfPropelledParticleDynamics.h"
#include "DatabaseNetCDFSPVConc.h"


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
    //...some default parameters
    int numpts = 200.0; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 70000; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps

    Dscalar dt = 0.01; //the time step size
    Dscalar p0 = 3.8;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 0.1;  // the self-propulsion

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:")) != -1)
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
    bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
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
    EOMPtr spp = make_shared<selfPropelledParticleDynamics>(numpts);
    //define a voronoi configuration with a quadratic energy functional
    
    //shared_ptr<voronoiQuadraticEnergy> spv  = make_shared<voronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible);
    shared_ptr<VoronoiQuadraticEnergyWithConc> spv  = make_shared<VoronoiQuadraticEnergyWithConc>(numpts,1.0,4.0,reproducible);
    //shared_ptr<VoronoiQuadraticEnergyWithTension> spv = make_shared<VoronoiQuadraticEnergyWithTension>(numpts,1.0,4.0,reproducible);

    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    spv->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    spv->setv0Dr(v0,1.0);


    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(spv);
    sim->addUpdater(spp,spv);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

  
    char dataname[1000];
    sprintf(dataname,"./vortest.nc");
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


    //Expample of passing in pointers
    //double* out = spv->computeConcentration();
    //for(int i=0; i<5; i++){cout << out[i] <<endl;}
    //spv->computeConcentration(numpts);
    //ArrayHandle<Dscalar> conc(spv->concentration,access_location::host,access_mode::read);
    //cout << conc.data[0]; 

    //run for additional timesteps, and record timing information
    int logSaveIdx = 0;
    int nextSave = 0;
    //int spacing[50] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 19, 23, 29, 36, 45, 56, 70, 87, 109, 136, 170, 212, 265, 331, 414, 517, 646, 808, 1010, 1262, 1577, 1971, 2464, 3080, 3850, 4812, 6015, 7518, 9397, 11746, 14682, 18353, 22940,28675, 35842, 44802, 56002};
    //int nextSave = spacing[logSaveIdx];
    t1=clock();
    sim->setCurrentTimestep(0);
    for(int ii = 0; ii < tSteps; ++ii)
      {
    
    if(ii == nextSave)
      {
            printf(" step %i\n",ii);
            ncdat.WriteState(spv);
            nextSave = (int)round(pow(pow(10.0,0.05),logSaveIdx));
            //nextSave = spacing[logSaveIdx];
            while(nextSave == ii)
          {
                logSaveIdx +=1;
                nextSave = (int)round(pow(pow(10.0,0.05),logSaveIdx));
            //    nextSave = spacing[logSaveIdx];
          };

      };
     
        //ncdat.WriteState(spv);
        sim->performTimestep();
      };
    t2=clock();
    Dscalar steptime = (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;

    cout << spv->reportq() << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
