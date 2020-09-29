#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "vertexQuadraticEnergy.h"
#include "selfPropelledCellVertexDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
/*!
This file compiles to produce an executable that can be used to reproduce the timing information
for the 2D AVM model found in the "cellGPU" paper, using the following parameters:
i = 1000
t = 4000
e = 0.01
dr = 1.0,
along with a range of v0 and p0. This program also demonstrates the use of brownian dynamics
applied to the vertices themselves.
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
vertex model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    int numpts = 200; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 1e5; //number of time steps to run after initialization
    int gaptime = 50; //When the gap will form
    int initSteps = 1e3; //number of initialization steps

    Dscalar dt = 0.001;
    Dscalar p0 = 3.7;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.01;
    Dscalar Dr = 1.0;
    Dscalar gamma = 0.05;
    int program_switch = 0; //various settings control output

    int c;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
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

    //possibly save output in netCDF format
    char dataname[256];
    sprintf(dataname,"./gaptest.nc");
    int Nvert = 2*numpts;
    AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);

    bool runSPV = true;//setting this to true will relax the random cell positions to something more uniform before running vertex model dynamics

    //We will define two potential equations of motion, and choose which later on.
    //define an equation of motion object...here for self-propelled cells
    EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
    //the next lines declare a potential brownian dynamics scheme at some targe temperature
    shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
    bd->setT(v0);
    //define a vertex model configuration with a quadratic energy functional
    shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,1.0,4.0,reproducible,runSPV);
    //set the cell preferences to uniformly have A_0 = 1, P_0 = p_0
    avm->setCellPreferencesUniform(1.0,p0);
    //set the cell activity to have D_r = 1. and a given v_0
    avm->setv0Dr(v0,1.0);
    //when an edge gets less than this long, perform a simple T1 transition
    avm->setT1Threshold(0.04);

    vector<int> vi(3*Nvert);
    vector<int> vf(3*Nvert);
    {
    ArrayHandle<int> vcn(avm->vertexCellNeighbors);
    for (int ii = 0; ii < 3*Nvert; ++ii)
        vi[ii]=vcn.data[ii];
    };

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(avm);
    sim->addUpdater(bd,avm);
    //one could have written "sim->addUpdater(spp,avm);" to use the active cell dynamics instead

    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
//    sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    sim->setReproducible(reproducible);

    avm->reportMeanVertexForce();

            ncdat.WriteState(avm);
    //perform some initial time steps. If program_switch < 0, save periodically to a netCDF database
    for (int timestep = 0; timestep < initSteps+1; ++timestep)
        {
        sim->performTimestep();
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            //ncdat.WriteState(avm);
            };
        };
    avm->reportMeanVertexForce();
            ncdat.WriteState(avm);
    
    //run for additional timesteps, and record timing information. Save frames to a database if desired
    cudaProfilerStart();
    t1=clock();
    for (int timestep = 0; timestep < gaptime; ++timestep)
        {
            ncdat.WriteState(avm);
        	sim->performTimestep();
        if(timestep%10==0)
            {
            cout << timestep << endl;
            };
        if(program_switch <0 && timestep%((int)(100/dt))==0)
            {
            cout << timestep << endl;
            if(timestep%100==0){
            ncdat.WriteState(avm);}
            };
        // if(timestep==gaptime)
        //     {
        //         cout << "Gap Time!!" << endl;
        //         avm->gapform(50);
        //         char dataname2[256];
        //         sprintf(dataname2,"./test%i.nc",fileidx);
        //         fileidx +=1;
                
        //     };

        };
    {
    int fileidx=2;
    cout << "Gap Time!!" << endl;
    for(int n=0; n<Nvert; ++n){cout << n << endl; avm->gapform(n);}
    //avm->gapform(58);
	//avm->gapform(62);
    //for(int n=0; n<Nvert; ++n){avm->gapform(n);}
    //avm->gapform(50);
    //avm->gapform(100);


    char dataname2[256];
    sprintf(dataname2,"./test%i.nc",fileidx);
    fileidx +=1;
    cout << avm->getNumberOfDegreesOfFreedom() << endl;
    AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
    for( int timestep = gaptime; timestep<tSteps; ++timestep)
        {                
            if(timestep%1000==0){
                ncdat2.WriteState(avm);
                cout << timestep << endl;
            }    
            sim->performTimestep();

        };


    // for( int timestep = gaptime; timestep<tSteps; ++timestep)
    // 	{  
    // 		fileidx = timestep-gaptime;
    // 		avm->gapform(fileidx);
    //         cout << timestep << endl;
    //         char dataname2[256];
    //         sprintf(dataname2,"../test%i.nc",fileidx);
    //         AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
    //         ncdat2.WriteState(avm);
    //         sim->performTimestep();
    //     }    




    }
    cudaProfilerStop();

    t2=clock();
    cout << "timestep time per iteration currently at " <<  (t2-t1)/(Dscalar)CLOCKS_PER_SEC/tSteps << endl << endl;
    avm->reportMeanVertexForce();
    cout << "Mean q = " << avm->reportq() << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
    };
