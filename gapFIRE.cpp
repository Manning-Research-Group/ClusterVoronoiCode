#include "std_include.h"

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define ENABLE_CUDA

#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "selfPropelledCellVertexDynamics.h"
#include "vertexQuadraticEnergy.h"
#include "DatabaseNetCDFSPV.h"
#include "DatabaseNetCDFAVM.h"
#include "EnergyMinimizerFIRE2D.h"

/*!
This file compiles to produce an executable that demonstrates how to use the energy minimization
functionality of cellGPU. Now that energy minimization behaves like any other equation of motion, this
demonstration is pretty straightforward
*/

//! A function of convenience for setting FIRE parameters
void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, Dscalar deltaT, Dscalar alphaStart,
        Dscalar deltaTMax, Dscalar deltaTInc, Dscalar deltaTDec, Dscalar alphaDec, int nMin,
        Dscalar forceCutoff)
    {
    emin->setDeltaT(deltaT);
    emin->setAlphaStart(alphaStart);
    emin->setDeltaTMax(deltaTMax);
    emin->setDeltaTInc(deltaTInc);
    emin->setDeltaTDec(deltaTDec);
    emin->setAlphaDec(alphaDec);
    emin->setNMin(nMin);
    emin->setForceCutoff(forceCutoff);
    };

int main(int argc, char*argv[])
{
    //as in the examples in the main directory, there are a bunch of default parameters that
    //can be changed from the command line
    int numpts = 200;
    int USE_GPU = -1;
    int c;
    int tSteps = 1;
    int initSteps = 1;

    Dscalar dt = 1e-4;
    Dscalar KA = 100.0;
    Dscalar p0 = 3.9;
    Dscalar a0 = 1.0;
    Dscalar v0 = 0.1;
    Dscalar gamma = 0.0;

    int current_run = 0; // the current run. useful for saving many runs of save params.
    //
    int program_switch = 1;
    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:k:h:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'k': KA = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
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

    clock_t t1,t2;
    bool reproducible = false;
    bool initializeGPU = true;
    if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
    else
        initializeGPU = false;

    char dataname[256];
    sprintf(dataname, "./data/gapFIRE_set01/gaptest_N_%i_ka_%.4f_dt_%.5f_a0_%.2f_p0_%.4f_run_%i.nc", numpts, KA, dt, a0, p0, current_run);
    //sprintf(dataname, "./data/gaptest_N_%i_ka_%.4f_dt_%.5f_a0_%.2f_p0_%.4f_run_%i.nc", numpts, KA, dt, a0, p0, current_run);
    //sprintf(dataname, "./data/em_200121set00OrigVertex_N_%i_ka_%.4f_dt_%.5f_a0_%.2f_p0_%.4f_t1th_%.2f/em_N_%i_ka_%.4f_dt_%.5f_a0_%.2f_p0_%.4f_t1th_%.2f_run_%i.nc", numpts, KA, dt, a0, p0, tt, numpts, KA, dt, a0, p0, tt, current_run);


    //program_switch == 1 --> vertex model
    if(program_switch == 1)
        {
        shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,a0,p0,reproducible);
        //shared_ptr<VertexQuadraticEnergyWithGaps> avm = make_shared<VertexQuadraticEnergyWithGaps>(numpts,a0,p0,reproducible);

        shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(avm);

        SimulationPtr sim = make_shared<Simulation>();
        sim->setConfiguration(avm);
        sim->addUpdater(fireMinimizer,avm);
        sim->setIntegrationTimestep(dt);
//        if(initSteps > 0)
//           sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
        sim->setCPUOperation(!initializeGPU);

        //Number of gaps between 0 and 2*numpts

        Dscalar mf;

        setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
        fireMinimizer->setMaximumIterations(1e6);
        sim->performTimestep();
        mf = fireMinimizer->getMaxForce();
        //ncdat.WriteState(avm);


        int gaps=2*numpts;
        //int gaps=1;
        for(int n=0; n<gaps; ++n){avm->gapform(n);}
        printf("End of Gap Formation");
        AVMDatabaseNetCDF ncdat(2*numpts+2*gaps,dataname,NcFile::Replace);
        ncdat.WriteState(avm);

        //for(int n=201; n<250; ++n){avm->cellDeath(n);}

        setFIREParameters(fireMinimizer,dt,0.99,0.1,1.1,0.95,.9,4,1e-12);
        fireMinimizer->setMaximumIterations(1e6);
        sim->performTimestep();
        mf = fireMinimizer->getMaxForce();


        char dataname2[256];
        sprintf(dataname2, "./data/gapFIRE_set01/gaptest_N_%i_ka_%.4f_dt_%.5f_a0_%.2f_p0_%.4f_run_%i_2.nc", numpts, KA, dt, a0, p0, current_run);
        AVMDatabaseNetCDF ncdat2(avm->getNumberOfDegreesOfFreedom(),dataname2,NcFile::Replace);
        ncdat2.WriteState(avm);


        printf("minimized value of q = %f\n",avm->reportq());
        Dscalar meanQ = avm->reportq();
        Dscalar varQ = avm->reportVarq();
        Dscalar2 variances = avm->reportVarAP();
        printf("current KA = %f\t Cell <q> = %f\t Var(p) = %g\n",KA,meanQ,variances.y);
        ncdat2.WriteState(avm);

        };
    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};
