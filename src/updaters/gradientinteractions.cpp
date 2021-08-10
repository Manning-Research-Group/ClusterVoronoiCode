#define ENABLE_CUDA

#include "gradientinteractions.h"
#include "selfPropelledParticleDynamics.cuh"
/*! \file gradientinteractions.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
gradientinteractions::gradientinteractions(int _N)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = true;
    mu = 1.0;
    Ndof = _N;
    noise.initialize(Ndof);
    displacements.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void gradientinteractions::spatialSorting()
    {
    //reIndexing = activeModel->returnItt();
    //reIndexRNG(noise.RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable; cast it as an active cell model
*/
void gradientinteractions::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    activeModel = dynamic_pointer_cast<Simple2DActiveCell>(model);
    }

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void gradientinteractions::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    }

/*!
The straightforward CPU implementation
*/
void gradientinteractions::integrateEquationsOfMotionCPU()
    {
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(activeModel->cellDirectors);
    ArrayHandle<Dscalar2> h_v(activeModel->cellVelocities);
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_motility(activeModel->Motility,access_location::host,access_mode::read);

    //Get the alignment data from the model
    activeModel->getalginment(alignArr);
    ArrayHandle<Dscalar2> h_gradalign(alignArr);

    //Get the gradient variables from the model
    activeModel->getGradvariables(gradArr);
    ArrayHandle<Dscalar> gradientvar(gradArr);

    Dscalar2 alignsum;
    alignsum.x=0;
    alignsum.y=0;
    Dscalar delT = gradientvar.data[1];

    for (int ii = 0; ii < Ndof; ++ii)
        {
        Dscalar2 Vcur = h_v.data[ii]; 
        Dscalar Tau = gradientvar.data[4];
        Dscalar2 currentAlign = h_gradalign.data[ii];

        //displace according to current velocities and forces
        h_disp.data[ii].x = delT*(Vcur.x + mu * h_f.data[ii].x);
        h_disp.data[ii].y = delT*(Vcur.y + mu * h_f.data[ii].y);

        Dscalar theta = h_cd.data[ii];
        //Calculate the polarity angle
        if (Vcur.x != 0. && Vcur.y != 0.)
            {
            theta = atan2(Vcur.y,Vcur.x);
            };
        h_cd.data[ii] =theta;

        Dscalar v0i = h_motility.data[ii].x;
        v0i = v0i*sqrt(1/(delT));
        Dscalar Dri = h_motility.data[ii].y;
        
        //Add Noise
        Dscalar noiseX =sqrt(2.0)*noise.getRealNormal(); 
        Dscalar noiseY =sqrt(2.0)*noise.getRealNormal();
        
        //External cells have a persistence time of 1
        if(currentAlign.x==0){
            Tau=1;
        }

        //Evolve the polarity of the cells
        h_v.data[ii].x =  Vcur.x+delT*(-Vcur.x/Tau+v0i*noiseX+mu*currentAlign.x);
        h_v.data[ii].y =  Vcur.y+delT*(-Vcur.y/Tau+v0i*noiseY+mu*currentAlign.y);  

        };

    }//end array handle scoping
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    //vector of displacements is mu*forces*timestep + v0's*timestep
    };

/*!
The straightforward GPU implementation
*/
void gradientinteractions::integrateEquationsOfMotionGPU()
    {
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);

    gpu_spp_eom_integration(d_f.data,
                 d_v.data,
                 d_disp.data,
                 d_motility.data,
                 d_cd.data,
                 d_RNG.data,
                 Ndof,
                 deltaT,
                 Timestep,
                 mu);
    };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    };
