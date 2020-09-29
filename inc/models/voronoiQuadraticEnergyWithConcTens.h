#ifndef VoronoiQuadraticEnergyWithConcTens_H
#define VoronoiQuadraticEnergyWithConcTens_H

#include "voronoiQuadraticEnergy.h"

/*! \file voronoiQuadraticEnergyWithConc.h */
/*!
A child class of VoronoiQuadraticEnergy, this implements an Voronoi model in 2D that can include concentration gradients
 */

class VoronoiQuadraticEnergyWithConcTens : public VoronoiQuadraticEnergy
    {
    public:
        //!initialize with random positions in a square box
        //voronoiQuadraticEnergyWithConc(int n,bool reprod = false) : VoronoiQuadraticEnergy(n,reprod){};
        VoronoiQuadraticEnergyWithConcTens(int n,bool reprod = false);
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        //voronoiQuadraticEnergyWithConcTens(int n, Dscalar A0, Dscalar P0,bool reprod = false) : VoronoiQuadraticEnergy(n,A0,P0,reprod){};
        VoronoiQuadraticEnergyWithConcTens(int n, Dscalar A0, Dscalar P0,bool reprod = false);
        //!compute the geometry and get the forces
        virtual void computeForces();

        virtual void InPolygon(int cell, int concgrid, int currentnum, int checkup, int checkdown, int checkleft, int checkright);
        virtual double VertAngle(double p1x, double p1y, double p2x, double p2y);


        GPUArray<Dscalar> concentration;
        GPUArray<Dscalar2> absvert;
        GPUArray<Dscalar> conccell;
        GPUArray<Dscalar> celladjust;
        int totalsteps;
        int Numcells;
        //int eggsalad;
        GPUArray<Dscalar2> conclocation;
        GPUArray<Dscalar2> cellcon;
        GPUArray<Dscalar2> pastcellpos;
        GPUArray<Dscalar2> cellvelocity;


    protected:



    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    friend class SPVDatabaseNetCDFConc;
    };

#endif
