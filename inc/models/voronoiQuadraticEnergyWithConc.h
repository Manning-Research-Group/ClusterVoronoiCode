#ifndef VoronoiQuadraticEnergyWithConc_H
#define VoronoiQuadraticEnergyWithConc_H

#include "voronoiQuadraticEnergy.h"

/*! \file voronoiQuadraticEnergyWithConc.h */
/*!
A child class of VoronoiQuadraticEnergy, this implements an Voronoi model in 2D that can include concentration gradients
 */

class VoronoiQuadraticEnergyWithConc : public VoronoiQuadraticEnergy
    {
    public:
        //!initialize with random positions in a square box
        //voronoiQuadraticEnergyWithConc(int n,bool reprod = false) : VoronoiQuadraticEnergy(n,reprod){};
        VoronoiQuadraticEnergyWithConc(int n,bool reprod = false);
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        //voronoiQuadraticEnergyWithConc(int n, Dscalar A0, Dscalar P0,bool reprod = false) : VoronoiQuadraticEnergy(n,A0,P0,reprod){};
        VoronoiQuadraticEnergyWithConc(int n, Dscalar A0, Dscalar P0,bool reprod = false);
        //!compute the geometry and get the forces
        virtual void computeForces();

        virtual void InPolygon(int cell, int concgrid, int currentnum, int checkup, int checkdown, int checkleft, int checkright, int uneven, Dscalar ybias, Dscalar xbias);
        virtual double VertAngle(double p1x, double p1y, double p2x, double p2y);

        virtual Dscalar computeEnergy();
        //!Compute force sets on the GPU
        virtual void ComputeForceSetsGPU();

        virtual void setCluster(int nclust);

        //!Compute the net force on particle i on the CPU with only a single tension value
        virtual void computeVoronoiSimpleTensionForceCPU(int i);

        //!call gpu_force_sets kernel caller
        virtual void computeVoronoiSimpleTensionForceSetsGPU();
        //!Compute the net force on particle i on the CPU with multiple tension values
        virtual void computeVoronoiTensionForceCPU(int i);
        //!call gpu_force_sets kernel caller
        virtual void computeVoronoiTensionForceSetsGPU();

        //!Use surface tension
        void setUseSurfaceTension(bool use_tension){Tension = use_tension;};
        //!Set surface tension, with only a single value of surface tension
        void setSurfaceTension(Dscalar g){gamma = g; simpleTension = true;};
        //!Set a general flattened 2d matrix describing surface tensions between many cell types
        void setSurfaceTension(vector<Dscalar> gammas);
        //!Get surface tension
        Dscalar getSurfaceTension(){return gamma;};

        GPUArray<Dscalar> concentration;
        GPUArray<Dscalar> linetension;
        GPUArray<Dscalar2> absvert;
        GPUArray<Dscalar> conccell;
        GPUArray<Dscalar> celladjust;
        int totalsteps;
        int Numcells;
        GPUArray<Dscalar2> conclocation;
        GPUArray<Dscalar2> cellcon;
        GPUArray<Dscalar2> pastcellpos;
        GPUArray<Dscalar2> cellvelocity;
        GPUArray<Dscalar2> celltenforces;


    protected:
        //!The value of surface tension between two cells of different type (some day make this more general)
        Dscalar gamma;
        //!A flag specifying whether the force calculation contains any surface tensions to compute
        bool Tension;
        //!A flag switching between "simple" tensions (only a single value of gamma for every unlike interaction) or not
        bool simpleTension;
        //!A flattened 2d matrix describing the surface tension, \gamma_{i,j} for types i and j
        GPUArray<Dscalar> tensionMatrix;


    //be friends with the associated Database class so it can access data to store or read
    friend class SPVDatabaseNetCDF;
    friend class SPVDatabaseNetCDFConc;
    };

#endif
