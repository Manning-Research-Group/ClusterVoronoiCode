#ifndef DATABASE_SPVC_H
#define DATABASE_SPVC_H

#include "voronoiQuadraticEnergyWithConc.h"
#include "voronoiQuadraticEnergyWithConcTens.h"
#include "DatabaseNetCDF.h"

/*! \file DatabaseNetCDFSPV.h */
//!Simple databse for reading/writing 2d spv states
/*!
Class for a state database for a 2d delaunay triangulation
the box dimensions are stored, the 2d unwrapped coordinate of the delaunay vertices,
and the shape index parameter for each vertex
*/
class SPVDatabaseNetCDFConc : public BaseDatabaseNetCDF
{
private:
    typedef shared_ptr<VoronoiQuadraticEnergyWithConc> STATE;
    int Nv; //!< number of vertices in delaunay triangulation
    NcDim *recDim, *NvDim, *dofDim, *boxDim, *unitDim, *concDim, *vertDim, *tensionDim; //!< NcDims we'll use
    NcVar *posVar, *typeVar, *directorVar, *BoxMatrixVar, *timeVar, *means0Var,*exVar, *concVar, *vertVar, *locVar, *actuals0Var, *cellvelVar, *cellconVar, *calcCellVelVar, *tensionVar, *forceVar, *tenforceVar, *alignVar; //!<NcVars we'll use
    bool exclusions; //!< Keep track of whether particle exlucsions should be saved
    int Current;    //!< keeps track of the current record when in write mode


public:
    SPVDatabaseNetCDFConc(int np, string fn="temp.nc", NcFile::FileMode mode=NcFile::ReadOnly,bool excluded = false);
    ~SPVDatabaseNetCDFConc(){File.close();};

protected:
    void SetDimVar();
    void GetDimVar();

public:
    int  GetCurrentRec(); //!<Return the current record of the database
    //!Get the total number of records in the database
    int GetNumRecs(){
                    NcDim *rd = File.get_dim("rec");
                    return rd->size();
                    };

    //!Write the current state of the system to the database. If the default value of "rec=-1" is used, just append the current state to a new record at the end of the database
    virtual void WriteState(STATE c, Dscalar time = -1.0, int rec=-1);
    //!Read the "rec"th entry of the database into SPV2D state c. If geometry=true, after reading a CPU-based triangulation is performed, and local geometry of cells computed.
    virtual void ReadState(STATE c, int rec,bool geometry=true);

};
#endif
