#define ENABLE_CUDA
#include "DatabaseNetCDFAVM.h"
/*! \file DatabaseNetCDFAVM.cpp */

/*! Base constructor implementation */
AVMDatabaseNetCDF::AVMDatabaseNetCDF(int np, string fn, NcFile::FileMode mode)
    : BaseDatabaseNetCDF(fn,mode),
      Nv(np),
      Current(0)
{
    Nc = np/2;
    switch(Mode)
        {
        case NcFile::ReadOnly:
            break;
        case NcFile::Write:
            GetDimVar();
            break;
        case NcFile::Replace:
            SetDimVar();
            break;
        case NcFile::New:
            SetDimVar();
            break;
        default:
            ;
        };
}

void AVMDatabaseNetCDF::SetDimVar()
{
    //Set the dimensions
    recDim  = File.add_dim("rec");
    NvDim   = File.add_dim("Nv",  Nv);
    ncDim   = File.add_dim("Nc",  Nc);
    nc2Dim  = File.add_dim("Nc2", 2* Nc);
    dofDim  = File.add_dim("dof",  Nv*2);
    NvnDim  = File.add_dim("Nvn", Nv*3);
    boxDim  = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);
    cvDim   = File.add_dim("cvdim",16*Nc);
    vnDim   = File.add_dim("vndim",Nc);

    //Set the variables
    posVar       = File.add_var("pos",       ncDscalar,recDim, dofDim);
    forceVar     = File.add_var("force",       ncDscalar,recDim, dofDim);
    vcneighVar   = File.add_var("VertexCellNeighbors",         ncInt,recDim, NvnDim );
    vneighVar    = File.add_var("Vneighs",         ncInt,recDim, NvnDim );
    cellTypeVar  = File.add_var("cellType",         ncDscalar,recDim, ncDim );
    directorVar  = File.add_var("director",         ncDscalar,recDim, ncDim );
    cellPosVar   = File.add_var("cellPositions",         ncDscalar,recDim, nc2Dim );
    BoxMatrixVar = File.add_var("BoxMatrix", ncDscalar,recDim, boxDim);
    meanqVar     = File.add_var("meanQ",     ncDscalar,recDim, unitDim);
    timeVar      = File.add_var("time",     ncDscalar,recDim, unitDim);
    cellVerVar   = File.add_var("cellVer",     ncDscalar,recDim, cvDim);
    cellVerNumVar   = File.add_var("cellVerNum",     ncDscalar,recDim, vnDim);
    actualp0Var     = File.add_var("p0", ncDscalar, recDim, ncDim);
    actuala0Var     = File.add_var("a0", ncDscalar, recDim, ncDim);
    actuals0Var     = File.add_var("s0", ncDscalar, recDim, ncDim);
}

void AVMDatabaseNetCDF::GetDimVar()
{
    //Get the dimensions
    recDim  = File.get_dim("rec");
    NvDim   = File.get_dim("Nv");
    ncDim   = File.get_dim("Nc");
    nc2Dim  = File.get_dim("Nc2");
    dofDim  = File.get_dim("dof");
    NvnDim  = File.get_dim("Nvn");
    boxDim  = File.get_dim("boxdim");
    unitDim = File.get_dim("unit");
    cvDim   = File.get_dim("cvdim");
    vnDim   = File.get_dim("vndim");
    //Get the variables
    posVar       = File.get_var("pos");
    forceVar     = File.get_var("force");
    vcneighVar   = File.get_var("VertexCellNeighbors");
    vneighVar    = File.get_var("Vneighs");
    cellTypeVar  = File.get_var("cellType");
    directorVar  = File.get_var("director");
    cellPosVar   = File.get_var("cellPositions");
    BoxMatrixVar = File.get_var("BoxMatrix");
    meanqVar     = File.get_var("meanQ");
    timeVar      = File.get_var("time");
    cellVerVar   = File.get_var("cellVer");
    cellVerNumVar   = File.get_var("cellVerNum");
    actualp0Var = File.get_var("p0");
    actuala0Var = File.get_var("a0");    
    actuals0Var = File.get_var("s0");
}


//////GONCA//////
////EDITED READ-STATE FUNCTION/////
void AVMDatabaseNetCDF::ReadState(STATE t, int rec, bool geometry)
    {
    GetDimVar();

    //get the current time
    timeVar-> set_cur(rec);
    timeVar->get(& t->currentTime,1,1);
    //set the box
    BoxMatrixVar-> set_cur(rec);
    std::vector<Dscalar> boxdata(4,0.0);
    BoxMatrixVar->get(&boxdata[0],1, boxDim->size());
    t->Box->setGeneral(boxdata[0],boxdata[1],boxdata[2],boxdata[3]);

    //get the positions
    posVar-> set_cur(rec);
    std::vector<Dscalar> posdata(2*Nv,0.0);
    posVar->get(&posdata[0],1, dofDim->size());

    ArrayHandle<Dscalar2> h_p(t->vertexPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        Dscalar px = posdata[(2*idx)];
        Dscalar py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };
    /////////////GONCA////////////////////
    //////////////////////////////////////
    //get the positions for cells
    cellPosVar->set_cur(rec);
    std::vector<Dscalar> cellposdata(2*Nc,0.0);
    cellPosVar->get(&cellposdata[0],1, nc2Dim->size());


    ArrayHandle<Dscalar2> h_cpos(t->cellPositions,access_location::host,access_mode::read);
    for (int idx = 0; idx < Nc; ++idx)
      {
    Dscalar px = cellposdata[(2*idx)];
    Dscalar py = cellposdata[(2*idx)+1];
    h_cpos.data[idx].x=px;
    h_cpos.data[idx].y=py;
      };
    /////////////////////////////////////////
    ////////////////////////////////////////   
    
    //set the vertex neighbors and vertex-cell neighbors
    ArrayHandle<int> h_vn(t->vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(t->vertexCellNeighbors,access_location::host,access_mode::read);
    vneighVar->set_cur(rec);
    vcneighVar->set_cur(rec);
    std::vector<Dscalar> vndat(3*Nv,0.0);
    std::vector<Dscalar> vcndat(3*Nv,0.0);
    vneighVar       ->get(&vndat[0],1,NvnDim->size());
    vcneighVar       ->get(&vcndat[0],1,NvnDim->size());
    for (int vv = 0; vv < Nv; ++vv)
        {
        for (int ii = 0 ;ii < 3; ++ii)
            {
            h_vn.data[3*vv+ii] = vndat[3*vv+ii];
            h_vcn.data[3*vv+ii] = vcndat[3*vv+ii];
            };
        };

    ////GONCA///
    //set the cell vertex number and cell vertices
    ArrayHandle<int> h_cv(t->cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(t->cellVertexNum,access_location::host,access_mode::read);  
    cellVerVar->set_cur(rec);
    cellVerNumVar->set_cur(rec);

    std::vector<int> cvDat(16*Nc,0.0);
    std::vector<int> cvNumDat(Nc,0.0);
    cellVerVar  ->get(&cvDat[0],1,cvDim->size());
    cellVerNumVar->get(&cvNumDat[0],1,vnDim->size());
    
    for (int ii = 0 ;ii < Nc; ++ii){
      h_cvn.data[ii]=cvNumDat[ii];
    }

    for(int ii=0; ii<Nc*16; ++ii){
    h_cv.data[ii]=cvDat[ii];
    }

    if (geometry)
      {
      t->computeGeometryCPU();
      };
    };

//////GONCA//////
////EDITED WRITE-STATE FUNCTION/////
void AVMDatabaseNetCDF::WriteState(STATE s, Dscalar time, int rec)
{
    Records +=1;
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> forcedat(2*Nv);
    std::vector<Dscalar> directordat(Nc);
    std::vector<int> typedat(Nc);
    std::vector<int> vndat(3*Nv);
    std::vector<int> vcndat(3*Nv);
    std::vector<Dscalar> p0dat(Nc);
    std::vector<Dscalar> a0dat(Nc);
    std::vector<Dscalar> s0dat(Nc);

    int idx = 0;

    ArrayHandle<Dscalar2> h_p(s->vertexPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(s->vertexForces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vn(s->vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(s->vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);

    std::vector<Dscalar> cellPosDat(2*Nc);
    s->getCellPositionsCPU();
    ArrayHandle<Dscalar2> h_cpos(s->cellPositions);
    for (int ii = 0; ii < Nc; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        directordat[ii] = h_cd.data[pidx];
        typedat[ii] = h_ct.data[pidx];
        cellPosDat[2*ii+0] = h_cpos.data[pidx].x;
        cellPosDat[2*ii+1] = h_cpos.data[pidx].y;
        };
    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdxVertex[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;
        Dscalar fx = h_f.data[pidx].x;
        Dscalar fy = h_f.data[pidx].y;
        forcedat[(2*idx)] = fx;
        forcedat[(2*idx)+1] = fy;
        idx +=1;
        };

    for (int ii = 0; ii < Nc; ++ii)
        {
        Dscalar currentp0 =  h_AP.data[ii].y;
        Dscalar currenta0 =  h_AP.data[ii].x;
        p0dat[ii] = currentp0;
        a0dat[ii] = currenta0;
        s0dat[ii] = currentp0/sqrt(currenta0);            
        }

    for (int vv = 0; vv < Nv; ++vv)
        {
        int vertexIndex = s->tagToIdxVertex[vv];
        for (int ii = 0 ;ii < 3; ++ii)
            {
            vndat[3*vv+ii] = s->idxToTagVertex[h_vn.data[3*vertexIndex+ii]];
            vcndat[3*vv+ii] = s->idxToTagVertex[h_vcn.data[3*vertexIndex+ii]];
            };
        };
    ArrayHandle<int> h_cv(s->cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(s->cellVertexNum,access_location::host,access_mode::read);

    int num;
    
    std::vector<int> cvDat(16*Nc);
    std::vector<int> cvNumDat(Nc);
    for(int ii=0; ii<16*Nc; ++ii){
      cvDat[ii]=-1;
    }


    for(int ii=0; ii<Nc; ++ii){
      num = h_cvn.data[ii];
      cvNumDat[ii]=num;
      for (int nn=0; nn<num; ++nn){
	cvDat[ii*(16)+nn]=h_cv.data[s->n_idx(nn,ii)];
      }
    }

    Dscalar meanq = s->reportq();

    //Write all the data
    timeVar     ->put_rec(&time,      rec);
    meanqVar    ->put_rec(&meanq,rec);
    posVar      ->put_rec(&posdat[0],     rec);
    forceVar    ->put_rec(&forcedat[0],     rec);
    vneighVar   ->put_rec(&vndat[0],      rec);
    vcneighVar  ->put_rec(&vcndat[0],      rec);
    directorVar ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);
    cellPosVar  ->put_rec(&cellPosDat[0],rec);
    cellTypeVar ->put_rec(&typedat[0],rec);
    cellVerVar  ->put_rec(&cvDat[0],rec);
    cellVerNumVar->put_rec(&cvNumDat[0],rec);
    actualp0Var     ->put_rec(&p0dat[0],      rec);
    actuala0Var     ->put_rec(&a0dat[0],      rec);
    actuals0Var     ->put_rec(&s0dat[0],      rec);

    File.sync();
}

