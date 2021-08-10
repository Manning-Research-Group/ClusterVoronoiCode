#define ENABLE_CUDA
#include "DatabaseNetCDFSPVConc.h"
/*! \file DatabaseNetCDFSPVConc.cpp */

SPVDatabaseNetCDFConc::SPVDatabaseNetCDFConc(int np, string fn, NcFile::FileMode mode, bool exclude)
    : BaseDatabaseNetCDF(fn,mode),
      Nv(np),
      Current(0),
      exclusions(exclude)
    {
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

void SPVDatabaseNetCDFConc::SetDimVar()
    {
    int gs = ceil(sqrt(Nv*10))*ceil(sqrt(Nv*10));
    int gsxy = ceil(0.75*sqrt(Nv*10))*ceil(1.5*sqrt(Nv*10));
    //Set the dimensions
    recDim = File.add_dim("rec");
    NvDim  = File.add_dim("Nv",  Nv);
    dofDim = File.add_dim("dof", Nv*2);
    boxDim = File.add_dim("boxdim",4);
    unitDim = File.add_dim("unit",1);
    concDim = File.add_dim("concentrationdim", gsxy*2);
    vertDim = File.add_dim("vertdim", Nv*16);
    tensionDim = File.add_dim("tensionDim", Nv*60);

    //Set the variables
    timeVar          = File.add_var("time",     ncDscalar,recDim, unitDim);
    means0Var          = File.add_var("means0",     ncDscalar,recDim, unitDim);
    posVar          = File.add_var("pos",       ncDscalar,recDim, dofDim);
    typeVar          = File.add_var("type",         ncInt,recDim, NvDim );
    directorVar          = File.add_var("director",         ncDscalar,recDim, NvDim );
    BoxMatrixVar    = File.add_var("BoxMatrix", ncDscalar,recDim, boxDim);
    concVar    = File.add_var("Concentration", ncDscalar,recDim, concDim);
    //vertVar    = File.add_var("Vertices", ncDscalar, recDim, vertDim);
    //locVar     = File.add_var("Locations", ncDscalar, recDim, concDim);
    cellconVar      = File.add_var("Cell Concentration", ncDscalar, recDim, NvDim);
    actuals0Var     = File.add_var("s0", ncDscalar, recDim, NvDim);
    cellvelVar      = File.add_var("Cell Velocity", ncDscalar, recDim, dofDim);
    //calcCellVelVar      = File.add_var("Calculated Cell Velocity", ncDscalar, recDim, dofDim);
    //tensionVar      = File.add_var("Line Tension", ncDscalar, recDim, tensionDim);
    forceVar          = File.add_var("Net Force",       ncDscalar,recDim, dofDim);
    tenforceVar          = File.add_var("Tension Force",       ncDscalar,recDim, dofDim);
    alignVar          = File.add_var("Alignment",       ncDscalar,recDim, dofDim);

    if(exclusions)
        exVar          = File.add_var("externalForce",       ncDscalar,recDim, dofDim);
    }

void SPVDatabaseNetCDFConc::GetDimVar()
    {
    //Get the dimensions
    recDim = File.get_dim("rec");
    boxDim = File.get_dim("boxdim");
    NvDim  = File.get_dim("Nv");
    dofDim = File.get_dim("dof");
    unitDim = File.get_dim("unit");
    concDim= File.get_dim("concentrationdim");
    //Get the variables
    posVar          = File.get_var("pos");
    typeVar          = File.get_var("type");
    directorVar          = File.get_var("director");
    means0Var          = File.get_var("means0");
    BoxMatrixVar    = File.get_var("BoxMatrix");
    timeVar    = File.get_var("time");
    concVar = File.get_var("Concentration");
    //vertVar = File.get_var("Vertices");
    cellconVar = File.get_var("Cell Concentration");
    actuals0Var = File.get_var("s0");
    cellvelVar  = File.get_var("Cell Velocity");
    //calcCellVelVar  = File.get_var("Calculated Cell Velocity");
    //tensionVar  = File.get_var("Line Tension");
    forceVar          = File.get_var("Net Force");
    tenforceVar          = File.get_var("Tension Force");
    alignVar          = File.get_var("Alignment");

    if(exclusions)
        exVar = File.get_var("externalForce");
    }

void SPVDatabaseNetCDFConc::WriteState(STATE s, Dscalar time, int rec)
    {
    if(rec<0)   rec = recDim->size();
    if (time < 0) time = s->currentTime;

    std::vector<Dscalar> boxdat(4,0.0);
    Dscalar x11,x12,x21,x22;
    s->Box->getBoxDims(x11,x12,x21,x22);
    boxdat[0]=x11;
    boxdat[1]=x12;
    boxdat[2]=x21;
    boxdat[3]=x22;

    int gs = ceil(sqrt(Nv*10))*ceil(sqrt(Nv*10));
    int gsxy = ceil(0.75*sqrt(Nv*10))*ceil(1.5*sqrt(Nv*10));
    std::vector<Dscalar> posdat(2*Nv);
    std::vector<Dscalar> directordat(Nv);
    std::vector<int> typedat(Nv);
    std::vector<Dscalar> concdat(gsxy*2);
    //std::vector<Dscalar> clocdat(gsxy*2);
    //std::vector<Dscalar> vertdat(Nv*16);
    std::vector<Dscalar> adjustdat(Nv);
    std::vector<Dscalar> s0dat(Nv);
    std::vector<Dscalar> veldat(2*Nv);
    //std::vector<Dscalar> calcveldat(2*Nv);
    //std::vector<Dscalar> tensiondat(Nv*60);
    std::vector<Dscalar> forcedat(2*Nv);
    std::vector<Dscalar> tenforcedat(2*Nv);
    std::vector<Dscalar> aligndat(2*Nv);
    int idx = 0;
    Dscalar means0=0.0;

    ArrayHandle<Dscalar2> h_p(s->cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(s->cellDirectors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_conc(s->concentration,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(s->cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_ex(s->exclusions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> vertpos(s->voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> avert(s->absvert,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(s->cellNeighborNum,access_location::host,access_mode::read);
    //ArrayHandle<Dscalar2> cloc(s->conclocation,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_adjust(s->celladjust,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(s->AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(s->cellVelocities,access_location::host,access_mode::read);
    //ArrayHandle<Dscalar2> h_velocity(s->cellvelocity,access_location::host,access_mode::read);
    //ArrayHandle<Dscalar> h_tension(s->linetension,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_celltenforces(s->celltenforces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(s->cellForces,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_align(s->gradalign,access_location::host,access_mode::read);

    for (int ii = 0; ii < Nv; ++ii)
        {
        int pidx = s->tagToIdx[ii];
        Dscalar px = h_p.data[pidx].x;
        Dscalar py = h_p.data[pidx].y;
        Dscalar vx = h_v.data[pidx].x;
        Dscalar vy = h_v.data[pidx].y;
        //Dscalar cvx = h_velocity.data[pidx].x;
        //Dscalar cvy = h_velocity.data[pidx].y;
        Dscalar fx = h_f.data[pidx].x;
        Dscalar fy = h_f.data[pidx].y;
        Dscalar fxten = h_celltenforces.data[pidx].x;
        Dscalar fyten = h_celltenforces.data[pidx].y;
        Dscalar alignx = h_align.data[pidx].x;
        Dscalar aligny = h_align.data[pidx].y;

        posdat[(2*idx)] = px;
        posdat[(2*idx)+1] = py;

        forcedat[(2*idx)] = fx;
        forcedat[(2*idx)+1] = fy;
        tenforcedat[(2*idx)] = fxten;
        tenforcedat[(2*idx)+1] = fyten;
        aligndat[(2*idx)] = alignx;
        aligndat[(2*idx)+1] = aligny;

        veldat[(2*idx)] = vx;
        veldat[(2*idx)+1] = vy;
        //calcveldat[(2*idx)] = cvx;
        //calcveldat[(2*idx)+1] = cvy;

        directordat[ii] = h_cd.data[pidx];
        if(h_ex.data[ii] == 0)
            typedat[ii] = h_ct.data[pidx];
        else
            typedat[ii] = h_ct.data[pidx]-5;

        //Current Concentration inside cell
        Dscalar cellconcentration = h_adjust.data[ii];
        adjustdat[ii]=cellconcentration;
        Dscalar currents0 =  h_AP.data[ii].y;
        s0dat[ii] = currents0;

        idx +=1;
        };
//    means0 = means0/Nv;
    means0 = s->reportq();
    int counts=0;
    for (int ii = 0; ii < gsxy; ++ii)
        {
        Dscalar c =h_conc.data[ii];
        concdat[ii]=c;
    //     Dscalar c2 =cloc.data[ii].x;
    //     Dscalar c3 =cloc.data[ii].y;
    //     clocdat[2*ii]=c2;
    //     clocdat[2*ii+1]=c3;
        }

    int sums=0;
    for(int ii=0; ii<Nv;ii++){
        sums+= h_nn.data[ii];
    }

    // for(int ii=0; ii<sums*2;ii++){
    //     if(ii%2==0)
    //         {vertdat[ii] = avert.data[ii/2].x;}
    //     else
    //         {vertdat[ii] = avert.data[ii/2].y;}
        //vertdat[i]= d;
    //}

   // for(int ii=0; ii<Nv*60;ii++){
    //    tensiondat[ii] = h_tension.data[ii]; 
    //}
    

    //cout << sizeof(avert) << endl;


    //Write all the data
    means0Var      ->put_rec(&means0,      rec);
    timeVar      ->put_rec(&time,      rec);
    posVar      ->put_rec(&posdat[0],     rec);
    typeVar       ->put_rec(&typedat[0],      rec);
    directorVar       ->put_rec(&directordat[0],      rec);
    BoxMatrixVar->put_rec(&boxdat[0],     rec);
    concVar     ->put_rec(&concdat[0],      rec);
    //vertVar     ->put_rec(&vertdat[0],     rec);
    //locVar      ->put_rec(&clocdat[0],      rec);
    cellconVar      ->put_rec(&adjustdat[0],      rec);
    actuals0Var     ->put_rec(&s0dat[0],      rec);
    cellvelVar      ->put_rec(&veldat[0],       rec);
    //calcCellVelVar      ->put_rec(&calcveldat[0],       rec);
    //tensionVar      ->put_rec(&tensiondat[0],       rec);
    forceVar      ->put_rec(&forcedat[0],     rec);
    tenforceVar      ->put_rec(&tenforcedat[0],     rec);
    alignVar      ->put_rec(&aligndat[0],     rec);

    if(exclusions)
        {
        ArrayHandle<Dscalar2> h_ef(s->external_forces,access_location::host,access_mode::read);
        std::vector<Dscalar> exdat(2*Nv);
        int id = 0;
        for (int ii = 0; ii < Nv; ++ii)
            {
            int pidx = s->tagToIdx[ii];
            Dscalar px = h_ef.data[pidx].x;
            Dscalar py = h_ef.data[pidx].y;
            exdat[(2*id)] = px;
            exdat[(2*id)+1] = py;
            id +=1;
            };
        exVar      ->put_rec(&exdat[0],     rec);
        };

    File.sync();
    }

void SPVDatabaseNetCDFConc::ReadState(STATE t, int rec,bool geometry)
    {
    //initialize the NetCDF dimensions and variables
    //test if there is exclusion data to read...
    int tester = File.num_vars();
    if (tester == 7)
        exclusions = true;
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

    ArrayHandle<Dscalar2> h_p(t->cellPositions,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        Dscalar px = posdata[(2*idx)];
        Dscalar py = posdata[(2*idx)+1];
        h_p.data[idx].x=px;
        h_p.data[idx].y=py;
        };

    //get cell types and cell directors
    typeVar->set_cur(rec);
    std::vector<int> ctdata(Nv,0.0);
    typeVar->get(&ctdata[0],1, NvDim->size());
    ArrayHandle<int> h_ct(t->cellType,access_location::host,access_mode::overwrite);

    directorVar->set_cur(rec);
    std::vector<Dscalar> cddata(Nv,0.0);
    directorVar->get(&cddata[0],1, NvDim->size());
    ArrayHandle<Dscalar> h_cd(t->cellDirectors,access_location::host,access_mode::overwrite);
    for (int idx = 0; idx < Nv; ++idx)
        {
        h_cd.data[idx]=cddata[idx];;
        h_ct.data[idx]=ctdata[idx];;
        };

    //read in excluded forces if applicable...
    if (tester == 7)
        {
        exVar-> set_cur(rec);
        std::vector<Dscalar> efdata(2*Nv,0.0);
        exVar->get(&posdata[0],1, dofDim->size());
        ArrayHandle<Dscalar2> h_ef(t->external_forces,access_location::host,access_mode::overwrite);
        for (int idx = 0; idx < Nv; ++idx)
            {
            Dscalar efx = efdata[(2*idx)];
            Dscalar efy = efdata[(2*idx)+1];
            h_ef.data[idx].x=efx;
            h_ef.data[idx].y=efy;
            };
        };


    //by default, compute the triangulation and geometrical information
    if(geometry)
        {
        t->resetDelLocPoints();
        t->updateCellList();
        t->globalTriangulationCGAL();
        t->resetLists();
        if(t->GPUcompute)
            t->computeGeometryGPU();
        else
            t->computeGeometryCPU();
        };
    }


