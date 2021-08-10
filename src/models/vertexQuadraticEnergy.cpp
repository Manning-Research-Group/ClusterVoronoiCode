#define ENABLE_CUDA

#include "vertexQuadraticEnergy.h"
#include "vertexQuadraticEnergy.cuh"
#include "voronoiQuadraticEnergy.h"
/*! \file vertexQuadraticEnergy.cpp */

/*!
\param n number of CELLS to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\param runSPVToInitialize the default constructor has the cells start as a Voronoi tesselation of
a random point set. Set this flag to true to relax this initial configuration via the Voronoi2D class
\post Initialize(n,runSPVToInitialize) is called, setCellPreferencesUniform(A0,P0), and
setModuliUniform(1.0,1.0)
*/
VertexQuadraticEnergy::VertexQuadraticEnergy(int n,Dscalar A0, Dscalar P0,bool reprod,bool runSPVToInitialize)
    {
    printf("Initializing %i cells with random positions as an initially Delaunay configuration in a square box... \n",n);
    Reproducible = reprod;
    initializeVertexModelBase(n,runSPVToInitialize);
    setCellPreferencesUniform(A0,P0);
    };

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2
*/

/* I can't make a new file because I'm dumb So we are saving this.
Dscalar VertexQuadraticEnergy::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);
    Energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        Energy += h_m.data[nn].x * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        Energy += h_m.data[nn].y * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };

    return Energy;
    };
*/

/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2 + \sum_{[i]\neq[j]} \gamma_{[i][j]}l_{ij}
*/
Dscalar VertexQuadraticEnergy::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    //first, compute the area and perimeter pieces...which are easy
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);
    Energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        Energy += h_m.data[nn].x * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        Energy += h_m.data[nn].y * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };

    //now, the potential line tension terms
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    for (int cell = 0; cell < Ncells; ++cell)
        {
        //get the Delaunay neighbors of the cell
        int neigh = h_nn.data[cell];
        vector<int> ns(neigh);
        vector<Dscalar2> voro(neigh);
        for (int nn = 0; nn < neigh; ++nn)
            {
            ns[nn] = h_n.data[n_idx(nn,cell)];
            voro[nn] = h_v.data[n_idx(nn,cell)];
            };

        Dscalar gamma = 0.0;
        Dscalar2 vlast, vnext,vcur;
        Dscalar2 dlast, dnext;
        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vcur = voro[nn];
            vnext = voro[(nn+1)%neigh];
            int baseNeigh = ns[nn];
            int typeI = h_ct.data[cell];
            int typeK = h_ct.data[baseNeigh];
            //if the cell types are different, calculate everything once
            if (typeI != typeK && cell < baseNeigh)
                {
                dnext.x = vcur.x-vnext.x;
                dnext.y = vcur.y-vnext.y;
                Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
                Energy += dnnorm*gamma;
                };
            vlast=vcur;
            };
        };
    return Energy;
    };



/*!
compute the geometry and the forces and the vertices, on either the GPU or CPU as determined by
flags
*/
void VertexQuadraticEnergy::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    //compute the current area and perimeter of every cell
    computeGeometry();
    //use this information to compute the net force on the vertices
    if(GPUcompute)
        {
        computeForcesGPU();
        }
    else
        {
        computeForcesCPU();
        };
    };

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex
*/

/*
void VertexQuadraticEnergy::computeForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its three cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        int cellIdx = h_vcn.data[fsidx];
        Dscalar Adiff = h_m.data[cellIdx].x*(h_AP.data[cellIdx].x - h_APpref.data[cellIdx].x);
        Dscalar Pdiff = h_m.data[cellIdx].y*(h_AP.data[cellIdx].y - h_APpref.data[cellIdx].y);
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetVertexModel is defined in inc/utility/functions.h
        computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);

        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };
    };
*/

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex...for the cpu part combine the simple and complex tension routines
*/
void VertexQuadraticEnergy::computeForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<Dscalar4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_f(vertexForces,access_location::host, access_mode::overwrite);
    ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);

    //first, compute the contribution to the force on each vertex from each of its three cells
    Dscalar2 vlast,vcur,vnext;
    Dscalar2 dEdv;
    Dscalar Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        //for the change in the energy of the cell, just repeat the vertexQuadraticEnergy part
        int cellIdx1 = h_vcn.data[fsidx];
        Dscalar Adiff = h_m.data[cellIdx1].x *(h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x);
        Dscalar Pdiff = h_m.data[cellIdx1].y *(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y);
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetVertexModel is defined in inc/utility/functions.h
        computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;

        //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
        int cellNeighs = h_cvn.data[cellIdx1];
        //find the index of vcur and vnext
        int vCurIdx = fsidx/3;
        int vNextInt = 0;
        if (h_cv.data[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
            {
            for (int nn = 0; nn < cellNeighs-1; ++nn)
                {
                int idx = h_cv.data[n_idx(nn,cellIdx1)];
                if (idx == vCurIdx)
                    vNextInt = nn +1;
                };
            };
        int vNextIdx = h_cv.data[n_idx(vNextInt,cellIdx1)];

        //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
        int cellIdx2 = 0;
        int cellOfSet = fsidx-3*vCurIdx;
        for (int cc = 0; cc < 3; ++cc)
            {
            if (cellOfSet == cc) continue;
            int cell2 = h_vcn.data[3*vCurIdx+cc];
            int cNeighs = h_cvn.data[cell2];
            for (int nn = 0; nn < cNeighs; ++nn)
                if (h_cv.data[n_idx(nn,cell2)] == vNextIdx)
                    cellIdx2 = cell2;
            }
        //now, determine the types of the two relevant cells, and add an extra force if needed
        int cellType1 = h_ct.data[cellIdx1];
        int cellType2 = h_ct.data[cellIdx2];
        Dscalar gamma = 0.0;
        if(cellType1 != cellType2)
            {
            Dscalar gammaEdge;
            gammaEdge = gamma;
            Dscalar2 dnext = vcur-vnext;
            Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            h_fs.data[fsidx].x -= gammaEdge*dnext.x/dnnorm;
            h_fs.data[fsidx].y -= gammaEdge*dnext.y/dnnorm;
            };
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        Dscalar2 ftemp = make_Dscalar2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };
    };




/*!
call kernels to (1) do force sets calculation, then (2) add them up
*/
void VertexQuadraticEnergy::computeForcesGPU()
    {
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_fs(vertexForceSets,access_location::device, access_mode::overwrite);
    ArrayHandle<Dscalar2> d_f(vertexForces,access_location::device, access_mode::overwrite);

    int nForceSets = voroCur.getNumElements();
    gpu_avm_force_sets(
                    d_vcn.data,
                    d_vc.data,
                    d_vln.data,
                    d_AP.data,
                    d_APpref.data,
                    d_fs.data,
                    nForceSets,
                    KA,
                    KP
                    );

    gpu_avm_sum_force_sets(
                    d_fs.data,
                    d_f.data,
                    Nvertices);
    };


/////GONCA////
/////////////
///i is index of vertex i////
///j is index of vertex j////
///This function returns a matrix of four elements (x11,x12,x21,x22) which will be the values of the dynamical matrix
//x11 = d^2 / dr_{i,X} dr_{j,X}
//x12 = d^2 / dr_{i,X} dr_{j,Y}
//x21 = d^2 / dr_{i,X} dr_{j,X}
//x22 = d^2 / dr_{i,Y} dr_{j,Y}
Matrix2x2 VertexQuadraticEnergy::d2Edridrj(int i, int j){
  Matrix2x2  answer;
  answer.x11 = 0.0; answer.x12=0.0; answer.x21=0.0;answer.x22=0.0;
  Dscalar DijXX, DijXY, DijYX, DijYY;


  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);

  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
    //printf("px=%lf\n", posdata[(2*i)]);
    //printf("py=%lf\n", posdata[(2*i)+1]);
  }


  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }
  //printf("nmax=%i\n",nmax);
  
  //for(int ii=0; ii<Ncells; ++ii){
  // printf("h_cvn.data[%i]=%i\n", ii, h_cvn.data[ii]);
  //}
  
  vector<int> cellVerticesdata(Ncells*16);
  //for(int ii=0; ii<Ncells*16; ++ii){
  // printf("h_cv.data[%i]=%i\n", ii, h_cv.data[ii]);   
  //}

  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn]; 
    }
  }

  //for(int ii=0; ii<Ncells*16; ++ii){
  //printf("cellVerticesdata[%i]=%i\n", ii, cellVerticesdata[ii]);
  //}


  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }

  //for (int ii=0; ii<3*Nvertices; ++ii){
  // printf("VertexCellNeighborsList[%i]=%i\n", ii, VertexCellNeighborsList[ii]);
  //}


  Dscalar d2AdxiXdxjX = 0;
  Dscalar d2AdxiYdxjY = 0;
  Dscalar d2AdxiXdxjY = 0;
  Dscalar d2AdxiYdxjX = 0;
  Dscalar d2PdxiXdxjX = 0;
  Dscalar d2PdxiYdxjY = 0;
  Dscalar d2PdxiXdxjY = 0;
  Dscalar d2PdxiYdxjX = 0;
  Dscalar dAdxiXdAdxjX = 0;
  Dscalar dAdxiYdAdxjY = 0;
  Dscalar dAdxiXdAdxjY = 0;
  Dscalar dAdxiYdAdxjX = 0;
  Dscalar dPdxiXdPdxjX = 0;
  Dscalar dPdxiYdPdxjY = 0;
  Dscalar dPdxiXdPdxjY = 0;
  Dscalar dPdxiYdPdxjX = 0;


  int vertexIDj=j;
  int CellsofVertexj[3];
  int vertexIDi=i;
  int CellsofVertexi[3];

  //WHAT ARE THREE CELLS THAT SURROUNDS THE VERTEX i OR j 
  for(int ii=0; ii<3; ++ii){
    CellsofVertexj[ii]=VertexCellNeighborsList[3*vertexIDj+ii];
    CellsofVertexi[ii]=VertexCellNeighborsList[3*vertexIDi+ii];
    //    printf("CellsofVertexj[%i]=%i\n", ii, CellsofVertexj[ii]);
  }

  //for(int ii=0; ii<3; ++ii){
  // printf("CellsofVertexi[%i]=%i\n", ii, CellsofVertexi[ii]);
  //}

  //HOW MANY CELLS ARE COMMON FOR VERTEX i AND j AND WHAT ARE THOSE CELLS?
  int numberofCellsinCommon = 0;
  int CellsinCommon[10];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (CellsofVertexi[i] == CellsofVertexj[j]) {
    CellsinCommon[numberofCellsinCommon] = CellsofVertexi[i];
    numberofCellsinCommon++;
      }
    }
  }
  //  printf("numberofCellsinCommon=%i\n", numberofCellsinCommon);


  ///IF i=j/////
  if(vertexIDi==vertexIDj){
for(int kk=0; kk<numberofCellsinCommon; ++kk){
  int cellID=CellsinCommon[kk];
  //printf("cellID=%i\n", cellID);
  num = h_cvn.data[cellID];
  //printf("num=%i\n",num);
  int cellVerticesInfo[num];
  for (int nn=0; nn<num; ++nn){
    //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);
  }
  for (int nn=0; nn<num; ++nn){
    cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
    //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);
  }
  //CALCULATE THE PERIMETER OF THE CELL
  Dscalar PerimeterCell=0;
  Dscalar2 rij[num];
  for(int nn=0; nn<num; ++nn){
    Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
    //printf("x=%lf\n", rij[nn].x);
    //printf("y=%lf\n", rij[nn].y);
  }
  for(int nn=0; nn<num-1; ++nn){
    Dscalar xdiff=rij[nn].x-rij[nn+1].x;
    Dscalar ydiff=rij[nn].y-rij[nn+1].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  }
  Dscalar xdiff=rij[num-1].x-rij[0].x;
  Dscalar ydiff=rij[num-1].y-rij[0].y;
  PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  //printf("PerimeterCell=%lf\n", PerimeterCell);

  //GET vcur, vnext and vlast INFO FOR BOTH CELLS i and j
  int vcuri=vertexIDi;
  int vcurLabeli=0;
  for (int nn=0; nn<num; ++nn){
    if(cellVerticesInfo[nn]==vcuri){
      vcurLabeli=nn;
    }
  }
  //printf("vcurLabeli=%i\n",  vcurLabeli);

  int vnexti;
  int vlasti;
  if (vcurLabeli + 1 > num-1){
    vnexti = cellVerticesInfo[0]; 
  }else{
    vnexti = cellVerticesInfo[vcurLabeli + 1];
  }
  //printf("vnexti=%i\n", vnexti);
  if (vcurLabeli == 0){
    vlasti = cellVerticesInfo[num-1];
  }else{
    vlasti = cellVerticesInfo[vcurLabeli - 1];
  }
  //printf("vlasti=%i\n", vlasti);

  int vcurj=vertexIDj;
  int vcurLabelj=0;
  for (int nn=0; nn<num; ++nn){
    if(cellVerticesInfo[nn]==vcurj){
      vcurLabelj=nn;
    }
  }
  //printf("vcurLabelj=%i\n",  vcurLabelj);
  int vnextj;
  int vlastj;
  if (vcurLabelj + 1 > num-1){
    vnextj = cellVerticesInfo[0];
  }else{
    vnextj = cellVerticesInfo[vcurLabelj + 1];
  }
  //printf("vnextj=%i\n", vnextj);
  if (vcurLabelj == 0){
    vlastj = cellVerticesInfo[num-1];
  }else{
    vlastj = cellVerticesInfo[vcurLabelj - 1];
  }
  //printf("vlastj=%i\n", vlastj);

  Dscalar2 minDistvcurjvnextj;
  Box->minDist(h_vp.data[vcurj], h_vp.data[vnextj], minDistvcurjvnextj);
  Dscalar minDistvcurjvnextjNorm= sqrt(minDistvcurjvnextj.x*minDistvcurjvnextj.x+minDistvcurjvnextj.y*minDistvcurjvnextj.y);
  Dscalar minDistvcurjvnextjNormCube=minDistvcurjvnextjNorm*minDistvcurjvnextjNorm*minDistvcurjvnextjNorm;

  Dscalar2 minDistvcurivnexti;
  Box->minDist(h_vp.data[vcuri], h_vp.data[vnexti], minDistvcurivnexti);
  Dscalar minDistvcurivnextiNorm= sqrt(minDistvcurivnexti.x*minDistvcurivnexti.x+minDistvcurivnexti.y*minDistvcurivnexti.y);
  Dscalar minDistvcurivnextiNormCube=minDistvcurivnextiNorm*minDistvcurivnextiNorm*minDistvcurivnextiNorm;

  Dscalar2 minDistvlastjvcurj;
  Box->minDist(h_vp.data[vlastj], h_vp.data[vcurj], minDistvlastjvcurj);
  Dscalar minDistvlastjvcurjNorm= sqrt(minDistvlastjvcurj.x*minDistvlastjvcurj.x+minDistvlastjvcurj.y*minDistvlastjvcurj.y);
  Dscalar minDistvlastjvcurjNormCube= minDistvlastjvcurjNorm*minDistvlastjvcurjNorm*minDistvlastjvcurjNorm;

  Dscalar2 minDistvlastivcuri;
  Box->minDist(h_vp.data[vlasti], h_vp.data[vcuri], minDistvlastivcuri);
  Dscalar minDistvlastivcuriNorm= sqrt(minDistvlastivcuri.x*minDistvlastivcuri.x+minDistvlastivcuri.y*minDistvlastivcuri.y);
  Dscalar minDistvlastivcuriNormCube= minDistvlastivcuriNorm*minDistvlastivcuriNorm*minDistvlastivcuriNorm;

  Dscalar2 minDistvnextjvlastj;
  Box->minDist(h_vp.data[vnextj], h_vp.data[vlastj], minDistvnextjvlastj);
  Dscalar minDistvnextjvlastjNorm= sqrt(minDistvnextjvlastj.x*minDistvnextjvlastj.x+minDistvnextjvlastj.y*minDistvnextjvlastj.y);
  Dscalar minDistvnextjvlastjNormCube=minDistvnextjvlastjNorm*minDistvnextjvlastjNorm*minDistvnextjvlastjNorm;
  
  Dscalar2 minDistvnextivlasti;
  Box->minDist(h_vp.data[vnexti], h_vp.data[vlasti], minDistvnextivlasti);
  Dscalar minDistvnextivlastiNorm= sqrt(minDistvnextivlasti.x*minDistvnextivlasti.x+minDistvnextivlasti.y*minDistvnextivlasti.y);
  Dscalar minDistvnextivlastiNormCube=minDistvnextivlastiNorm*minDistvnextivlastiNorm*minDistvnextivlastiNorm;


  d2PdxiXdxjX += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
    *(1/minDistvcurjvnextjNorm - (minDistvcurjvnextj.x*minDistvcurjvnextj.x/minDistvcurjvnextjNormCube)
    + 1/minDistvlastjvcurjNorm - (minDistvlastjvcurj.x*minDistvlastjvcurj.x/minDistvlastjvcurjNormCube));
  //printf("d2PdxiXdxjX=%lf\n", d2PdxiXdxjX);

  d2PdxiYdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
    *(1/minDistvcurjvnextjNorm - (minDistvcurjvnextj.y*minDistvcurjvnextj.y/minDistvcurjvnextjNormCube)
      + 1/minDistvlastjvcurjNorm - (minDistvlastjvcurj.y*minDistvlastjvcurj.y/minDistvlastjvcurjNormCube));
  //printf("d2PdxiYdxjY=%lf\n", d2PdxiYdxjY);

  d2PdxiXdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
    *(- (minDistvcurjvnextj.y*minDistvcurjvnextj.x/minDistvcurjvnextjNormCube)
      - (minDistvlastjvcurj.y*minDistvlastjvcurj.x/minDistvlastjvcurjNormCube));
  //printf("d2PdxiXdxjY=%lf\n", d2PdxiXdxjY);

  d2PdxiYdxjX = d2PdxiXdxjY;

  dAdxiXdAdxjX += (1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.y*minDistvnextjvlastj.y;
  //printf("dAdxiXdAdxjX=%lf\n", dAdxiXdAdxjX);

  dAdxiYdAdxjY += (1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.x*minDistvnextjvlastj.x;
  //printf("dAdxiYdAdxjY=%lf\n", dAdxiYdAdxjY);

  dAdxiXdAdxjY += -(1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.y*minDistvnextjvlastj.x;
  //printf("dAdxiXdAdxjY=%lf\n", dAdxiXdAdxjY);

  dAdxiYdAdxjX += -(1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.x*minDistvnextjvlastj.y;
  //printf("dAdxiYdAdxjX=%lf\n", dAdxiYdAdxjX);

  dPdxiXdPdxjX += 2*h_m.data[cellID].y*(minDistvcurivnexti.x/minDistvcurivnextiNorm-minDistvlastivcuri.x/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.x/minDistvcurjvnextjNorm-minDistvlastjvcurj.x/minDistvlastjvcurjNorm);
  //printf("dPdxiXdPdxjX=%lf\n", dPdxiXdPdxjX);

  dPdxiYdPdxjY += 2*h_m.data[cellID].y*(minDistvcurivnexti.y/minDistvcurivnextiNorm-minDistvlastivcuri.y/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.y/minDistvcurjvnextjNorm-minDistvlastjvcurj.y/minDistvlastjvcurjNorm);
  //printf("dPdxiYdPdxjY=%lf\n", dPdxiYdPdxjY);

  dPdxiXdPdxjY += 2*h_m.data[cellID].y*(minDistvcurivnexti.x/minDistvcurivnextiNorm-minDistvlastivcuri.x/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.y/minDistvcurjvnextjNorm-minDistvlastjvcurj.y/minDistvlastjvcurjNorm);
  //  printf("dPdxiXdPdxjY=%lf\n", dPdxiXdPdxjY);

  dPdxiYdPdxjX += 2*h_m.data[cellID].y*(minDistvcurivnexti.y/minDistvcurivnextiNorm-minDistvlastivcuri.y/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.x/minDistvcurjvnextjNorm-minDistvlastjvcurj.x/minDistvlastjvcurjNorm);
  //printf("dPdxiYdPdxjX=%lf\n", dPdxiYdPdxjX);

//end of for loop for kk=numberofCellsinCommon  
}
//end of if vertexIDi==vertexIDj condition
}

  ///IF i!=j/////                                                                                                                 
  if(vertexIDi!=vertexIDj){
    for(int kk=0; kk<numberofCellsinCommon; ++kk){
      int cellID=CellsinCommon[kk];
      //  printf("cellID=%i\n", cellID);
      num = h_cvn.data[cellID];
      //printf("num=%i\n",num);
      int cellVerticesInfo[num];
      for (int nn=0; nn<num; ++nn){
    //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);
      }
      for (int nn=0; nn<num; ++nn){
    cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
    //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);
      }

      //CALCULATE THE PERIMETER AND AREA OF THE CELL                                                         
      Dscalar2 rij[num];
      for(int nn=0; nn<num; ++nn){
    Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
    //printf("x=%lf\n", rij[nn].x);                                                             
    //printf("y=%lf\n", rij[nn].y);                                                             
      }
      Dscalar PerimeterCell=0;
      for(int nn=0; nn<num-1; ++nn){
    Dscalar xdiff=rij[nn].x-rij[nn+1].x;
    Dscalar ydiff=rij[nn].y-rij[nn+1].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
      }
      Dscalar xdiff=rij[num-1].x-rij[0].x;
      Dscalar ydiff=rij[num-1].y-rij[0].y;
      PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
      //printf("PerimeterCell=%lf\n", PerimeterCell);

      Dscalar AreaCell=0;
      for(int nn=1; nn<num-1; ++nn){
        AreaCell+=(1./2.)*rij[nn].x*(rij[nn+1].y-rij[nn-1].y);
      }
      AreaCell+=(1./2.)*rij[0].x*(rij[1].y-rij[num-1].y);
      AreaCell+=(1./2.)*rij[num-1].x*(rij[0].y-rij[num-2].y);
      //printf("AreaCell=%lf\n", AreaCell);

      //GET vcur, vnext and vlast INFO FOR BOTH CELLS i and j                                                             
      int vcuri=vertexIDi;
      int vcurLabeli=0;
      for (int nn=0; nn<num; ++nn){
    if(cellVerticesInfo[nn]==vcuri){
      vcurLabeli=nn;
    }
      }
      //printf("vcurLabeli=%i\n",  vcurLabeli);

      int vnexti;
      int vlasti;
      if (vcurLabeli + 1 > num-1){
    vnexti = cellVerticesInfo[0];
      }else{
    vnexti = cellVerticesInfo[vcurLabeli + 1];
      }
      //printf("vnexti=%i\n", vnexti);
      if (vcurLabeli == 0){
    vlasti = cellVerticesInfo[num-1];
      }else{
    vlasti = cellVerticesInfo[vcurLabeli - 1];
      }
      //printf("vlasti=%i\n", vlasti);

      int vcurj=vertexIDj;
      int vcurLabelj=0;
      for (int nn=0; nn<num; ++nn){
    if(cellVerticesInfo[nn]==vcurj){
      vcurLabelj=nn;
    }
      }
      //printf("vcurLabelj=%i\n",  vcurLabelj);
      int vnextj;
      int vlastj;
      if (vcurLabelj + 1 > num-1){
    vnextj = cellVerticesInfo[0];
      }else{
    vnextj = cellVerticesInfo[vcurLabelj + 1];
      }
      //printf("vnextj=%i\n", vnextj);
      if (vcurLabelj == 0){
    vlastj = cellVerticesInfo[num-1];
      }else{
    vlastj = cellVerticesInfo[vcurLabelj - 1];
      }
      //printf("vlastj=%i\n", vlastj);
      Dscalar2 minDistvcurjvnextj;
      Box->minDist(h_vp.data[vcurj], h_vp.data[vnextj], minDistvcurjvnextj);
      Dscalar minDistvcurjvnextjNorm= sqrt(minDistvcurjvnextj.x*minDistvcurjvnextj.x+minDistvcurjvnextj.y*minDistvcurjvnextj.y);
      Dscalar minDistvcurjvnextjNormCube=minDistvcurjvnextjNorm*minDistvcurjvnextjNorm*minDistvcurjvnextjNorm;

      Dscalar2 minDistvcurivnexti;
      Box->minDist(h_vp.data[vcuri], h_vp.data[vnexti], minDistvcurivnexti);
      Dscalar minDistvcurivnextiNorm= sqrt(minDistvcurivnexti.x*minDistvcurivnexti.x+minDistvcurivnexti.y*minDistvcurivnexti.y);
      Dscalar minDistvcurivnextiNormCube=minDistvcurivnextiNorm*minDistvcurivnextiNorm*minDistvcurivnextiNorm;

      Dscalar2 minDistvlastjvcurj;
      Box->minDist(h_vp.data[vlastj], h_vp.data[vcurj], minDistvlastjvcurj);
      Dscalar minDistvlastjvcurjNorm= sqrt(minDistvlastjvcurj.x*minDistvlastjvcurj.x+minDistvlastjvcurj.y*minDistvlastjvcurj.y);
      Dscalar minDistvlastjvcurjNormCube= minDistvlastjvcurjNorm*minDistvlastjvcurjNorm*minDistvlastjvcurjNorm;

      Dscalar2 minDistvlastivcuri;
      Box->minDist(h_vp.data[vlasti], h_vp.data[vcuri], minDistvlastivcuri);
      Dscalar minDistvlastivcuriNorm= sqrt(minDistvlastivcuri.x*minDistvlastivcuri.x+minDistvlastivcuri.y*minDistvlastivcuri.y);
      Dscalar minDistvlastivcuriNormCube= minDistvlastivcuriNorm*minDistvlastivcuriNorm*minDistvlastivcuriNorm;

      Dscalar2 minDistvnextjvlastj;
      Box->minDist(h_vp.data[vnextj], h_vp.data[vlastj], minDistvnextjvlastj);
      Dscalar minDistvnextjvlastjNorm= sqrt(minDistvnextjvlastj.x*minDistvnextjvlastj.x+minDistvnextjvlastj.y*minDistvnextjvlastj.y);
      Dscalar minDistvnextjvlastjNormCube=minDistvnextjvlastjNorm*minDistvnextjvlastjNorm*minDistvnextjvlastjNorm;

      Dscalar2 minDistvnextivlasti;
      Box->minDist(h_vp.data[vnexti], h_vp.data[vlasti], minDistvnextivlasti);
      Dscalar minDistvnextivlastiNorm= sqrt(minDistvnextivlasti.x*minDistvnextivlasti.x+minDistvnextivlasti.y*minDistvnextivlasti.y);
      Dscalar minDistvnextivlastiNormCube=minDistvnextivlastiNorm*minDistvnextivlastiNorm*minDistvnextivlastiNorm;

      dAdxiXdAdxjX += (1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.y*minDistvnextjvlastj.y;
      //printf("dAdxiXdAdxjX=%lf\n", dAdxiXdAdxjX);

      dAdxiYdAdxjY += (1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.x*minDistvnextjvlastj.x;
      //printf("dAdxiYdAdxjY=%lf\n", dAdxiYdAdxjY);

      dAdxiXdAdxjY += -(1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.y*minDistvnextjvlastj.x;
      //printf("dAdxiXdAdxjY=%lf\n", dAdxiXdAdxjY);

      dAdxiYdAdxjX += -(1./4.)*2.*h_m.data[cellID].x*minDistvnextivlasti.x*minDistvnextjvlastj.y;
      //printf("dAdxiYdAdxjX=%lf\n", dAdxiYdAdxjX);

      dPdxiXdPdxjX += 2*h_m.data[cellID].y*(minDistvcurivnexti.x/minDistvcurivnextiNorm-minDistvlastivcuri.x/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.x/minDistvcurjvnextjNorm-minDistvlastjvcurj.x/minDistvlastjvcurjNorm);
      //printf("dPdxiXdPdxjX=%lf\n", dPdxiXdPdxjX);

      dPdxiYdPdxjY += 2*h_m.data[cellID].y*(minDistvcurivnexti.y/minDistvcurivnextiNorm-minDistvlastivcuri.y/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.y/minDistvcurjvnextjNorm-minDistvlastjvcurj.y/minDistvlastjvcurjNorm);
      //printf("dPdxiYdPdxjY=%lf\n", dPdxiYdPdxjY);

      dPdxiXdPdxjY += 2*h_m.data[cellID].y*(minDistvcurivnexti.x/minDistvcurivnextiNorm-minDistvlastivcuri.x/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.y/minDistvcurjvnextjNorm-minDistvlastjvcurj.y/minDistvlastjvcurjNorm);
      //printf("dPdxiXdPdxjY=%lf\n", dPdxiXdPdxjY);

      dPdxiYdPdxjX += 2*h_m.data[cellID].y*(minDistvcurivnexti.y/minDistvcurivnextiNorm-minDistvlastivcuri.y/minDistvlastivcuriNorm)
    *(minDistvcurjvnextj.x/minDistvcurjvnextjNorm-minDistvlastjvcurj.x/minDistvlastjvcurjNorm);
      //printf("dPdxiYdPdxjX=%lf\n", dPdxiYdPdxjX);

      ///IF vertexIDi == vnextj
      if(vertexIDi == vnextj){
    d2AdxiXdxjY += 2*h_m.data[cellID].x*(AreaCell - h_APpref.data[cellID].x)*(-1./2.);
    //printf("d2AdxiXdxjY=%lf\n", d2AdxiXdxjY);

        d2AdxiYdxjX += 2*h_m.data[cellID].x*(AreaCell - h_APpref.data[cellID].x)*(1./2.);
        //printf("d2AdxiYdxjX=%lf\n", d2AdxiYdxjX);

    d2PdxiXdxjX += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
      *(-1/minDistvcurjvnextjNorm + (minDistvcurjvnextj.x*minDistvcurjvnextj.x/minDistvcurjvnextjNormCube));
    //printf("d2PdxiXdxjX=%lf\n", d2PdxiXdxjX);


    d2PdxiYdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
      *(-1/minDistvcurjvnextjNorm + (minDistvcurjvnextj.y*minDistvcurjvnextj.y/minDistvcurjvnextjNormCube));
    //printf("d2PdxiYdxjY=%lf\n", d2PdxiYdxjY);

    d2PdxiXdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
      *((minDistvcurjvnextj.y*minDistvcurjvnextj.x/minDistvcurjvnextjNormCube));
    //printf("d2PdxiXdxjY=%lf\n", d2PdxiXdxjY);

    d2PdxiYdxjX = d2PdxiXdxjY;

      ///end of if vertexIDi == vnextj condition  
      }

      ///IF vertexIDi == vlastj                                                                                                                  
      if(vertexIDi == vlastj){
        d2AdxiXdxjY += 2*h_m.data[cellID].x*(AreaCell - h_APpref.data[cellID].x)*(1./2.);
        //printf("d2AdxiXdxjY=%lf\n", d2AdxiXdxjY);

        d2AdxiYdxjX += 2*h_m.data[cellID].x*(AreaCell - h_APpref.data[cellID].x)*(-1./2.);
        //printf("d2AdxiYdxjX=%lf\n", d2AdxiYdxjX);

        d2PdxiXdxjX += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
          *(-1/minDistvlastjvcurjNorm + (minDistvlastjvcurj.x*minDistvlastjvcurj.x/minDistvlastjvcurjNormCube));
        //printf("d2PdxiXdxjX=%lf\n", d2PdxiXdxjX);

        d2PdxiYdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
          *(-1/minDistvlastjvcurjNorm + (minDistvlastjvcurj.y*minDistvlastjvcurj.y/minDistvlastjvcurjNormCube));
        //printf("d2PdxiYdxjY=%lf\n", d2PdxiYdxjY);

        d2PdxiXdxjY += 2*h_m.data[cellID].y*(PerimeterCell - h_APpref.data[cellID].y)
          *((minDistvlastjvcurj.y*minDistvlastjvcurj.x/minDistvlastjvcurjNormCube));
        //printf("d2PdxiXdxjY=%lf\n", d2PdxiXdxjY);

        d2PdxiYdxjX = d2PdxiXdxjY;
    //printf("d2PdxiYdxjX=%lf\n", d2PdxiYdxjX);
    ///end of if vertexIDi == vnextj condition                                                                                                  
      }


      //end of for loop for kk=numberofCellsinCommon    
    }
    //end of if vertexIDi!=vertexIDj condition
  }

 DijXX=d2AdxiXdxjX + d2PdxiXdxjX + dAdxiXdAdxjX + dPdxiXdPdxjX;
 DijXY=d2AdxiXdxjY + d2PdxiXdxjY + dAdxiXdAdxjY + dPdxiXdPdxjY;
 DijYX=d2AdxiYdxjX + d2PdxiYdxjX + dAdxiYdAdxjX + dPdxiYdPdxjX;
 DijYY=d2AdxiYdxjY + d2PdxiYdxjY + dAdxiYdAdxjY + dPdxiYdPdxjY;
 //printf("DijXX=%lf\n", DijXX);
 //printf("DijXY=%lf\n", DijXY);
 //printf("DijYX=%lf\n", DijYX);
 //printf("DijYY=%lf\n", DijYY);

 answer.x11=DijXX;
 answer.x12=DijXY;
 answer.x21=DijYX;
 answer.x22=DijYY;

 return answer;
};


Dscalar VertexQuadraticEnergy::d2Edgamma2(int cellID){

  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);

  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
  }

  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }                                                                                                     

  vector<int> cellVerticesdata(Ncells*16);  

  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn];
    }
  }                                                                

  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }

  num = h_cvn.data[cellID];
  //printf("num=%i\n",num);                                                                               
  int cellVerticesInfo[num];
  //for (int nn=0; nn<num; ++nn){
    //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);          
  //}
  for (int nn=0; nn<num; ++nn){
    cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
    //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);                                     
  }

  //CALCULATE THE PERIMETER OF THE CELL                                                             
  Dscalar PerimeterCell=0;
  Dscalar2 rij[num];
  Dscalar2 poslist[num];
  for(int nn=0; nn<num; ++nn){
    Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
    //printf("x=%lf\n", rij[nn].x);                                                                 
    //printf("y=%lf\n", rij[nn].y);                                                                 
  }
  for(int nn=0; nn<num-1; ++nn){
    Dscalar xdiff=rij[nn].x-rij[nn+1].x;
    Dscalar ydiff=rij[nn].y-rij[nn+1].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  }
  Dscalar xdiff=rij[num-1].x-rij[0].x;
  Dscalar ydiff=rij[num-1].y-rij[0].y;
  PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  //printf("PerimeterCell=%lf\n", PerimeterCell); 

  for(int nn=0; nn<num; ++nn){
    poslist[nn]=h_vp.data[cellVerticesInfo[nn]];
    //printf("x=%lf\n", poslist[nn].x);                                                                  
    // printf("y=%lf\n", poslist[nn].y);                                                              
  }

  Dscalar x11,x12,x21,x22;
  Box->getBoxDims(x11,x12,x21,x22);
  Dscalar Ly=x22;
  int qListiiplus1y[num];
  for(int nn=0; nn<num; ++nn){
    qListiiplus1y[nn]=0.;
  }
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].y < poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
      qListiiplus1y[nn]=1;
    }
  }
  if( (poslist[num-1].y<poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
    qListiiplus1y[num-1]=1;
  }

  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].y>poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
      qListiiplus1y[nn]=-1;
    }
  }
  if( (poslist[num-1].y>poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
    qListiiplus1y[num-1]=-1;
  }
  //for(int nn=0; nn<num; ++nn){
  //printf("qListiiplus1y=%i\n",qListiiplus1y[nn]);
  //}

  Dscalar ymin=100.;
  int Positionofy0;
  for(int nn=0; nn<num; ++nn){
    if(poslist[nn].y<ymin){
      ymin=poslist[nn].y;
      Positionofy0=nn;
    }
  }
  Dscalar2 pos0=poslist[Positionofy0];
  //printf("pos0x=%lf\n pos0y=%lf\n", pos0.x,pos0.y);


  Dscalar dAdgamma=0.;
  for(int nn=0; nn<num-1; ++nn){
    Dscalar2 minDisti0;
    Dscalar2 minDistiplus10;
    Box->minDist(poslist[nn], pos0, minDisti0);
    Box->minDist(poslist[nn+1], pos0, minDistiplus10);
    dAdgamma+=(1./2.)*qListiiplus1y[nn]*Ly*(minDisti0.y+minDistiplus10.y);
  }
  Dscalar2 minDistlast0;
  Dscalar2 minDistfirst0;
  Box->minDist(poslist[num-1], pos0, minDistlast0);
  Box->minDist(poslist[0], pos0, minDistfirst0);
  dAdgamma+=(1./2.)*qListiiplus1y[num-1]*Ly*(minDistlast0.y+minDistfirst0.y);
  //printf("dAdgamma=%lf\n",dAdgamma);

  Dscalar dPdgamma=0.;
  for(int nn=0; nn<num-1; ++nn){
    Dscalar2 minDistiiplus1;
    Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
    dPdgamma+=qListiiplus1y[nn]*Ly*(minDistiiplus1.x)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y);
  }
  Dscalar2 minDistlastfirst;
  Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
  dPdgamma+=qListiiplus1y[num-1]*Ly*(minDistlastfirst.x)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y);
  //printf("dPdgamma=%lf\n",dPdgamma);

  Dscalar d2Pdgamma2=0.;
  for(int nn=0; nn<num-1; ++nn){
    Dscalar2 minDistiiplus1;
    Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
    Dscalar minDistiiplus1Norm=sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y);
    d2Pdgamma2 += ( (qListiiplus1y[nn]*Ly*qListiiplus1y[nn]*Ly) / minDistiiplus1Norm)
      -( (qListiiplus1y[nn]*Ly*qListiiplus1y[nn]*Ly*minDistiiplus1.x*minDistiiplus1.x) / (minDistiiplus1Norm*minDistiiplus1Norm*minDistiiplus1Norm));
  }
  Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
  Dscalar minDistlastfirstNorm=sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y);
  d2Pdgamma2 += ( (qListiiplus1y[num-1]*Ly*qListiiplus1y[num-1]*Ly) / minDistlastfirstNorm)
    -( (qListiiplus1y[num-1]*Ly*qListiiplus1y[num-1]*Ly*minDistlastfirst.x*minDistlastfirst.x) / (minDistlastfirstNorm*minDistlastfirstNorm*minDistlastfirstNorm));
  //printf("d2Pdgamma2=%lf\n", d2Pdgamma2);
  //printf("total=%lf\n", ((2*KA*dAdgamma*dAdgamma) + (2*KP*(PerimeterCell-h_APpref.data[cellID].y)*d2Pdgamma2) + (2*KP*dPdgamma*dPdgamma)) );

  return ((2*h_m.data[cellID].x*dAdgamma*dAdgamma) + (2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)*d2Pdgamma2) + (2*h_m.data[cellID].y*dPdgamma*dPdgamma));

}


Dscalar2 VertexQuadraticEnergy::d2Edgammadrialpha(int vertexi){

  Dscalar2 d2Edgammadrialpha;

  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);

  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
  }

  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }

  vector<int> cellVerticesdata(Ncells*16);

  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn];
    }
  }

  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }

  int vertexIDi=vertexi;
  int CellsofVertexi[3];

  //WHAT ARE THREE CELLS THAT SURROUNDS THE VERTEX i         
  for(int ii=0; ii<3; ++ii){
    CellsofVertexi[ii]=VertexCellNeighborsList[3*vertexIDi+ii];
  }

  Dscalar d2EdgammadriXvalue = 0.;
  Dscalar d2EdgammadriYvalue = 0;
  for(int kk=0; kk<3; ++kk){
    int cellID=CellsofVertexi[kk];
    num = h_cvn.data[cellID];
    //printf("num=%i\n",num);
    int cellVerticesInfo[num];
    //for (int nn=0; nn<num; ++nn){      
    //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);   
    //}
    for (int nn=0; nn<num; ++nn){
      cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
      //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);
    }
    //CALCULATE THE PERIMETER OF THE CELL                                                                                    
    Dscalar PerimeterCell=0;
    Dscalar2 rij[num];
    for(int nn=0; nn<num; ++nn){
      Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
      //printf("x=%lf\n", rij[nn].x);                                                                                        
      //printf("y=%lf\n", rij[nn].y);                                                                                        
    }
    for(int nn=0; nn<num-1; ++nn){
      Dscalar xdiff=rij[nn].x-rij[nn+1].x;
      Dscalar ydiff=rij[nn].y-rij[nn+1].y;
      PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
    }
    Dscalar xdiff=rij[num-1].x-rij[0].x;
    Dscalar ydiff=rij[num-1].y-rij[0].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
    //printf("PerimeterCell=%lf\n", PerimeterCell); 

    Dscalar AreaCell=0;
    for(int nn=1; nn<num-1; ++nn){
      AreaCell+=(1./2.)*rij[nn].x*(rij[nn+1].y-rij[nn-1].y);
    }
    AreaCell+=(1./2.)*rij[0].x*(rij[1].y-rij[num-1].y);
    AreaCell+=(1./2.)*rij[num-1].x*(rij[0].y-rij[num-2].y);
    //printf("AreaCell=%lf\n", AreaCell); 

    Dscalar2 poslist[num];
    for(int nn=0; nn<num; ++nn){
      poslist[nn]=h_vp.data[cellVerticesInfo[nn]];
      //printf("x=%lf\n", poslist[nn].x);                                                                  
      // printf("y=%lf\n", poslist[nn].y);                                                              
    }

    Dscalar x11,x12,x21,x22;
    Box->getBoxDims(x11,x12,x21,x22);
    Dscalar Ly=x22;
    int qListiiplus1y[num];
    for(int nn=0; nn<num; ++nn){
      qListiiplus1y[nn]=0.;
    }
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].y < poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
    qListiiplus1y[nn]=1;
      }
    }
    if( (poslist[num-1].y<poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
      qListiiplus1y[num-1]=1;
    }

    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].y>poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
    qListiiplus1y[nn]=-1;
      }
    }
    if( (poslist[num-1].y>poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
      qListiiplus1y[num-1]=-1;
    }
    //for(int nn=0; nn<num; ++nn){
    //printf("qListiiplus1y=%i\n",qListiiplus1y[nn]);
    //}

    Dscalar ymin=100.;
    int Positionofy0;
    for(int nn=0; nn<num; ++nn){
      if(poslist[nn].y<ymin){
    ymin=poslist[nn].y;
    Positionofy0=nn;
      }
    }
    Dscalar2 pos0=poslist[Positionofy0];
    //printf("pos0x=%lf\n pos0y=%lf\n", pos0.x,pos0.y);



    //GET vcur, vnext and vlast                                                             
    int vcuri=vertexIDi;
    int vcurLabeli=0;
    for (int nn=0; nn<num; ++nn){
      if(cellVerticesInfo[nn]==vcuri){
    vcurLabeli=nn;
      }
    }
    //printf("vcurLabeli=%i\n",  vcurLabeli);                                                                            

    int vnexti;
    int vlasti;
    if (vcurLabeli + 1 > num-1){
      vnexti = cellVerticesInfo[0];
    }else{
      vnexti = cellVerticesInfo[vcurLabeli + 1];
    }
    // printf("vnexti=%i\n", vnexti);                                                                                     
    if (vcurLabeli == 0){
      vlasti = cellVerticesInfo[num-1];
    }else{
      vlasti = cellVerticesInfo[vcurLabeli - 1];
    }
    //printf("vlasti=%i\n", vlasti);    


    int qCurNexty = 0;
    if( (h_vp.data[vcuri].y < h_vp.data[vnexti].y) && ( abs(h_vp.data[vcuri].y - h_vp.data[vnexti].y) > (Ly/2.)) ){
      qCurNexty=1;
    }
    if( (h_vp.data[vcuri].y > h_vp.data[vnexti].y) && ( abs(h_vp.data[vcuri].y - h_vp.data[vnexti].y) > (Ly/2.) ) ){
      qCurNexty=-1;
    }

    int qLastCury=0;
    if( (h_vp.data[vlasti].y < h_vp.data[vcuri].y) && ( abs(h_vp.data[vlasti].y - h_vp.data[vcuri].y) > (Ly/2.)) ){
      qLastCury=1;
    }
    if( (h_vp.data[vlasti].y > h_vp.data[vcuri].y) && ( abs(h_vp.data[vlasti].y - h_vp.data[vcuri].y) > (Ly/2.) ) ){
      qLastCury=-1;
    }
    //printf("qCurNexty=%i\n",qCurNexty);
    //printf("qLastCury=%i\n",qLastCury);

    int qNextLast=0;
    if( (h_vp.data[vnexti].y < h_vp.data[vlasti].y) && ( abs(h_vp.data[vnexti].y - h_vp.data[vlasti].y) > (Ly/2.)) ){
      qNextLast=1;
    }
    if( (h_vp.data[vnexti].y > h_vp.data[vlasti].y) && ( abs(h_vp.data[vnexti].y - h_vp.data[vlasti].y) > (Ly/2.) ) ){
      qNextLast=-1;
    }



    Dscalar dAdgamma=0.;
    for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDisti0;
      Dscalar2 minDistiplus10;
      Box->minDist(poslist[nn], pos0, minDisti0);
      Box->minDist(poslist[nn+1], pos0, minDistiplus10);
      dAdgamma+=(1./2.)*qListiiplus1y[nn]*Ly*(minDisti0.y+minDistiplus10.y);
    }
    Dscalar2 minDistlast0;
    Dscalar2 minDistfirst0;
    Box->minDist(poslist[num-1], pos0, minDistlast0);
    Box->minDist(poslist[0], pos0, minDistfirst0);
    dAdgamma+=(1./2.)*qListiiplus1y[num-1]*Ly*(minDistlast0.y+minDistfirst0.y);
    //printf("dAdgamma=%lf\n",dAdgamma);

    Dscalar dPdgamma=0.;
    for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDistiiplus1;
      Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
      dPdgamma+=qListiiplus1y[nn]*Ly*(minDistiiplus1.x)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y);
    }
    Dscalar2 minDistlastfirst;
    Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
    dPdgamma+=qListiiplus1y[num-1]*Ly*(minDistlastfirst.x)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y);
    //printf("dPdgamma=%lf\n",dPdgamma);


    Dscalar2 minDistvcurivnexti;
    Box->minDist(h_vp.data[vcuri], h_vp.data[vnexti], minDistvcurivnexti);
    Dscalar minDistvcurivnextiNorm= sqrt(minDistvcurivnexti.x*minDistvcurivnexti.x+minDistvcurivnexti.y*minDistvcurivnexti.y);
    Dscalar minDistvcurivnextiNormCube=minDistvcurivnextiNorm*minDistvcurivnextiNorm*minDistvcurivnextiNorm;
    Dscalar2 minDistvlastivcuri;
    Box->minDist(h_vp.data[vlasti], h_vp.data[vcuri], minDistvlastivcuri);
    Dscalar minDistvlastivcuriNorm= sqrt(minDistvlastivcuri.x*minDistvlastivcuri.x+minDistvlastivcuri.y*minDistvlastivcuri.y);
    Dscalar minDistvlastivcuriNormCube= minDistvlastivcuriNorm*minDistvlastivcuriNorm*minDistvlastivcuriNorm;

    Dscalar2 minDistvnextivlasti;
    Box->minDist(h_vp.data[vnexti], h_vp.data[vlasti], minDistvnextivlasti);

    d2EdgammadriXvalue+=h_m.data[cellID].x*dAdgamma*minDistvnextivlasti.y
      +2*h_m.data[cellID].y*dPdgamma*(minDistvcurivnexti.x/minDistvcurivnextiNorm - minDistvlastivcuri.x/minDistvlastivcuriNorm)
      +2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)
      *( (qCurNexty*Ly / minDistvcurivnextiNorm) - (qCurNexty*Ly*minDistvcurivnexti.x*minDistvcurivnexti.x / minDistvcurivnextiNormCube)
     - (qLastCury*Ly / minDistvlastivcuriNorm) + (qLastCury*Ly*minDistvlastivcuri.x*minDistvlastivcuri.x / minDistvlastivcuriNormCube) );
   //printf("d2EdgammadriXvalue=%lf\n",d2EdgammadriXvalue);

    d2EdgammadriYvalue+=h_m.data[cellID].x*dAdgamma*(-minDistvnextivlasti.x)
      +2*h_m.data[cellID].y*dPdgamma*(minDistvcurivnexti.y/minDistvcurivnextiNorm - minDistvlastivcuri.y/minDistvlastivcuriNorm)
      +h_m.data[cellID].x*(AreaCell-h_APpref.data[cellID].x)*(-qNextLast)*Ly
      +2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)*(-(qCurNexty*Ly*minDistvcurivnexti.x*minDistvcurivnexti.y / minDistvcurivnextiNormCube)
                                                     + (qLastCury*Ly*minDistvlastivcuri.x*minDistvlastivcuri.y / minDistvlastivcuriNormCube));
    //printf("d2EdgammadriYvalue=%lf\n",d2EdgammadriYvalue);

  }

  d2Edgammadrialpha.x=d2EdgammadriXvalue;
  d2Edgammadrialpha.y=d2EdgammadriYvalue;

  return d2Edgammadrialpha;  

}


Dscalar VertexQuadraticEnergy::PureSheard2Edgamma2(int cellID){
    
  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);
    
  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
  }
    
  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }
    
  vector<int> cellVerticesdata(Ncells*16);
    
  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn];
    }
  }
    
  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }
    
  num = h_cvn.data[cellID];
  //printf("num=%i\n",num);
  int cellVerticesInfo[num];
  //for (int nn=0; nn<num; ++nn){
  //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);
  //}
  for (int nn=0; nn<num; ++nn){
    cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
    //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);
  }
    
  //CALCULATE THE PERIMETER OF THE CELL
  Dscalar PerimeterCell=0;
  Dscalar2 rij[num];
  Dscalar2 poslist[num];
  for(int nn=0; nn<num; ++nn){
    Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
    //printf("x=%lf\n", rij[nn].x);
    //printf("y=%lf\n", rij[nn].y);
  }
  for(int nn=0; nn<num-1; ++nn){
    Dscalar xdiff=rij[nn].x-rij[nn+1].x;
    Dscalar ydiff=rij[nn].y-rij[nn+1].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  }
  Dscalar xdiff=rij[num-1].x-rij[0].x;
  Dscalar ydiff=rij[num-1].y-rij[0].y;
  PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
  //printf("PerimeterCell=%lf\n", PerimeterCell);
    
  Dscalar AreaCell=0;
  for(int nn=1; nn<num-1; ++nn){
    AreaCell+=(1./2.)*rij[nn].x*(rij[nn+1].y-rij[nn-1].y);
  }
  AreaCell+=(1./2.)*rij[0].x*(rij[1].y-rij[num-1].y);
  AreaCell+=(1./2.)*rij[num-1].x*(rij[0].y-rij[num-2].y);
  //printf("AreaCell=%lf\n", AreaCell);
    
    
  for(int nn=0; nn<num; ++nn){
    poslist[nn]=h_vp.data[cellVerticesInfo[nn]];
    //printf("x=%lf\n", poslist[nn].x);
    // printf("y=%lf\n", poslist[nn].y);
  }
    
  Dscalar x11,x12,x21,x22;
  Box->getBoxDims(x11,x12,x21,x22);
  Dscalar Lx=x11;
  Dscalar Ly=x22;
  int qListiiplus1y[num];
  for(int nn=0; nn<num; ++nn){
    qListiiplus1y[nn]=0.;
  }
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].y < poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
      qListiiplus1y[nn]=1;
    }
  }
  if( (poslist[num-1].y<poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
    qListiiplus1y[num-1]=1;
  }
    
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].y>poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
      qListiiplus1y[nn]=-1;
    }
  }
  if( (poslist[num-1].y>poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
    qListiiplus1y[num-1]=-1;
  }
  //for(int nn=0; nn<num; ++nn){
  //printf("qListiiplus1y=%i\n",qListiiplus1y[nn]);
  //}
    
    
  int qListiiplus1x[num];
  for(int nn=0; nn<num; ++nn){
    qListiiplus1x[nn]=0.;
  }
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].x < poslist[nn+1].x) && (abs(poslist[nn].x-poslist[nn+1].x)>(Lx/2.)) ){
      qListiiplus1x[nn]=1;
    }
  }
  if( (poslist[num-1].x<poslist[0].x) && (abs(poslist[num-1].x-poslist[0].x)>(Lx/2.)) ){
    qListiiplus1x[num-1]=1;
  }
    
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn].x>poslist[nn+1].x) && (abs(poslist[nn].x-poslist[nn+1].x)>(Lx/2.)) ){
      qListiiplus1x[nn]=-1;
    }
  }
  if( (poslist[num-1].x>poslist[0].x) && (abs(poslist[num-1].x-poslist[0].x)>(Lx/2.)) ){
    qListiiplus1x[num-1]=-1;
  }
    
    
  Dscalar ymin=100.;
  int Positionofy0;
  for(int nn=0; nn<num; ++nn){
    if(poslist[nn].y<ymin){
      ymin=poslist[nn].y;
      Positionofy0=nn;
    }
  }
  Dscalar2 pos0=poslist[Positionofy0];
  //printf("pos0x=%lf\n pos0y=%lf\n", pos0.x,pos0.y);
    
  int qListi0y[num];
  for(int nn=0; nn<num; ++nn){
    qListi0y[nn]=0.;
  }
  for(int nn=0; nn<num; ++nn){
    if( (poslist[nn].y < pos0.y) && (abs(poslist[nn].y-pos0.y)>(Ly/2.)) ){
      qListi0y[nn]=1;
    }
  }
  for(int nn=0; nn<num; ++nn){
    if( (poslist[nn].y > pos0.y) && (abs(poslist[nn].y-pos0.y)>(Ly/2.)) ){
      qListi0y[nn]=-1;
    }
  }
    
  int qListiplus10y[num];
  for(int nn=0; nn<num; ++nn){
    qListiplus10y[nn]=0.;
  }
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn+1].y < pos0.y) && (abs(poslist[nn+1].y-pos0.y)>(Ly/2.)) ){
      qListiplus10y[nn]=1;
    }
  }
  if( (poslist[0].y < pos0.y) && (abs(poslist[0].y-pos0.y)>(Ly/2.)) ){
    qListiplus10y[0]=1;
  }
  for(int nn=0; nn<num-1; ++nn){
    if( (poslist[nn+1].y > pos0.y) && (abs(poslist[nn+1].y-pos0.y)>(Ly/2.)) ){
      qListiplus10y[nn]=-1;
    }
  }
  if( (poslist[0].y > pos0.y) && (abs(poslist[0].y-pos0.y)>(Ly/2.)) ){
    qListiplus10y[0]=-1;
  }


  Dscalar dAdgamma=0.;
  for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDisti0;
      Dscalar2 minDistiplus10;
      Dscalar2 minDistiiplus1;
      Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
      Box->minDist(poslist[nn], pos0, minDisti0);
      Box->minDist(poslist[nn+1], pos0, minDistiplus10);
      dAdgamma+=(1./2.)*(qListiiplus1x[nn]*Lx*(minDisti0.y+minDistiplus10.y)+minDistiiplus1.x*(-qListi0y[nn]*Ly-qListiplus10y[nn]*Ly));
    }
    Dscalar2 minDistlast0;
    Dscalar2 minDistfirst0;
    Box->minDist(poslist[num-1], pos0, minDistlast0);
    Box->minDist(poslist[0], pos0, minDistfirst0);
    Dscalar2 minDistlastfirst;
    Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
    dAdgamma+=(1./2.)*(qListiiplus1x[num-1]*Lx*(minDistlast0.y+minDistfirst0.y)+minDistlastfirst.x*(-qListi0y[num-1]*Ly-qListiplus10y[num-1]*Ly));
    //printf("dAdgamma=%lf\n",dAdgamma);



    Dscalar dPdgamma=0.;
    for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDistiiplus1;
      Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
      dPdgamma+=(qListiiplus1x[nn]*Lx*(minDistiiplus1.x)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y))+(-qListiiplus1y[nn]*Ly*(minDistiiplus1.y)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y));
  }
  minDistlastfirst;
  Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
  dPdgamma+=(qListiiplus1x[num-1]*Lx*(minDistlastfirst.x)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y))+(-qListiiplus1y[num-1]*Ly*(minDistlastfirst.y)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y));
  //printf("dPdgamma=%lf\n",dPdgamma);

  Dscalar d2Adgamma2=0.;
  for(int nn=0; nn<num-1; ++nn){
    Dscalar2 minDistiiplus1;
    Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
    Dscalar minDistiiplus1Norm=sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y);
    d2Adgamma2 += (-qListiiplus1x[nn]*Lx + minDistiiplus1.x)*(qListi0y[nn]*Ly+qListiplus10y[nn]*Ly);
  }
  minDistlastfirst;
  Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
  d2Adgamma2 += (-qListiiplus1x[num-1]*Lx + minDistlastfirst.x)*(qListi0y[num-1]*Ly+qListiplus10y[num-1]*Ly);


  Dscalar d2Pdgamma2=0.;
  for(int nn=0; nn<num-1; ++nn){
    Dscalar2 minDistiiplus1;
    Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
    Dscalar minDistiiplus1Norm=sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y);
    d2Pdgamma2 += ( ( (qListiiplus1x[nn]*Lx*qListiiplus1x[nn]*Lx) + (qListiiplus1y[nn]*Ly*qListiiplus1y[nn]*Ly) ) / minDistiiplus1Norm)
      +( (2* qListiiplus1x[nn]*Lx*qListiiplus1y[nn]*Ly*minDistiiplus1.x*minDistiiplus1.y) / (minDistiiplus1Norm*minDistiiplus1Norm*minDistiiplus1Norm))
      -( ((minDistiiplus1.x*minDistiiplus1.x*qListiiplus1x[nn]*Lx*qListiiplus1x[nn]*Lx) + (minDistiiplus1.y*minDistiiplus1.y*qListiiplus1y[nn]*Ly*qListiiplus1y[nn]*Ly))/ (minDistiiplus1Norm*minDistiiplus1Norm*minDistiiplus1Norm) );
  }
  Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
  Dscalar minDistlastfirstNorm=sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y);
  d2Pdgamma2 += ( ( (qListiiplus1x[num-1]*Lx*qListiiplus1x[num-1]*Lx) + (qListiiplus1y[num-1]*Ly*qListiiplus1y[num-1]*Ly) ) / minDistlastfirstNorm)
    +( (2* qListiiplus1x[num-1]*Lx*qListiiplus1y[num-1]*Ly*minDistlastfirst.x*minDistlastfirst.y) / (minDistlastfirstNorm*minDistlastfirstNorm*minDistlastfirstNorm))
    -( ((minDistlastfirst.x*minDistlastfirst.x*qListiiplus1x[num-1]*Lx*qListiiplus1x[num-1]*Lx) + (minDistlastfirst.y*minDistlastfirst.y*qListiiplus1y[num-1]*Ly*qListiiplus1y[num-1]*Ly))/ (minDistlastfirstNorm*minDistlastfirstNorm*minDistlastfirstNorm) );
    
  //printf("d2Pdgamma2=%lf\n", d2Pdgamma2);
  //printf("total=%lf\n", ((2*KA*dAdgamma*dAdgamma) + (2*KP*(PerimeterCell-h_APpref.data[cellID].y)*d2Pdgamma2) + (2*KP*dPdgamma*dPdgamma)) );

  return ((2*h_m.data[cellID].x*dAdgamma*dAdgamma) + 2*h_m.data[cellID].x*(AreaCell-h_APpref.data[cellID].x)*d2Adgamma2+ (2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)*d2Pdgamma2) + (2*h_m.data[cellID].y*dPdgamma*dPdgamma));
    
}

Dscalar2 VertexQuadraticEnergy::PureSheard2Edgammadrialpha(int vertexi){
    
  Dscalar2 d2Edgammadrialpha;
    
  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);
    
  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
  }
    
  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }

  vector<int> cellVerticesdata(Ncells*16);
    
  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn];
    }
  }
    
  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }
    
  int vertexIDi=vertexi;
  int CellsofVertexi[3];
    
  //WHAT ARE THREE CELLS THAT SURROUNDS THE VERTEX i
  for(int ii=0; ii<3; ++ii){
    CellsofVertexi[ii]=VertexCellNeighborsList[3*vertexIDi+ii];
  }

  Dscalar d2EdgammadriXvalue = 0.;
  Dscalar d2EdgammadriYvalue = 0;
  for(int kk=0; kk<3; ++kk){
    int cellID=CellsofVertexi[kk];
    num = h_cvn.data[cellID];
    //printf("num=%i\n",num);
    int cellVerticesInfo[num];
    //for (int nn=0; nn<num; ++nn){
    //printf("cellVerticesdata[cellID*(16)+nn]=%i\n", cellVerticesdata[cellID*(16)+nn]);
    //}
    for (int nn=0; nn<num; ++nn){
      cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
      //printf("cellVerticesInfo[%i]=%i\n",  nn, cellVerticesInfo[nn]);
    }
    //CALCULATE THE PERIMETER OF THE CELL
    Dscalar PerimeterCell=0;
    Dscalar2 rij[num];
    for(int nn=0; nn<num; ++nn){
      Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
      //printf("x=%lf\n", rij[nn].x);
      //printf("y=%lf\n", rij[nn].y);
    }
    for(int nn=0; nn<num-1; ++nn){
      Dscalar xdiff=rij[nn].x-rij[nn+1].x;
      Dscalar ydiff=rij[nn].y-rij[nn+1].y;
      PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
    }
    Dscalar xdiff=rij[num-1].x-rij[0].x;
    Dscalar ydiff=rij[num-1].y-rij[0].y;
    PerimeterCell+=sqrt((xdiff*xdiff)+(ydiff*ydiff));
    //printf("PerimeterCell=%lf\n", PerimeterCell);
        
    Dscalar AreaCell=0;
    for(int nn=1; nn<num-1; ++nn){
      AreaCell+=(1./2.)*rij[nn].x*(rij[nn+1].y-rij[nn-1].y);
    }
    AreaCell+=(1./2.)*rij[0].x*(rij[1].y-rij[num-1].y);
    AreaCell+=(1./2.)*rij[num-1].x*(rij[0].y-rij[num-2].y);
    //printf("AreaCell=%lf\n", AreaCell);

    Dscalar2 poslist[num];
    for(int nn=0; nn<num; ++nn){
      poslist[nn]=h_vp.data[cellVerticesInfo[nn]];
      //printf("x=%lf\n", poslist[nn].x);
      // printf("y=%lf\n", poslist[nn].y);
    }
        
    Dscalar x11,x12,x21,x22;
    Box->getBoxDims(x11,x12,x21,x22);
    Dscalar Lx=x11;
    Dscalar Ly=x22;
    int qListiiplus1y[num];
    for(int nn=0; nn<num; ++nn){
      qListiiplus1y[nn]=0.;
    }
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].y < poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
    qListiiplus1y[nn]=1;
      }
    }
    if( (poslist[num-1].y<poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
      qListiiplus1y[num-1]=1;
    }
        
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].y>poslist[nn+1].y) && (abs(poslist[nn].y-poslist[nn+1].y)>(Ly/2.)) ){
    qListiiplus1y[nn]=-1;
      }
    }
    if( (poslist[num-1].y>poslist[0].y) && (abs(poslist[num-1].y-poslist[0].y)>(Ly/2.)) ){
      qListiiplus1y[num-1]=-1;
    }
    //for(int nn=0; nn<num; ++nn){
    //printf("qListiiplus1y=%i\n",qListiiplus1y[nn]);
    //}

    int qListiiplus1x[num];
    for(int nn=0; nn<num; ++nn){
      qListiiplus1x[nn]=0.;
    }
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].x < poslist[nn+1].x) && (abs(poslist[nn].x-poslist[nn+1].x)>(Lx/2.)) ){
    qListiiplus1x[nn]=1;
      }
    }
    if( (poslist[num-1].x<poslist[0].x) && (abs(poslist[num-1].x-poslist[0].x)>(Lx/2.)) ){
      qListiiplus1x[num-1]=1;
    }
        
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn].x>poslist[nn+1].x) && (abs(poslist[nn].x-poslist[nn+1].x)>(Lx/2.)) ){
    qListiiplus1x[nn]=-1;
      }
    }
    if( (poslist[num-1].x>poslist[0].x) && (abs(poslist[num-1].x-poslist[0].x)>(Lx/2.)) ){
      qListiiplus1x[num-1]=-1;
    }

        
    Dscalar ymin=100.;
    int Positionofy0;
    for(int nn=0; nn<num; ++nn){
      if(poslist[nn].y<ymin){
    ymin=poslist[nn].y;
    Positionofy0=nn;
      }
    }
    Dscalar2 pos0=poslist[Positionofy0];
    //printf("pos0x=%lf\n pos0y=%lf\n", pos0.x,pos0.y);

    int qListi0y[num];
    for(int nn=0; nn<num; ++nn){
      qListi0y[nn]=0.;
    }
    for(int nn=0; nn<num; ++nn){
      if( (poslist[nn].y < pos0.y) && (abs(poslist[nn].y-pos0.y)>(Ly/2.)) ){
    qListi0y[nn]=1;
      }
    }
    for(int nn=0; nn<num; ++nn){
      if( (poslist[nn].y > pos0.y) && (abs(poslist[nn].y-pos0.y)>(Ly/2.)) ){
    qListi0y[nn]=-1;
      }
    }
        
    int qListiplus10y[num];
    for(int nn=0; nn<num; ++nn){
      qListiplus10y[nn]=0.;
    }
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn+1].y < pos0.y) && (abs(poslist[nn+1].y-pos0.y)>(Ly/2.)) ){
    qListiplus10y[nn]=1;
      }
    }
    if( (poslist[0].y < pos0.y) && (abs(poslist[0].y-pos0.y)>(Ly/2.)) ){
      qListiplus10y[0]=1;
    }
    for(int nn=0; nn<num-1; ++nn){
      if( (poslist[nn+1].y > pos0.y) && (abs(poslist[nn+1].y-pos0.y)>(Ly/2.)) ){
    qListiplus10y[nn]=-1;
      }
    }
    if( (poslist[0].y > pos0.y) && (abs(poslist[0].y-pos0.y)>(Ly/2.)) ){
      qListiplus10y[0]=-1;
    }
    //GET vcur, vnext and vlast
    int vcuri=vertexIDi;
    int vcurLabeli=0;
    for (int nn=0; nn<num; ++nn){
      if(cellVerticesInfo[nn]==vcuri){
    vcurLabeli=nn;
      }
    }
    //printf("vcurLabeli=%i\n",  vcurLabeli);
        
    int vnexti;
    int vlasti;
    if (vcurLabeli + 1 > num-1){
      vnexti = cellVerticesInfo[0];
    }else{
      vnexti = cellVerticesInfo[vcurLabeli + 1];
    }
    // printf("vnexti=%i\n", vnexti);
    if (vcurLabeli == 0){
      vlasti = cellVerticesInfo[num-1];
    }else{
      vlasti = cellVerticesInfo[vcurLabeli - 1];
    }
    //printf("vlasti=%i\n", vlasti);
    int qCurNexty = 0;
    if( (h_vp.data[vcuri].y < h_vp.data[vnexti].y) && ( abs(h_vp.data[vcuri].y - h_vp.data[vnexti].y) > (Ly/2.)) ){
      qCurNexty=1;
    }
    if( (h_vp.data[vcuri].y > h_vp.data[vnexti].y) && ( abs(h_vp.data[vcuri].y - h_vp.data[vnexti].y) > (Ly/2.) ) ){
      qCurNexty=-1;
    }

    int qCurNextx = 0;
    if( (h_vp.data[vcuri].x < h_vp.data[vnexti].x) && ( abs(h_vp.data[vcuri].x - h_vp.data[vnexti].x) > (Lx/2.)) ){
      qCurNextx=1;
    }
    if( (h_vp.data[vcuri].x > h_vp.data[vnexti].x) && ( abs(h_vp.data[vcuri].x - h_vp.data[vnexti].x) > (Lx/2.) ) ){
      qCurNextx=-1;
    }
        
    int qLastCury=0;
    if( (h_vp.data[vlasti].y < h_vp.data[vcuri].y) && ( abs(h_vp.data[vlasti].y - h_vp.data[vcuri].y) > (Ly/2.)) ){
      qLastCury=1;
    }
    if( (h_vp.data[vlasti].y > h_vp.data[vcuri].y) && ( abs(h_vp.data[vlasti].y - h_vp.data[vcuri].y) > (Ly/2.) ) ){
      qLastCury=-1;
    }
        
    int qLastCurx=0;
    if( (h_vp.data[vlasti].x < h_vp.data[vcuri].x) && ( abs(h_vp.data[vlasti].x - h_vp.data[vcuri].x) > (Lx/2.)) ){
      qLastCurx=1;
    }
    if( (h_vp.data[vlasti].x > h_vp.data[vcuri].x) && ( abs(h_vp.data[vlasti].x - h_vp.data[vcuri].x) > (Lx/2.) ) ){
      qLastCurx=-1;
    }
        
    //printf("qCurNexty=%i\n",qCurNexty);
    //printf("qLastCury=%i\n",qLastCury);

    int qNextLast=0;
    if( (h_vp.data[vnexti].y < h_vp.data[vlasti].y) && ( abs(h_vp.data[vnexti].y - h_vp.data[vlasti].y) > (Ly/2.)) ){
      qNextLast=1;
    }
    if( (h_vp.data[vnexti].y > h_vp.data[vlasti].y) && ( abs(h_vp.data[vnexti].y - h_vp.data[vlasti].y) > (Ly/2.) ) ){
      qNextLast=-1;
    }
        
    int qNextLastx=0;
    if( (h_vp.data[vnexti].x < h_vp.data[vlasti].x) && ( abs(h_vp.data[vnexti].x - h_vp.data[vlasti].x) > (Lx/2.)) ){
      qNextLastx=1;
    }
    if( (h_vp.data[vnexti].x > h_vp.data[vlasti].x) && ( abs(h_vp.data[vnexti].x - h_vp.data[vlasti].x) > (Lx/2.) ) ){
      qNextLastx=-1;
    }

    Dscalar dAdgamma=0.;
    for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDisti0;
      Dscalar2 minDistiplus10;
      Dscalar2 minDistiiplus1;
      Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
      Box->minDist(poslist[nn], pos0, minDisti0);
      Box->minDist(poslist[nn+1], pos0, minDistiplus10);
      dAdgamma+=(1./2.)*(qListiiplus1x[nn]*Lx*(minDisti0.y+minDistiplus10.y)+minDistiiplus1.x*(-qListi0y[nn]*Ly-qListiplus10y[nn]*Ly));
    }
    Dscalar2 minDistlast0;
    Dscalar2 minDistfirst0;
    Box->minDist(poslist[num-1], pos0, minDistlast0);
    Box->minDist(poslist[0], pos0, minDistfirst0);
    Dscalar2 minDistlastfirst;
    Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
    dAdgamma+=(1./2.)*(qListiiplus1x[num-1]*Lx*(minDistlast0.y+minDistfirst0.y)+minDistlastfirst.x*(-qListi0y[num-1]*Ly-qListiplus10y[num-1]*Ly));
    //printf("dAdgamma=%lf\n",dAdgamma);
        
    Dscalar dPdgamma=0.;
    for(int nn=0; nn<num-1; ++nn){
      Dscalar2 minDistiiplus1;
      Box->minDist(poslist[nn], poslist[nn+1], minDistiiplus1);
      dPdgamma+=(qListiiplus1x[nn]*Lx*(minDistiiplus1.x)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y))+(-qListiiplus1y[nn]*Ly*(minDistiiplus1.y)/sqrt(minDistiiplus1.x*minDistiiplus1.x+minDistiiplus1.y*minDistiiplus1.y));
    }
    minDistlastfirst;
    Box->minDist(poslist[num-1], poslist[0], minDistlastfirst);
    dPdgamma+=(qListiiplus1x[num-1]*Lx*(minDistlastfirst.x)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y))+(-qListiiplus1y[num-1]*Ly*(minDistlastfirst.y)/sqrt(minDistlastfirst.x*minDistlastfirst.x+minDistlastfirst.y*minDistlastfirst.y));
    //printf("dPdgamma=%lf\n",dPdgamma);

    Dscalar2 minDistvcurivnexti;
    Box->minDist(h_vp.data[vcuri], h_vp.data[vnexti], minDistvcurivnexti);
    Dscalar minDistvcurivnextiNorm= sqrt(minDistvcurivnexti.x*minDistvcurivnexti.x+minDistvcurivnexti.y*minDistvcurivnexti.y);
    Dscalar minDistvcurivnextiNormCube=minDistvcurivnextiNorm*minDistvcurivnextiNorm*minDistvcurivnextiNorm;
    Dscalar2 minDistvlastivcuri;
    Box->minDist(h_vp.data[vlasti], h_vp.data[vcuri], minDistvlastivcuri);
    Dscalar minDistvlastivcuriNorm= sqrt(minDistvlastivcuri.x*minDistvlastivcuri.x+minDistvlastivcuri.y*minDistvlastivcuri.y);
    Dscalar minDistvlastivcuriNormCube= minDistvlastivcuriNorm*minDistvlastivcuriNorm*minDistvlastivcuriNorm;
        
    Dscalar2 minDistvnextivlasti;
    Box->minDist(h_vp.data[vnexti], h_vp.data[vlasti], minDistvnextivlasti);
        
        d2EdgammadriXvalue+=h_m.data[cellID].x*dAdgamma*minDistvnextivlasti.y
      +2*h_m.data[cellID].y*dPdgamma*(minDistvcurivnexti.x/minDistvcurivnextiNorm - minDistvlastivcuri.x/minDistvlastivcuriNorm)
      +2*h_m.data[cellID].x*(AreaCell-h_APpref.data[cellID].x)*(-qNextLast*Ly/2.)
      +2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)
      *( ((qCurNextx*Lx / minDistvcurivnextiNorm) - (qLastCurx*Lx / minDistvlastivcuriNorm))
         + (( (-qCurNextx*Lx*minDistvcurivnexti.x*minDistvcurivnexti.x)+ (qCurNexty*Ly*minDistvcurivnexti.x*minDistvcurivnexti.y))/minDistvcurivnextiNormCube)
         - (((-qLastCurx*Lx*minDistvlastivcuri.x*minDistvlastivcuri.x)+ (qLastCury*Ly*minDistvlastivcuri.x*minDistvlastivcuri.y))/minDistvlastivcuriNormCube) );
        //printf("d2EdgammadriXvalue=%lf\n",d2EdgammadriXvalue);
        
        d2EdgammadriYvalue+=h_m.data[cellID].x*dAdgamma*(-minDistvnextivlasti.x)
      +2*h_m.data[cellID].y*dPdgamma*(minDistvcurivnexti.y/minDistvcurivnextiNorm - minDistvlastivcuri.y/minDistvlastivcuriNorm)
      +h_m.data[cellID].x*(AreaCell-h_APpref.data[cellID].x)*(-qNextLastx*Lx)
      +2*h_m.data[cellID].y*(PerimeterCell-h_APpref.data[cellID].y)
      *( ((-qCurNexty*Ly / minDistvcurivnextiNorm) + (qLastCury*Ly / minDistvlastivcuriNorm))
         + (( (qCurNexty*Ly*minDistvcurivnexti.y*minDistvcurivnexti.y)+ (-qCurNextx*Lx*minDistvcurivnexti.x*minDistvcurivnexti.y))/minDistvcurivnextiNormCube)
         - (((qLastCury*Ly*minDistvlastivcuri.y*minDistvlastivcuri.y)+ (-qLastCurx*Lx*minDistvlastivcuri.x*minDistvlastivcuri.y))/minDistvlastivcuriNormCube) );
        //printf("d2EdgammadriYvalue=%lf\n",d2EdgammadriYvalue);
        
  }

  d2Edgammadrialpha.x=d2EdgammadriXvalue;
  d2Edgammadrialpha.y=d2EdgammadriYvalue;

  return d2Edgammadrialpha;  
    
}


Dscalar2 VertexQuadraticEnergy::psi6(int cellID){

  Dscalar2 psi6;
  ArrayHandle<Dscalar2> h_vp(vertexPositions,access_location::host,access_mode::read);
  ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
  ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
  ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
  ArrayHandle<Dscalar2> h_m(Moduli,access_location::host,access_mode::read);

  vector<Dscalar> posdata(2*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    Dscalar px = h_vp.data[i].x;
    Dscalar py = h_vp.data[i].y;
    posdata[(2*i)] = px;
    posdata[(2*i)+1] = py;
    //printf("px=%lf\n", posdata[(2*i)]);                                                                            
    //printf("py=%lf\n", posdata[(2*i)+1]);                                                                          
  }

  int nmax,num;
  nmax=h_cvn.data[0];
  for(int ii=1; ii<Ncells; ++ii){
    num = h_cvn.data[ii];
    if (num > h_cvn.data[ii-1]){
      nmax = num;
    }
  }

  vector<int> cellVerticesdata(Ncells*16);
  for(int ii=0; ii<Ncells; ++ii){
    for(int nn=0; nn<16; ++nn){
      cellVerticesdata[(ii*(16))+nn]=h_cv.data[(ii*(16))+nn];
    }
  }

  vector<int> VertexCellNeighborsList(3*Nvertices);
  for (int i = 0; i < Nvertices; ++i){
    VertexCellNeighborsList[3*i] = h_vcn.data[3*i];
    VertexCellNeighborsList[3*i+1] = h_vcn.data[3*i+1];
    VertexCellNeighborsList[3*i+2] = h_vcn.data[3*i+2];
  }

  num = h_cvn.data[cellID];
  int cellVerticesInfo[num];
  for (int nn=0; nn<num; ++nn){
    cellVerticesInfo[nn]=cellVerticesdata[cellID*(16)+nn];
  }

  Dscalar2 rij[num];
  for(int nn=0; nn<num; ++nn){
    Box->minDist(h_vp.data[cellVerticesInfo[0]], h_vp.data[cellVerticesInfo[nn]], rij[nn]);
  }

  Dscalar2 centerofmass;
  centerofmass.x=0.;
  centerofmass.y=0.;
  for(int nn=0; nn<num; ++nn){
    centerofmass.x+=rij[nn].x;
    centerofmass.y+=rij[nn].y;  
  }
  centerofmass.x=centerofmass.x/num;
  centerofmass.y=centerofmass.y/num;

  //printf("cm.x=%lf\n",centerofmass.x);
  //printf("cm.y=%lf\n",centerofmass.y);

  Dscalar angles[num];
  for(int nn=0; nn<num; ++nn){
    angles[nn]=0.;
  }

  for(int nn=0; nn<num-1; ++nn){
    Dscalar aa=sqrt( (rij[nn].x-rij[nn+1].x)*(rij[nn].x-rij[nn+1].x) + (rij[nn].y-rij[nn+1].y)*(rij[nn].y-rij[nn+1].y) );
    Dscalar bb=sqrt( (rij[nn].x-centerofmass.x)*(rij[nn].x-centerofmass.x) + (rij[nn].y-centerofmass.y)*(rij[nn].y-centerofmass.y) );
    Dscalar cc=sqrt( (rij[nn+1].x-centerofmass.x)*(rij[nn+1].x-centerofmass.x) + (rij[nn+1].y-centerofmass.y)*(rij[nn+1].y-centerofmass.y) );
    angles[nn]=acos(((bb*bb)+(cc*cc)-(aa*aa))/(2*bb*cc));
    //printf("angles=%lf\n",angles[nn]);
  }

  Dscalar aa=sqrt( (rij[num-1].x-rij[0].x)*(rij[num-1].x-rij[0].x) + (rij[num-1].y-rij[0].y)*(rij[num-1].y-rij[0].y) );
  Dscalar bb=sqrt( (rij[num-1].x-centerofmass.x)*(rij[num-1].x-centerofmass.x) + (rij[num-1].y-centerofmass.y)*(rij[num-1].y-centerofmass.y) );
  Dscalar cc=sqrt( (rij[0].x-centerofmass.x)*(rij[0].x-centerofmass.x) + (rij[0].y-centerofmass.y)*(rij[0].y-centerofmass.y) );
  angles[num-1]=acos(((bb*bb)+(cc*cc)-(aa*aa))/(2*bb*cc));
  //printf("angles=%lf\n",angles[num-1]);

  Dscalar RePart=0.;
  Dscalar ImPart=0.;  

  for(int nn=0; nn<num; ++nn){
    RePart+=cos(6*angles[nn]);
    ImPart+=sin(6*angles[nn]);
  }

  RePart=RePart/num;
  ImPart=ImPart/num;

  psi6.x=RePart;
  psi6.y=ImPart;
  //printf("RePart=%lf\n",RePart);
  //printf("ImPart=%lf\n",ImPart);
  //Dscalar psi6Square;
  //psi6Square=(RePart*RePart)+(ImPart*ImPart);

  return psi6;

}
