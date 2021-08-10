#define ENABLE_CUDA

#include "voronoiQuadraticEnergyWithConc.h"
#include "voronoiQuadraticEnergyWithConc.cuh"
#include <math.h>
/*! \file voronoiQuadraticEnergyWithConc.cpp */



/*!
This function definesa matrix, \gamma_{i,j}, describing the imposed tension between cell types i and
j. This function both sets that matrix and sets the flag telling the computeForces function to call
the more general tension force computations...
\pre the vector has n^2 elements, where n is the number of types, and the values of type in the system
must be of the form 0, 1, 2, ...n. The input vector should be laid out as:
gammas[0] = g_{0,0}  (an irrelevant value that is never called)
gammas[1] = g_{0,1}
gammas[n] = g_{0,n}
gammas[n+1] = g_{1,0} (physically, this better be the same as g_{0,1})
gammas[n+2] = g_{1,1} (again, never used)
...
gammas[n^2-1] = g_{n,n}
*/
void VoronoiQuadraticEnergyWithConc::setSurfaceTension(vector<Dscalar> gammas)
    {
    simpleTension = false;
    //set the tension matrix to the right size, and the indexer
    tensionMatrix.resize(gammas.size());
    int n = sqrt(gammas.size());
    cellTypeIndexer = Index2D(n);

    ArrayHandle<Dscalar> tensions(tensionMatrix,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < gammas.size(); ++ii)
        {   
        int typeI = ii/n;
        int typeJ = ii - typeI*n;
        tensions.data[cellTypeIndexer(typeJ,typeI)] = gammas[ii];
        };
    };


/*!
\param n number of cells to initialize
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNcellsG) is called, as is setCellPreferenceUniform(1.0,4.0)
*/
VoronoiQuadraticEnergyWithConc::VoronoiQuadraticEnergyWithConc(int n, Dscalar Lenx, Dscalar Leny, bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box... \n",n);
    Reproducible = reprod;
    initializeVoronoiQuadraticEnergy(n);
    setCellPreferencesUniform(1.0,4.0);
    totalsteps=0;

    int gridsteps = ceil(sqrt(n*10));


    Dscalar Lx = 10;
    Dscalar Ly = 50;

    //Get the box dimesions set
    boxdim.resize(2);
    ArrayHandle<Dscalar> boxdimensions(boxdim,access_location::host,access_mode::overwrite);
    boxdimensions.data[0]=Lx;
    boxdimensions.data[1]=Ly;

    Dscalar xbias = boxdimensions.data[0]/(float)sqrt(n);
    Dscalar ybias = boxdimensions.data[1]/(float)sqrt(n);
  
    //Dscalar ybias = (float)50/(float)sqrt(n);
    //Dscalar xbias = (float)10/(float)sqrt(n);

    int ygrid = ceil(ybias*gridsteps);
    int xgrid = ceil(xbias*gridsteps);

    //Initialize Conc
    Numcells=n;
    concentration.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);

    //Initialize Conc ceneter locations
    conclocation.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::overwrite);

    //Initialize Current Cell Concentration (Won't check until first update)
    cellcon.resize(Numcells);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);

    conccell.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    celladjust.resize(n);

    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    pastcellpos.resize(Numcells);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);

    celltenforces.resize(Numcells);
    ArrayHandle<Dscalar2> h_celltenforces(celltenforces,access_location::host,access_mode::overwrite);

    gradalign.resize(Numcells);
    ArrayHandle<Dscalar2> h_align(gradalign,access_location::host,access_mode::overwrite);

    cellvelocity.resize(Numcells);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);

    linetension.resize(Numcells*60);
    ArrayHandle<Dscalar> h_tension(linetension,access_location::host,access_mode::overwrite);

    for(int i=0; i<xgrid*ygrid;i++){
        concel.data[i]=-1;
        if(i<n){celcon.data[i].x=0;
            celcon.data[i].y=0;
            h_past.data[i].x=h_p.data[i].x;
            h_past.data[i].y=h_p.data[i].y;
            h_velocity.data[i].x=0;
            h_velocity.data[i].y=0;
            }

        //Mid Stripe
        //if((i>-1+gridsteps*ceil(gridsteps/2))&&(i<gridsteps*ceil(1+gridsteps/2)))
        //if((i>xgrid*floor(ygrid/2-1)-1)&&(i<xgrid*ceil(ygrid/2)))
        //	{conc.data[i+(ygrid*xgrid)]=100.0;}
        //else{conc.data[i] = 0.0;}


        //Top Stripe
        for(int j=0; j < ygrid;j++){
        	if((i>xgrid*j-1)&&(i<xgrid*(j+1)))
        		{conc.data[i+(ygrid*xgrid)]=100.0*(((float)j/(float)ygrid));
        		 conc.data[i]=100.0*(((float)j/(float)ygrid));
        		}

        	}

        //if(i>ceil(ygrid-1)*xgrid-1){conc.data[i+(ygrid*xgrid)]=100.0;}
        //else{conc.data[i] = 0.0;}

        cloc.data[i].x=xbias*(sqrt(n)/(2*xgrid))*(2*(i%xgrid)+1);
        cloc.data[i].y=ybias*(sqrt(n)/(2*ygrid))*(2*floor(i/xgrid)+1);


        }   
	    
    };

/*!
\param n number of cells to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNG) is called
*/
VoronoiQuadraticEnergyWithConc::VoronoiQuadraticEnergyWithConc(int n,Dscalar A0, Dscalar P0, Dscalar Lenx, Dscalar Leny,bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box...\n ",n);
    Reproducible = reprod;
    initializeVoronoiQuadraticEnergy(n);
    setCellPreferencesUniform(A0,P0);
    setv0Dr(0.05,1.0);
    totalsteps=0;
    int gridsteps = ceil(sqrt(n*10));

    //Dscalar Lx = 10;
    //Dscalar Ly = 50;

    //Get the box dimesions set
    boxdim.resize(2);
    ArrayHandle<Dscalar> boxdimensions(boxdim,access_location::host,access_mode::overwrite);
    boxdimensions.data[0]=Lenx;
    boxdimensions.data[1]=Leny;

    Dscalar xbias = boxdimensions.data[0]/(float)sqrt(n);
    Dscalar ybias = boxdimensions.data[1]/(float)sqrt(n);    

    //Dscalar ybias = (float)50/(float)sqrt(n);
    //Dscalar xbias = (float)10/(float)sqrt(n);

    int ygrid = ceil(ybias*gridsteps);
    int xgrid = ceil(xbias*gridsteps);

    //Initialize Conc
    Numcells=n;
    concentration.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);

    //Initialize Conc ceneter locations
    conclocation.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::overwrite);

    //Initialize Current Cell Concentration (Won't check until first update)
    cellcon.resize(Numcells);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);

    conccell.resize(xgrid*ygrid*2);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    celladjust.resize(n);

    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    pastcellpos.resize(Numcells);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);

    celltenforces.resize(Numcells);
    ArrayHandle<Dscalar2> h_celltenforces(celltenforces,access_location::host,access_mode::overwrite);

    gradalign.resize(Numcells);
    ArrayHandle<Dscalar2> h_align(gradalign,access_location::host,access_mode::overwrite);

    cellvelocity.resize(Numcells);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);

    linetension.resize(Numcells*60);
    ArrayHandle<Dscalar> h_tension(linetension,access_location::host,access_mode::overwrite);

    for(int i=0; i<xgrid*ygrid;i++){
        concel.data[i]=-1;
        if(i<n){celcon.data[i].x=0;
            celcon.data[i].y=0;
            h_past.data[i].x=h_p.data[i].x;
            h_past.data[i].y=h_p.data[i].y;
            h_velocity.data[i].x=0;
            h_velocity.data[i].y=0;
            }

        //MidStripe
        // if((i>xgrid*floor(ygrid/2-1)-1)&&(i<xgrid*ceil(ygrid/2)))
        // 	{
        // 		conc.data[i+(ygrid*xgrid)]=100.0;
        // 	}
        // else{conc.data[i] = 0.0;}


        //Top Stripe
        //AutoComplete
        for(int j=0; j < ygrid;j++){
        	//if((i>ceil(ygrid-(j+1))*xgrid-1)&&(i<xgrid*ceil(ygrid-j)))
        	if((i>xgrid*j-1)&&(i<xgrid*(j+1)))
        		{conc.data[i+(ygrid*xgrid)]=100.0*(((float)j/(float)ygrid));
        		 conc.data[i]=100.0*(((float)j/(float)ygrid));
        		}

        	}

	   //  if(i<xgrid)
	   //  	{conc.data[i]=0.0;
			 // conc.data[i+(xgrid*ygrid)]=0.0;
	   //  	}
	   //  if((i>xgrid*(ygrid-1)-1)&&(i<xgrid*(ygrid)))
	   //   	{conc.data[i+(xgrid*ygrid)]=100.0;
	   //   	 conc.data[i]=100.0;
	   //   	}		        	
        //if(i>ceil(ygrid-1)*xgrid-1){conc.data[i+(ygrid*xgrid)]=100.0;}
        //else{conc.data[i] = 0.0;}

	    cloc.data[i].x=xbias*(sqrt(n)/(2*xgrid))*(2*(i%xgrid)+1);
        cloc.data[i].y=ybias*(sqrt(n)/(2*ygrid))*(2*floor(i/xgrid)+1);


        }   

    };

void VoronoiQuadraticEnergyWithConc::setGradvariables(Dscalar deltac, Dscalar deltat, Dscalar degradTau, int CIL, Dscalar cellTau)
    {
    //Set the variables needed for the gradient
    gradientvar.resize(5);
    ArrayHandle<Dscalar> gradvars(gradientvar,access_location::host,access_mode::overwrite);
    gradvars.data[0]=deltac;
    gradvars.data[1]=deltat;
    gradvars.data[2]=degradTau;
    gradvars.data[3]=CIL;
    gradvars.data[4]=cellTau;

    };


void VoronoiQuadraticEnergyWithConc::setCluster(int nclust)
    {
   //Define the cluster
    setCellTypeUniform(0);    
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_pref(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar> boxdimensions(boxdim,access_location::host,access_mode::read);
    

    Dscalar xbias = boxdimensions.data[0]/(float)sqrt(Numcells);
    Dscalar ybias = boxdimensions.data[1]/(float)sqrt(Numcells);    

    //Dscalar ybias = (float)50/(float)sqrt(Numcells);
    //Dscalar xbias = (float)10/(float)sqrt(Numcells);

    int Nclustermax=nclust;

    //Initalize cluster near the low end of the gradient
    int clustercenter;
    int iniatlize = 0;	  
	int gridsteps = ceil(sqrt(Numcells*10));
	int ygrid = ceil(ybias*gridsteps);
	int xgrid = ceil(xbias*gridsteps);

    double distancemin=2*sqrt(nclust);
    double distancemax=2*sqrt(nclust)+1;

    while (iniatlize==0)
    {
        for(int i=0; i<Numcells; i++)
        {
            if((h_p.data[i].y<distancemax) && (h_p.data[i].y>distancemin))
                {
                    iniatlize=1;
                    clustercenter=i;
                    break;
                }
        }
        distancemax+=1.0;
    }

    vector<int> clusterarray(Nclustermax);    
    for(int i=0; i<Nclustermax; i++)
    {
        clusterarray[i]=-1;
    }
    
    int Ncluster=1;
    clusterarray[Ncluster-1]=clustercenter;

    while(Ncluster < Nclustermax){
        for(int j=0; j<Ncluster; j++)
        {
            int neigh = h_nn.data[clusterarray[j]];
            vector<int> ns(neigh);
            if(Ncluster >= Nclustermax){break;}
            for (int nn = 0; nn < neigh; ++nn)
                {
                    ns[nn]=h_n.data[n_idx(nn,clusterarray[j])];
                    int checkin=0;
                    if(Ncluster >= Nclustermax){break;}
                    for(int k=0; k<Ncluster; k++)
                        {
                            if(ns[nn]==clusterarray[k])
                                {checkin=1;}
                        } 
                    if(checkin==0)
                        {
                            clusterarray[Ncluster]=ns[nn];
                            Ncluster+=1;
                        }
                } 
        }
    }

    for(int i=0; i < Ncluster; i++)
        {
            h_ct.data[clusterarray[i]]=1;        
        }

   	//Reset Gradient Iniaitlization 
    for(int i=0; i<xgrid*ygrid;i++){
    for(int j=0; j < ygrid;j++){
    	//if((i>ceil(ygrid-(j+1))*xgrid-1)&&(i<xgrid*ceil(ygrid-j)))
    	if((i>xgrid*j-1)&&(i<xgrid*(j+1)))
    		{conc.data[i+(ygrid*xgrid)]=100.0*(((float)j/(float)ygrid));
    		 conc.data[i]=100.0*(((float)j/(float)ygrid));
    		}

    	}
    }

    printf("Finished with cluster initialization\n");
    };



/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VoronoiQuadraticEnergyWithConc::computeForces()
{
    //Update Concentration
    int gridsteps = ceil(sqrt(Numcells*10)); // There are roughly 10 gridpoints per cell
    totalsteps += 1;
    double up,down,left,right;

    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(cellVelocities,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> cellpos(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar> boxdimensions(boxdim,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> gradvars(gradientvar,access_location::host,access_mode::read);

    Dscalar xbias = boxdimensions.data[0]/(float)sqrt(Numcells);
    Dscalar ybias = boxdimensions.data[1]/(float)sqrt(Numcells); 

    double vx=0.0;
    double vy=0.0;
    double dt=gradvars.data[1];
    double dx = 0.1; // There are 10 gridpoints per cell of length 1

    //Define by Peclet
    double Deff = 1;
    double aeff = 0;
    double r =(dt/(dx*dx))*Deff;
    double r2=(dt/dx)*aeff;
	
	int ygrid = ceil(ybias*gridsteps);
	int xgrid = ceil(xbias*gridsteps);
    int indent = 0;
    int degradationswitch = 0;

    if(gradvars.data[2]!=0){degradationswitch=1;}
    if(totalsteps%2==0){indent=xgrid*ygrid;}

    for(int y=0; y<xgrid*ygrid;y++)
    {
        //Check Up
        if(y>xgrid*ygrid-xgrid)
            {up=conc.data[(xgrid-(xgrid*ygrid-y))+indent];}
        else{up=conc.data[(y+xgrid)+indent];}
        //Check Down
        if(y<xgrid)
            {down=conc.data[xgrid*ygrid-xgrid+y+indent];}
        else{down=conc.data[(y-xgrid)+indent];}	
        //Check Left
        if(y%xgrid==0)
            {left=conc.data[y+xgrid-1+indent];}
        else{left=conc.data[y-1+indent];}
        //Check Right
        if((y+1)%xgrid==0)
            {right=conc.data[y-xgrid+1+indent];}
        else{right=conc.data[y+1+indent];}

    	if(concel.data[y]==-1)
	    	{
	        vx=0.0;
	        vy=0.0;
	    	}
   		else
   			{
	        vx=h_velocity.data[int(concel.data[y])].x; 
	        vy=h_velocity.data[int(concel.data[y])].y; 
	    	}


        if(degradationswitch==1){conc.data[y+(xgrid*ygrid-indent)]=conc.data[y+indent]+r*(-4*conc.data[y+indent]+up+down+left+right)+0.5*r2*(vx*(right-left)+vy*(up-down))-(dt/gradvars.data[2])*conc.data[y+indent];}
		//Advection and Degradation
   		else{conc.data[y+(xgrid*ygrid-indent)]=conc.data[y+indent]+r*(-4*conc.data[y+indent]+up+down+left+right)+0.5*r2*(vx*(right-left)+vy*(up-down));}
    }

    //Fix Bounds
    for(int y=0; y<xgrid*ygrid;y++){
    //Top Stripe
    //if(y>ceil(ygrid-1)*xgrid-1){conc.data[y+(ygrid*xgrid)]=100.0;}

    if(y<xgrid)
    	{conc.data[y]=0.0;
		 conc.data[y+(xgrid*ygrid)]=0.0;
    	}
    if((y>xgrid*(ygrid-1)-1)&&(y<xgrid*(ygrid)))
     	{conc.data[y+(xgrid*ygrid)]=100.0;
     	 conc.data[y]=100.0;
     	}	    
    }


    //
    //Examine Cell Concentration    
    ArrayHandle<Dscalar2> vertpos(voroCur,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);   
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite); 

    for(int k=0; k<xgrid*ygrid;k++){
        concel.data[k]=-1;
    }
    //Set up saving for cell verticies
    int sums=0;
    double checkx=0;
    double checky=0;
    for(int ii=0; ii<Numcells;ii++){
        //Calculate Velocity and make sure the peroidic box doesn't mess it up
        checkx=(cellpos.data[ii].x-h_past.data[ii].x);
        checky=(cellpos.data[ii].y-h_past.data[ii].y);

        if(checkx>.9*sqrt(Numcells)*xbias){checkx=checkx-sqrt(Numcells)*xbias;}
        if(checky>.9*sqrt(Numcells)*ybias){checky=checky-sqrt(Numcells)*ybias;}
        if(checkx<-.9*sqrt(Numcells)*xbias){checkx=checkx+sqrt(Numcells)*xbias;}
        if(checky<-.9*sqrt(Numcells)*ybias){checky=checky+sqrt(Numcells)*ybias;}

        h_velocity.data[ii].x=checkx/dt;
        h_velocity.data[ii].y=checky/dt;
        h_past.data[ii].x=cellpos.data[ii].x;
        h_past.data[ii].y=cellpos.data[ii].y;
        sums+= h_nn.data[ii];
        celcon.data[ii].x=0;
        celcon.data[ii].y=0;}
    absvert.resize(sums*2);
    ArrayHandle<Dscalar2> avert(absvert,access_location::host,access_mode::overwrite);

    int total=0;
    int currentnum=0;
    for(int i=0; i<Numcells;i++){
        int checkup = 0;
        int checkdown = 0;
        int checkleft = 0;
        int checkright = 0;
        
            //Find Verticies
            for(int j=0; j< h_nn.data[i];j++){
                avert.data[currentnum].x=vertpos.data[n_idx(j,i)].x+cellpos.data[i].x;
                avert.data[currentnum].y=vertpos.data[n_idx(j,i)].y+cellpos.data[i].y;

                //If any vertex is out of the bounds we will have to check the respective image polygon
                if(avert.data[currentnum].x<0){checkright=1;}
                if(avert.data[currentnum].x>sqrt(Numcells)*xbias){checkleft=1;}
                if(avert.data[currentnum].y<0){checkup=1;}
                if(avert.data[currentnum].y>sqrt(Numcells)*ybias){checkdown=1;}           

                currentnum += 1;}
        for(int k=0; k<xgrid*ygrid; k++){
            if(concel.data[k]==-1){
                if((sqrt(pow(cellpos.data[i].x-cloc.data[k].x,2)+pow(cellpos.data[i].y-cloc.data[k].y,2))<1)||((checkup==1)&&(cloc.data[k].y>sqrt(Numcells)*ybias-1))||((checkdown==1)&&(cloc.data[k].y<1))||((checkleft==1)&&(cloc.data[k].x<1))||((checkright==1)&&(cloc.data[k].x>sqrt(Numcells)*xbias-1))){
                InPolygon(i,k,currentnum,checkup,checkdown,checkleft,checkright, ybias, xbias);
                }
            }
        }

        total+=celcon.data[i].y;
        if(celcon.data[i].y==0){celcon.data[i].y=1;}
        Dscalar const currentconc = ((celcon.data[i].x))/((celcon.data[i].y))/100;
        ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
        ArrayHandle<Dscalar> h_adjust(celladjust,access_location::host,access_mode::overwrite);
        h_adjust.data[i] = currentconc;
    }


    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    computeGeometry();
    if (GPUcompute)
        {
        ComputeForceSetsGPU();
        SumForcesGPU();
        }
    else
        {
        if(Tension)
            {
            if (simpleTension)
                {
                for (int ii = 0; ii < Ncells; ++ii)
                    computeVoronoiSimpleTensionForceCPU(ii);
                }
            else
                {
                for (int ii = 0; ii < Ncells; ++ii)
                    computeVoronoiTensionForceCPU(ii);
                };
            }
        else
            {
            for (int ii = 0; ii < Ncells; ++ii)
                computeVoronoiForceCPU(ii);
            };
        };
    };

/*!
\pre The geoemtry (area and perimeter) has already been calculated
\post calculate the contribution to the net force on every particle from each of its voronoi vertices
via a cuda call
*/
void VoronoiQuadraticEnergyWithConc::ComputeForceSetsGPU()
    {
        if(Tension)
            {
            if (simpleTension)
                computeVoronoiSimpleTensionForceSetsGPU();
            else
                computeVoronoiTensionForceSetsGPU();
            }
        else
            computeVoronoiForceSetsGPU();
    };



//Checks to see if cell i contains the point in the concentration grid k or if it's peridoic images do
void VoronoiQuadraticEnergyWithConc::InPolygon(int i, int k, int currentnum, int checkup, int checkdown, int checkleft, int checkright, Dscalar ybias, Dscalar xbias)
{
ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
ArrayHandle<Dscalar2> avert(absvert,access_location::host,access_mode::read);
ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::read);
ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);
ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);
int gridsteps = ceil(sqrt(Numcells*10));
double p1x,p1y,p2x,p2y,dtheta;

//Check if a point is inside a cell
        double angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1; 
        }


//Examine Images of Cells
    if(checkup==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y+sqrt(Numcells)*ybias;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y+sqrt(Numcells)*ybias;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y+sqrt(Numcells)*ybias;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
    }
    }

    if(checkdown==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y-sqrt(Numcells)*ybias;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y-sqrt(Numcells)*ybias;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y-sqrt(Numcells)*ybias;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
    }
    }

    if(checkleft==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x-sqrt(Numcells)*xbias;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x-sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x-sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
    }
    }

    if(checkright==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x+sqrt(Numcells)*xbias;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x+sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x+sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
    }
    }


    if(checkright+checkleft+checkdown+checkup>1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells)*xbias;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells)*ybias;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells)*ybias;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells)*xbias;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells)*ybias;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
    }
    }


};


//Calculates the angle difference between two verticies of polygon and a test point inside it
double VoronoiQuadraticEnergyWithConc::VertAngle(double p1x, double p1y, double p2x, double p2y)
{
double theta1,theta2,dtheta;
    theta1 =atan2(p1y,p1x);
    theta2 =atan2(p2y,p2x);
    dtheta = theta2 - theta1;
    while (dtheta > M_PI){
    dtheta -= 2*M_PI;}
    while (dtheta < -M_PI){
    dtheta += 2*M_PI;}
}




/*!
Returns the quadratic energy functional:
E = \sum_{cells} K_A(A_i-A_i,0)^2 + K_P(P_i-P_i,0)^2 + \sum_{[i]\neq[j]} \gamma_{[i][j]}l_{ij}
*/
Dscalar VoronoiQuadraticEnergyWithConc::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();
    //first, compute the area and perimeter pieces...which are easy
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APP(AreaPeriPreferences,access_location::host,access_mode::read);
    Energy = 0.0;
    for (int nn = 0; nn  < Ncells; ++nn)
        {
        Energy += KA * (h_AP.data[nn].x-h_APP.data[nn].x)*(h_AP.data[nn].x-h_APP.data[nn].x);
        Energy += KP * (h_AP.data[nn].y-h_APP.data[nn].y)*(h_AP.data[nn].y-h_APP.data[nn].y);
        };

    //now, the potential line tension terms
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);
    //Conc 
    ArrayHandle<Dscalar> h_adjust(celladjust,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar> h_tension(linetension,access_location::host,access_mode::overwrite);

    int count = 0;
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
                if (simpleTension)
                    //Energy += dnnorm*gamma;
                	if(typeI != 0 && typeK !=0)
                		{
                            Energy += 0*dnnorm*gamma*(1-h_adjust.data[cell]);
                        }
                	else
                		{
                            Energy += dnnorm*gamma*(2-h_adjust.data[cell]);
                        }
                else
                {
                    Energy += dnnorm*h_tm.data[cellTypeIndexer(typeK,typeI)];
                }
                
                };
            vlast=vcur;
            };
        };
    return Energy;
    };


/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices
*/
void VoronoiQuadraticEnergyWithConc::computeVoronoiSimpleTensionForceSetsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);

    gpu_VoronoiSimpleTension_force_sets(
                    d_p.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_vc.data,
                    d_vln.data,
                    d_forceSets.data,
                    d_nidx.data,
                    d_ct.data,
                    KA,
                    KP,
                    gamma,
                    NeighIdxNum,n_idx,*(Box));
    };

/*!
Calculate the contributions to the net force on particle "i" from each of particle i's voronoi
vertices, using the general surface tension matrix
*/
void VoronoiQuadraticEnergyWithConc::computeVoronoiTensionForceSetsGPU()
    {
    ArrayHandle<Dscalar2> d_p(cellPositions,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int2> d_delSets(delSets,access_location::device,access_mode::read);
    ArrayHandle<int> d_delOther(delOther,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_forceSets(forceSets,access_location::device,access_mode::overwrite);
    ArrayHandle<int2> d_nidx(NeighIdxs,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<Dscalar2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<Dscalar4> d_vln(voroLastNext,access_location::device,access_mode::read);

    ArrayHandle<Dscalar> d_tm(tensionMatrix,access_location::device,access_mode::read);

    gpu_VoronoiTension_force_sets(
                    d_p.data,
                    d_AP.data,
                    d_APpref.data,
                    d_delSets.data,
                    d_delOther.data,
                    d_vc.data,
                    d_vln.data,
                    d_forceSets.data,
                    d_nidx.data,
                    d_ct.data,
                    d_tm.data,
                    cellTypeIndexer,
                    KA,
                    KP,
                    NeighIdxNum,n_idx,*(Box));
    };
/*!
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void VoronoiQuadraticEnergyWithConc::computeVoronoiSimpleTensionForceCPU(int i)
    {
    Dscalar Pthreshold = THRESHOLD;
    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_vel(cellVelocities,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tension(linetension,access_location::host,access_mode::overwrite);

    ArrayHandle<Dscalar2> h_external_forces(external_forces,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_exes(exclusions,access_location::host,access_mode::read);
  	ArrayHandle<Dscalar> h_adjust(celladjust,access_location::host,access_mode::overwrite);
	ArrayHandle<Dscalar2> h_celltenforces(celltenforces,access_location::host,access_mode::overwrite);	      	
	ArrayHandle<Dscalar2> h_align(gradalign,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar> gradvars(gradientvar,access_location::host,access_mode::read);

    //get Delaunay neighbors of the cell
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
        };

    //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
    vector<Dscalar2> voro(neigh);
    vector<Matrix2x2> dhdri(neigh);
    Matrix2x2 Id;
    Dscalar2 circumcent;
    Dscalar2 rij,rik;
    Dscalar2 nnextp,nlastp;
    Dscalar2 rjk;
    Dscalar2 pi = h_p.data[i];

   nlastp = h_p.data[ns[ns.size()-1]];
    Box->minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        int id = n_idx(nn,i);
        nnextp = h_p.data[ns[nn]];
        Box->minDist(nnextp,pi,rik);
        voro[nn] = h_v.data[id];
        rjk.x =rik.x-rij.x;
        rjk.y =rik.y-rij.y;

        Dscalar2 dbDdri,dgDdri,dDdriOD,z;
        Dscalar betaD = -dot(rik,rik)*dot(rij,rjk);
        Dscalar gammaD = dot(rij,rij)*dot(rik,rjk);
        Dscalar cp = rij.x*rjk.y - rij.y*rjk.x;
        Dscalar D = 2*cp*cp;


        z.x = betaD*rij.x+gammaD*rik.x;
        z.y = betaD*rij.y+gammaD*rik.y;

        dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
        dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

        dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
        dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

        dDdriOD.x = (-2.0*rjk.y)/cp;
        dDdriOD.y = (2.0*rjk.x)/cp;

        dhdri[nn] = Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));

        rij=rik;
        };

    Dscalar2 vlast,vnext,vother;
    vlast = voro[neigh-1];

    //start calculating forces
    Dscalar2 forceSum;
    Dscalar2 tenforceSum;
    forceSum.x=0.0;forceSum.y=0.0;
    tenforceSum.x=0.0;tenforceSum.y=0.0;

    Dscalar Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
    Dscalar Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

    Dscalar2 vcur;
    vlast = voro[neigh-1];
    for(int nn = 0; nn < neigh; ++nn)
        {
        //first, let's do the self-term, dE_i/dr_i
        vcur = voro[nn];
        vnext = voro[(nn+1)%neigh];
        int baseNeigh = ns[nn];
        int other_idx = nn - 1;
        if (other_idx < 0) other_idx += neigh;
        int otherNeigh = ns[other_idx];


        Dscalar2 dAidv,dPidv,dTidv;
        dTidv.x = 0.0;
        dTidv.y = 0.0;
        dAidv.x = 0.5*(vlast.y-vnext.y);
        dAidv.y = 0.5*(vnext.x-vlast.x);

        Dscalar2 dlast,dnext;
        dlast.x = vlast.x-vcur.x;
        dlast.y=vlast.y-vcur.y;

        Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

        dnext.x = vcur.x-vnext.x;
        dnext.y = vcur.y-vnext.y;
        Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;
        dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        //individual line tensions
        if(h_ct.data[i] != h_ct.data[baseNeigh])
            {
            dTidv.x -= dnext.x/dnnorm;
            dTidv.y -= dnext.y/dnnorm;
            };
        if(h_ct.data[i] != h_ct.data[otherNeigh])
            {
            dTidv.x += dlast.x/dlnorm;
            dTidv.y += dlast.y/dlnorm;
            };
        //
        //now let's compute the other terms...first we need to find the third voronoi
        //position that v_cur is connected to
        //
        int neigh2 = h_nn.data[baseNeigh];
        int DT_other_idx=-1;
        for (int n2 = 0; n2 < neigh2; ++n2)
            {
            int testPoint = h_n.data[n_idx(n2,baseNeigh)];
            if(testPoint == otherNeigh) DT_other_idx = h_n.data[n_idx((n2+1)%neigh2,baseNeigh)];
            };
        if(DT_other_idx == otherNeigh || DT_other_idx == baseNeigh || DT_other_idx == -1)
            {
            printf("Triangulation problem %i\n",DT_other_idx);
            throw std::exception();
            };
        Dscalar2 nl1 = h_p.data[otherNeigh];
        Dscalar2 nn1 = h_p.data[baseNeigh];
        Dscalar2 no1 = h_p.data[DT_other_idx];

        Dscalar2 r1,r2,r3;
        Box->minDist(nl1,pi,r1);
        Box->minDist(nn1,pi,r2);
        Box->minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        Dscalar Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        Dscalar Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        Dscalar Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        Dscalar Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        Dscalar2 dAkdv,dPkdv,dTkdv;
        dTkdv.x = 0.0;
        dTkdv.y = 0.0;
        dAkdv.x = 0.5*(vnext.y-vother.y);
        dAkdv.y = 0.5*(vother.x-vnext.x);

        dlast.x = vnext.x-vcur.x;
        dlast.y=vnext.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vother.x;
        dnext.y = vcur.y-vother.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPkdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPkdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        if(h_ct.data[i]!=h_ct.data[baseNeigh])
            {
            dTkdv.x +=dlast.x/dlnorm;
            dTkdv.y +=dlast.y/dlnorm;
            };
        if(h_ct.data[otherNeigh]!=h_ct.data[baseNeigh])
            {
            dTkdv.x -=dnext.x/dnnorm;
            dTkdv.y -=dnext.y/dnnorm;
            };
            
        Dscalar2 dAjdv,dPjdv,dTjdv;
        dTjdv.x = 0.0;
        dTjdv.y = 0.0;
        dAjdv.x = 0.5*(vother.y-vlast.y);
        dAjdv.y = 0.5*(vlast.x-vother.x);

        dlast.x = vother.x-vcur.x;
        dlast.y=vother.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vlast.x;
        dnext.y = vcur.y-vlast.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);

        h_tension.data[50*i+5*(nn)] = vcur.x+h_p.data[i].x;
        h_tension.data[50*i+5*(nn)+1] = vcur.y+h_p.data[i].y;
        h_tension.data[50*i+5*(nn)+2] = vnext.x+h_p.data[i].x;
        h_tension.data[50*i+5*(nn)+3] = vnext.y+h_p.data[i].y;

        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPjdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPjdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        if(h_ct.data[i]!=h_ct.data[otherNeigh])
            {
            dTjdv.x -=dnext.x/dnnorm;
            dTjdv.y -=dnext.y/dnnorm;
            };
        if(h_ct.data[otherNeigh]!=h_ct.data[baseNeigh])
            {
            dTjdv.x +=dlast.x/dlnorm;
            dTjdv.y +=dlast.y/dlnorm;
            };

        Dscalar2 dEdv;
        Dscalar tempgamma;


        //If we are coupling HIT to the gradient we will directly use the input coupling. Else we set HIT to a constant
        if(gradvars.data[3]==0.0)
            {tempgamma = (gradvars.data[0])*5*gamma*(2-h_adjust.data[i]);}
        else
            {tempgamma = 10*gamma;}

        dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x + tempgamma*dTidv.x;
        dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y + tempgamma*dTidv.y;
        dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x + tempgamma*dTkdv.x;
        dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y + tempgamma*dTkdv.y;
        dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x + tempgamma*dTjdv.x;
        dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y + tempgamma*dTjdv.y;

        Dscalar2 temp = dEdv*dhdri[nn];

        forceSum.x += temp.x;
        forceSum.y += temp.y;


        Dscalar2 tensiondEdv;
        tensiondEdv.x = tempgamma*dTidv.x;
        tensiondEdv.y = tempgamma*dTidv.y;
        tensiondEdv.x += tempgamma*dTkdv.x;
        tensiondEdv.y += tempgamma*dTkdv.y;
        tensiondEdv.x += tempgamma*dTjdv.x;
        tensiondEdv.y += tempgamma*dTjdv.y;

        Dscalar2 temp2 =tensiondEdv*dhdri[nn];
        tenforceSum.x += temp2.x;
        tenforceSum.y += temp2.y;
 
        vlast=vcur;
        };


    //Working
    h_f.data[i].x=forceSum.x;
    h_f.data[i].y=forceSum.y;

    h_celltenforces.data[i].x=tenforceSum.x;
    h_celltenforces.data[i].y=tenforceSum.y;

//Alignment Vector
		if(h_ct.data[i] != 0)
		{        
		
		int neigh = h_nn.data[i];
		Dscalar2 vlast,vnext,Alignsum;
		int currentneigh;

		Alignsum.x=0;
		Alignsum.y=0;

        vlast = voro[neigh-1];
        for (int nn = 0; nn < neigh; ++nn)
            {
            vnext=voro[nn];
            currentneigh = h_n.data[n_idx(nn,i)];

            Dscalar dx = vlast.x-vnext.x;
            Dscalar dy = vlast.y-vnext.y;
            Dscalar effectgamma = 0;
            Dscalar lij = sqrt(dx*dx+dy*dy);
			if((h_ct.data[i] != 0 && h_ct.data[currentneigh] ==0))
			    {
			        effectgamma = (gradvars.data[0])*5*gamma*(1-h_adjust.data[i]);
			    }
			else
				{
					effectgamma = 0;
				}
		

			Dscalar2 curpos,neighpos,rijdist;
			curpos = h_p.data[i];
			neighpos = h_p.data[currentneigh];

			Box->minDist(curpos,neighpos,rijdist);
			Dscalar magr = sqrt(rijdist.x*rijdist.x+rijdist.y*rijdist.y);

			Alignsum.x += lij*effectgamma*rijdist.x/magr;
			Alignsum.y += lij*effectgamma*rijdist.y/magr;

			vlast = vnext;
        	}


		h_align.data[i].x = Alignsum.x;
		h_align.data[i].y = Alignsum.y;

			}
		else
			{
				h_align.data[i].x = 0;
				h_align.data[i].y = 0;
			}

    if(particleExclusions)
        {
        if(h_exes.data[i] != 0)
            {
            h_f.data[i].x = 0.0;
            h_f.data[i].y = 0.0;
            h_external_forces.data[i].x=-forceSum.x;
            h_external_forces.data[i].y=-forceSum.y;
            };
        }
    };

/*!
\param i The particle index for which to compute the net force, assuming addition tension terms between unlike particles
\post the net force on cell i is computed
*/
void VoronoiQuadraticEnergyWithConc::computeVoronoiTensionForceCPU(int i)
    {
    Dscalar Pthreshold = THRESHOLD;
    //read in all the data we'll need
    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_f(cellForces,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(voroCur,access_location::host,access_mode::read);

    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);

    ArrayHandle<Dscalar2> h_external_forces(external_forces,access_location::host,access_mode::overwrite);
    ArrayHandle<int> h_exes(exclusions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_tm(tensionMatrix,access_location::host,access_mode::read);

    //get Delaunay neighbors of the cell
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
        };

    //compute base set of voronoi points, and the derivatives of those points w/r/t cell i's position
    vector<Dscalar2> voro(neigh);
    vector<Matrix2x2> dhdri(neigh);
    Matrix2x2 Id;
    Dscalar2 circumcent;
    Dscalar2 rij,rik;
    Dscalar2 nnextp,nlastp;
    Dscalar2 rjk;
    Dscalar2 pi = h_p.data[i];

    nlastp = h_p.data[ns[ns.size()-1]];
    Box->minDist(nlastp,pi,rij);
    for (int nn = 0; nn < neigh;++nn)
        {
        int id = n_idx(nn,i);
        nnextp = h_p.data[ns[nn]];
        Box->minDist(nnextp,pi,rik);
        voro[nn] = h_v.data[id];
        rjk.x =rik.x-rij.x;
        rjk.y =rik.y-rij.y;

        Dscalar2 dbDdri,dgDdri,dDdriOD,z;
        Dscalar betaD = -dot(rik,rik)*dot(rij,rjk);
        Dscalar gammaD = dot(rij,rij)*dot(rik,rjk);
        Dscalar cp = rij.x*rjk.y - rij.y*rjk.x;
        Dscalar D = 2*cp*cp;


        z.x = betaD*rij.x+gammaD*rik.x;
        z.y = betaD*rij.y+gammaD*rik.y;

        dbDdri.x = 2*dot(rij,rjk)*rik.x+dot(rik,rik)*rjk.x;
        dbDdri.y = 2*dot(rij,rjk)*rik.y+dot(rik,rik)*rjk.y;

        dgDdri.x = -2*dot(rik,rjk)*rij.x-dot(rij,rij)*rjk.x;
        dgDdri.y = -2*dot(rik,rjk)*rij.y-dot(rij,rij)*rjk.y;

        dDdriOD.x = (-2.0*rjk.y)/cp;
        dDdriOD.y = (2.0*rjk.x)/cp;

        dhdri[nn] = Id+1.0/D*(dyad(rij,dbDdri)+dyad(rik,dgDdri)-(betaD+gammaD)*Id-dyad(z,dDdriOD));

        rij=rik;
        };

    Dscalar2 vlast,vnext,vother;
    vlast = voro[neigh-1];

    //start calculating forces
    Dscalar2 forceSum;
    forceSum.x=0.0;forceSum.y=0.0;

    Dscalar Adiff = KA*(h_AP.data[i].x - h_APpref.data[i].x);
    Dscalar Pdiff = KP*(h_AP.data[i].y - h_APpref.data[i].y);

    Dscalar2 vcur;
    vlast = voro[neigh-1];
    for(int nn = 0; nn < neigh; ++nn)
        {
        //first, let's do the self-term, dE_i/dr_i
        vcur = voro[nn];
        vnext = voro[(nn+1)%neigh];
        int baseNeigh = ns[nn];
        int other_idx = nn - 1;
        if (other_idx < 0) other_idx += neigh;
        int otherNeigh = ns[other_idx];


        Dscalar2 dAidv,dPidv,dTidv;
        dTidv.x = 0.0;
        dTidv.y = 0.0;
        dAidv.x = 0.5*(vlast.y-vnext.y);
        dAidv.y = 0.5*(vnext.x-vlast.x);

        Dscalar2 dlast,dnext;
        dlast.x = vlast.x-vcur.x;
        dlast.y=vlast.y-vcur.y;

        Dscalar dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);

        dnext.x = vcur.x-vnext.x;
        dnext.y = vcur.y-vnext.y;

        Dscalar dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;
        dPidv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPidv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        //individual line tensions
        int typeI = h_ct.data[i];
        int typeJ = h_ct.data[otherNeigh];
        int typeK = h_ct.data[baseNeigh];
        if(typeI != typeK)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeK,typeI)];
            dTidv.x -= g*dnext.x/dnnorm;
            dTidv.y -= g*dnext.y/dnnorm;
            };
        if(typeI != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeI)];
            dTidv.x += g*dlast.x/dlnorm;
            dTidv.y += g*dlast.y/dlnorm;
            };
        //
        //now let's compute the other terms...first we need to find the third voronoi
        //position that v_cur is connected to
        //
        int neigh2 = h_nn.data[baseNeigh];
        int DT_other_idx=-1;
        for (int n2 = 0; n2 < neigh2; ++n2)
            {
            int testPoint = h_n.data[n_idx(n2,baseNeigh)];
            if(testPoint == otherNeigh) DT_other_idx = h_n.data[n_idx((n2+1)%neigh2,baseNeigh)];
            };
        if(DT_other_idx == otherNeigh || DT_other_idx == baseNeigh || DT_other_idx == -1)
            {
            printf("Triangulation problem %i\n",DT_other_idx);
            throw std::exception();
            };
        Dscalar2 nl1 = h_p.data[otherNeigh];
        Dscalar2 nn1 = h_p.data[baseNeigh];
        Dscalar2 no1 = h_p.data[DT_other_idx];

        Dscalar2 r1,r2,r3;
        Box->minDist(nl1,pi,r1);
        Box->minDist(nn1,pi,r2);
        Box->minDist(no1,pi,r3);

        Circumcenter(r1,r2,r3,vother);

        Dscalar Akdiff = KA*(h_AP.data[baseNeigh].x  - h_APpref.data[baseNeigh].x);
        Dscalar Pkdiff = KP*(h_AP.data[baseNeigh].y  - h_APpref.data[baseNeigh].y);
        Dscalar Ajdiff = KA*(h_AP.data[otherNeigh].x - h_APpref.data[otherNeigh].x);
        Dscalar Pjdiff = KP*(h_AP.data[otherNeigh].y - h_APpref.data[otherNeigh].y);

        Dscalar2 dAkdv,dPkdv,dTkdv;
        dTkdv.x = 0.0;
        dTkdv.y = 0.0;
        dAkdv.x = 0.5*(vnext.y-vother.y);
        dAkdv.y = 0.5*(vother.x-vnext.x);

        dlast.x = vnext.x-vcur.x;
        dlast.y=vnext.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vother.x;
        dnext.y = vcur.y-vother.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPkdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPkdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        if(typeI != typeK)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeK,typeI)];
            dTkdv.x += g*dlast.x/dlnorm;
            dTkdv.y += g*dlast.y/dlnorm;
            };
        if(typeK != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeK)];
            dTkdv.x -= g*dnext.x/dnnorm;
            dTkdv.y -= g*dnext.y/dnnorm;
            };
            
        Dscalar2 dAjdv,dPjdv,dTjdv;
        dTjdv.x = 0.0;
        dTjdv.y = 0.0;
        dAjdv.x = 0.5*(vother.y-vlast.y);
        dAjdv.y = 0.5*(vlast.x-vother.x);

        dlast.x = vother.x-vcur.x;
        dlast.y=vother.y-vcur.y;
        dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
        dnext.x = vcur.x-vlast.x;
        dnext.y = vcur.y-vlast.y;
        dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        if(dnnorm < Pthreshold)
            dnnorm = Pthreshold;
        if(dlnorm < Pthreshold)
            dlnorm = Pthreshold;

        dPjdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
        dPjdv.y = dlast.y/dlnorm - dnext.y/dnnorm;

        if(typeI != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeI)];
            dTjdv.x -= g*dnext.x/dnnorm;
            dTjdv.y -= g*dnext.y/dnnorm;
            };
        if(typeK != typeJ)
            {
            Dscalar g = h_tm.data[cellTypeIndexer(typeJ,typeK)];
            dTjdv.x += g*dlast.x/dlnorm;
            dTjdv.y += g*dlast.y/dlnorm;
            };

        Dscalar2 dEdv;

        dEdv.x = 2.0*Adiff*dAidv.x + 2.0*Pdiff*dPidv.x + dTidv.x;
        dEdv.y = 2.0*Adiff*dAidv.y + 2.0*Pdiff*dPidv.y + dTidv.y;
        dEdv.x += 2.0*Akdiff*dAkdv.x + 2.0*Pkdiff*dPkdv.x + dTkdv.x;
        dEdv.y += 2.0*Akdiff*dAkdv.y + 2.0*Pkdiff*dPkdv.y + dTkdv.y;
        dEdv.x += 2.0*Ajdiff*dAjdv.x + 2.0*Pjdiff*dPjdv.x + dTjdv.x;
        dEdv.y += 2.0*Ajdiff*dAjdv.y + 2.0*Pjdiff*dPjdv.y + dTjdv.y;

        Dscalar2 temp = dEdv*dhdri[nn];
        forceSum.x += temp.x;
        forceSum.y += temp.y;
        
        vlast=vcur;
        };

    h_f.data[i].x=forceSum.x;
    h_f.data[i].y=forceSum.y;

    cout << "Tensions?" << endl;

    
    if(particleExclusions)
        {
        if(h_exes.data[i] != 0)
            {
            h_f.data[i].x = 0.0;
            h_f.data[i].y = 0.0;
            h_external_forces.data[i].x=-forceSum.x;
            h_external_forces.data[i].y=-forceSum.y;
            };
        }
    };





