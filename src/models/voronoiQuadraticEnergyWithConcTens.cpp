#define ENABLE_CUDA

#include "voronoiQuadraticEnergyWithConcTens.h"
#include <math.h>
/*! \file voronoiQuadraticEnergyWithConcTens.cpp */

/*!
\param n number of cells to initialize
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNcellsG) is called, as is setCellPreferenceUniform(1.0,4.0)
*/
VoronoiQuadraticEnergyWithConcTens::VoronoiQuadraticEnergyWithConcTens(int n, bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box... \n",n);
    Reproducible = reprod;
    initializeVoronoiQuadraticEnergy(n);
    setCellPreferencesUniform(1.0,4.0);
    totalsteps=0;

    int gridsteps = ceil(sqrt(n*10));
    //Initialize Conc
    Numcells=n;
    concentration.resize(gridsteps*gridsteps*2);
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);

    //Initialize Conc ceneter locations
    conclocation.resize(gridsteps*gridsteps*2);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::overwrite);

    //Initialize Current Cell Concentration (Won't check until first update)
    cellcon.resize(gridsteps);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);

    conccell.resize(gridsteps*gridsteps);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    celladjust.resize(n);

    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    pastcellpos.resize(Numcells);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);

    cellvelocity.resize(Numcells);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);

    for(int i=0; i<gridsteps*gridsteps;i++){
        concel.data[i]=-1;
        if(i<n){celcon.data[i].x=0;
            celcon.data[i].y=0;
            h_past.data[i].x=h_p.data[i].x;
            h_past.data[i].y=h_p.data[i].y;
            h_velocity.data[i].x=0;
            h_velocity.data[i].y=0;
            }
        //if((i<gridsteps)||((i+1)%gridsteps==0)||(i%gridsteps==0)||(i>gridsteps*gridsteps-gridsteps)){conc.data[i]=0.0;}
        //else{conc.data[i]=100.0;}
       //if((i<gridsteps)||(i>gridsteps*gridsteps-gridsteps)){conc.data[i] = 100.0;}


        //Right Wall
        //if(i%gridsteps==0){conc.data[i]=100.0;}


        //Mid Stripe
        if((i>-1+gridsteps*ceil(gridsteps/2))&&(i<gridsteps*ceil(1+gridsteps/2))){conc.data[i+(gridsteps*gridsteps)]=100.0;}
        else{conc.data[i] = 0.0;}

        //Center Point
        //if(i==ceil(gridsteps/2)-1+gridsteps*ceil(gridsteps/2)){conc.data[i+(gridsteps*gridsteps)]=100.0;}
        //else{conc.data[i] = 0.0;}


        cloc.data[i].x=(sqrt(n)/(2*gridsteps))*(2*((gridsteps*gridsteps-i-1)%gridsteps)+1);
        cloc.data[i].y=(sqrt(n)/(2*gridsteps))*(2*((i)/gridsteps)+1);
           
        }
    //Define the cluster
    setCellTypeUniform(0);    
    int i = 0;
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    h_ct.data[i]=1;
    h_mot.data[i].x=0.1;
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
        h_ct.data[ns[nn]]=1;
        h_mot.data[ns[nn]].x=1.0;    
        };

    };

/*!
\param n number of cells to initialize
\param A0 set uniform preferred area for all cells
\param P0 set uniform preferred perimeter for all cells
\param reprod should the simulation be reproducible (i.e. call a RNG with a fixed seed)
\post initializeVoronoiQuadraticEnergy(n,initGPURNG) is called
*/
VoronoiQuadraticEnergyWithConcTens::VoronoiQuadraticEnergyWithConcTens(int n,Dscalar A0, Dscalar P0,bool reprod)
    {
    printf("Initializing %i cells with random positions in a square box...\n ",n);
    Reproducible = reprod;
    initializeVoronoiQuadraticEnergy(n);
    setCellPreferencesUniform(A0,P0);
    setv0Dr(0.05,1.0);
    totalsteps=0;
    int gridsteps = ceil(sqrt(n*10));

    //Initialize Conc
    Numcells=n;
    concentration.resize(gridsteps*gridsteps*2);
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);

    //Initialize Conc ceneter locations
    conclocation.resize(gridsteps*gridsteps*2);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::overwrite);

    //Initialize Current Cell Concentration (Won't check until first update)
    cellcon.resize(Numcells);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);

    conccell.resize(gridsteps*gridsteps);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    celladjust.resize(n);

    ArrayHandle<Dscalar2> h_p(cellPositions,access_location::host,access_mode::read);
    pastcellpos.resize(Numcells);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);

    cellvelocity.resize(Numcells);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);

    for(int i=0; i<gridsteps*gridsteps;i++){
        concel.data[i]=-1;
        if(i<n){celcon.data[i].x=0;
            celcon.data[i].y=0;
            h_past.data[i].x=h_p.data[i].x;
            h_past.data[i].y=h_p.data[i].y;
            h_velocity.data[i].x=0;
            h_velocity.data[i].y=0;
            }
        //if((i<gridsteps)||((i+1)%gridsteps==0)||(i%gridsteps==0)||(i>gridsteps*gridsteps-gridsteps)){conc.data[i]=0.0;}
        //else{conc.data[i]=100.0;}
        //if((i<gridsteps)){conc.data[i] = 100.0;}

        //Right Wall
        //if(i%gridsteps==0){conc.data[i]=100.0;}


        //Mid Stripe
        if((i>-1+gridsteps*ceil(gridsteps/2))&&(i<gridsteps*ceil(1+gridsteps/2))){conc.data[i+(gridsteps*gridsteps)]=100.0;}
        else{conc.data[i] = 0.0;}

        //Center Point
        //if(i==ceil(gridsteps/2)-1+gridsteps*ceil(gridsteps/2)){conc.data[i+(gridsteps*gridsteps)]=100.0;}
        //else{conc.data[i] = 0.0;}


        cloc.data[i].x=(sqrt(n)/(2*gridsteps))*(2*((gridsteps*gridsteps-i-1)%gridsteps)+1);
        cloc.data[i].y=(sqrt(n)/(2*gridsteps))*(2*((i)/gridsteps)+1);
        }

    //Define the cluster
    setCellTypeUniform(0);
    int i = 0;
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(cellNeighbors,access_location::host,access_mode::read);
    h_ct.data[i]=1;
    h_mot.data[i].x=0.1;
    int neigh = h_nn.data[i];
    vector<int> ns(neigh);
    for (int nn = 0; nn < neigh; ++nn)
        {
        ns[nn]=h_n.data[n_idx(nn,i)];
        h_ct.data[ns[nn]]=1;
        h_mot.data[ns[nn]].x=1.0;    
        };

    };

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VoronoiQuadraticEnergyWithConcTens::computeForces()
{
    //Update Concentration
    int gridsteps = ceil(sqrt(Numcells*10));
    totalsteps += 1;
    double up,down,left,right;
    int indent = 0;
    if(totalsteps%2==0){indent=gridsteps*gridsteps;}
    ArrayHandle<Dscalar> conc(concentration,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> cloc(conclocation,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_v(cellVelocities,access_location::host,access_mode::read);
    ArrayHandle<Dscalar> concel(conccell,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> cellpos(cellPositions,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> h_past(pastcellpos,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_velocity(cellvelocity,access_location::host,access_mode::overwrite);

    double vx=0.0;
    double vy=0;
    double dt=0.01;
    //double r = 2*0.025;
    double r =0.1125;
    //double r2 = 0.025;
    double r2=1.0*0.1;
    double tau= 10.0*sqrt(Numcells)/sqrt(200);

    //cout << cellpos.data[1].x << endl;
    //cout << h_past.data[1].x << endl;

    for(int y=0; y<gridsteps*gridsteps;y++){
        //Check Up
        if(y<gridsteps)
            {up=conc.data[gridsteps*gridsteps-gridsteps+y+indent];}
        else{up=conc.data[(y-gridsteps)+indent];}
        //Check Down
        if(y>gridsteps*gridsteps-gridsteps)
            {down=conc.data[(gridsteps-(gridsteps*gridsteps-y))+indent];}
        else{down=conc.data[(y+gridsteps)+indent];}
        //Check Left
        if(y%gridsteps==0)
            {left=conc.data[y+gridsteps-1+indent];}
        else{left=conc.data[y-1+indent];}
        //Check Right
        if(y==gridsteps*gridsteps-1)
            {right=conc.data[indent];}
        else if((y+1)%gridsteps==0)
            {right=conc.data[y-gridsteps+1+indent];}
        else{right=conc.data[y+1+indent];}
    if(concel.data[y]==-1){
        vx=0.0;
        vy=0.0;
    }
    else{
        vx=0.0;
        vy=0.0;
       //vx=h_v.data[int(concel.data[y])].x;
       //vy=h_v.data[int(concel.data[y])].y;
        //vx=h_velocity.data[int(concel.data[y])].x; //Full working
        //vy=h_velocity.data[int(concel.data[y])].y; //Full working
            

    }

    conc.data[y+(gridsteps*gridsteps-indent)]=conc.data[y+indent]+r*(-4*conc.data[y+indent]+up+down+left+right)+0.5*r2*(vx*(right-left)+vy*(up-down))-(dt/tau)*conc.data[y+indent];
    //cout << vx << " " << vy << " " << y << endl;
    }


    //Fix Bounds
    for(int y=0; y<gridsteps*gridsteps;y++){
    //if((y<gridsteps)||((y+1)%gridsteps==0)||(y%gridsteps==0)||(y>gridsteps*gridsteps-gridsteps)){conc.data[y+(gridsteps*gridsteps-indent)]=0.0;}
    //if(y<gridsteps){conc.data[y+(gridsteps*gridsteps-indent)]=100.0;}
    //else if(y>gridsteps*gridsteps-gridsteps-1){conc.data[y+(gridsteps*gridsteps-indent)]=0.0;}

    //Right Wall
    //if(y%gridsteps==0){conc.data[y+(gridsteps*gridsteps-indent)]=100.0;}
    //else if(((y+1)%(gridsteps))==0){conc.data[y+(gridsteps*gridsteps-indent)]=0.0;}
    
    //Center Stripe
    //if(y<gridsteps){conc.data[y+(gridsteps*gridsteps-indent)]=0.0;}
    //else if(y>gridsteps*gridsteps-gridsteps-1){conc.data[y+(gridsteps*gridsteps-indent)]=0.0;}
    //else if((y>-1+gridsteps*ceil(gridsteps/2))&&(y<gridsteps*ceil(1+gridsteps/2))){conc.data[y+(gridsteps*gridsteps-indent)]=100.0;}
    
    //Currently Used Center Stripe
    if((y>-1+gridsteps*ceil(gridsteps/2))&&(y<gridsteps*ceil(1+gridsteps/2))){conc.data[y+(gridsteps*gridsteps-indent)]=100.0;}
   
    //Center Point
    //if(y==ceil(gridsteps/2)-1+gridsteps*ceil(gridsteps/2)){conc.data[y+(gridsteps*gridsteps)]=100.0;}

    }



    //Examine Cell Concentration    
    ArrayHandle<Dscalar2> vertpos(voroCur,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<Dscalar2> celcon(cellcon,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_p(AreaPeriPreferences,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_mot(Motility,access_location::host,access_mode::readwrite);   
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::overwrite); 

    for(int k=0; k<gridsteps*gridsteps;k++){
        concel.data[k]=-1;
    }
    //Set up saving for cell verticies
    int sums=0;
    double checkx=0;
    double checky=0;
    for(int ii=0; ii<Numcells;ii++){
        //cout << ii << endl;
        //cout << h_v.data[ii].x << endl;
        //cout << h_v.data[ii].y << endl;
        //cout << pow((pow((cellpos.data[ii].x-h_past.data[ii].x),2)+pow((cellpos.data[ii].y-h_past.data[ii].y),2))/0.01,0.5) << endl;
        
        //Calculate Velocity and make sure the peroidic box doesn't mess it up
        checkx=(cellpos.data[ii].x-h_past.data[ii].x);
        checky=(cellpos.data[ii].y-h_past.data[ii].y);

        if(checkx>.9*sqrt(Numcells)){checkx=checkx-sqrt(Numcells);}
        if(checky>.9*sqrt(Numcells)){checky=checky-sqrt(Numcells);}
        if(checkx<-.9*sqrt(Numcells)){checkx=checkx+sqrt(Numcells);}
        if(checky<-.9*sqrt(Numcells)){checky=checky+sqrt(Numcells);}

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
        //Only the cluster cells depending on conc
        if(h_ct.data[i]!=0){
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
                    if(avert.data[currentnum].x>sqrt(Numcells)){checkleft=1;}
                    if(avert.data[currentnum].y<0){checkup=1;}
                    if(avert.data[currentnum].y>sqrt(Numcells)){checkdown=1;}           

                    currentnum += 1;}
            for(int k=0; k<gridsteps*gridsteps; k++){
                if(concel.data[k]==-1){
                    if((sqrt(pow(cellpos.data[i].x-cloc.data[k].x,2)+pow(cellpos.data[i].y-cloc.data[k].y,2))<1)||((checkup==1)&&(cloc.data[k].y>sqrt(Numcells)-1))||((checkdown==1)&&(cloc.data[k].y<1))||((checkleft==1)&&(cloc.data[k].x<1))||((checkright==1)&&(cloc.data[k].x>sqrt(Numcells)-1))){
                    InPolygon(i,k,currentnum,checkup,checkdown,checkleft,checkright);
                    }
                }
            }

            //((checkup==1)&&(cellpos.data[i].y>sqrt(Numcells)-1))||((checkdown==1)&&(cellpos.data[i].y<1))||((checkleft==1)&&(cellpos.data[i].x<1))||((checkright==1)&&(cellpos.data[i].x>sqrt(Numcells)-1))
            //(checkup+checkdown+checkleft+checkright!=0))


            total+=celcon.data[i].y;
            if(celcon.data[i].y==0){celcon.data[i].y=1;}
            Dscalar const currentconc = ((celcon.data[i].x))/((celcon.data[i].y))/100;
            ArrayHandle<Dscalar2> h_AP(AreaPeri,access_location::host,access_mode::read);
            ArrayHandle<Dscalar> h_adjust(celladjust,access_location::host,access_mode::overwrite);
            h_adjust.data[i] = currentconc;
            //cout << h_AP.data[i].y << endl;
            //cout << h_mot.data[i].x << endl;
            //cout << 3.8-0.2*(currentconc) << endl; 
            h_p.data[i].y =  3.8+0.4*(currentconc);
            //h_mot.data[i].x = 0.2 - 0.1*(currentconc);

        }
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
        for (int ii = 0; ii < Ncells; ++ii)
            computeVoronoiForceCPU(ii);
        };
    };


//Checks to see if cell i contains the point in the concentration grid k or if it's peridoic images do
void VoronoiQuadraticEnergyWithConcTens::InPolygon(int i, int k, int currentnum, int checkup, int checkdown, int checkleft, int checkright)
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
        //cout << k << endl;
        //if(i==6){cout << k << endl;}   
        }


//Examine Images of Cells
    if(checkup==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y+sqrt(Numcells);
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y+sqrt(Numcells);}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y+sqrt(Numcells);}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
        //cout << k << endl;
       // if(i==6){cout << k << endl;}
    }
    }

    if(checkdown==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x;
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y-sqrt(Numcells);
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x;
                p2y =avert.data[currentnum-1].y-cloc.data[k].y-sqrt(Numcells);}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x;
                p2y =avert.data[currentnum-j].y-cloc.data[k].y-sqrt(Numcells);}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
        //cout << k << endl;
       // if(i==6){cout << k << endl;}
    }
    }

    if(checkleft==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x-sqrt(Numcells);
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x-sqrt(Numcells);
                p2y =avert.data[currentnum-1].y-cloc.data[k].y;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x-sqrt(Numcells);
                p2y =avert.data[currentnum-j].y-cloc.data[k].y;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
        //cout << k << endl;
      //  if(i==6){cout << k << endl;}
    }
    }

    if(checkright==1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x+sqrt(Numcells);
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y;
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x+sqrt(Numcells);
                p2y =avert.data[currentnum-1].y-cloc.data[k].y;}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x+sqrt(Numcells);
                p2y =avert.data[currentnum-j].y-cloc.data[k].y;}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
        //cout << k << endl;
    //  if(i==6){cout << k  << endl;}
    }
    }


    if(checkright+checkleft+checkdown+checkup>1){
    angle=0;
        for(int j=0; j<h_nn.data[i];j++){
            p1x = avert.data[currentnum-j-1].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells);
            p1y = avert.data[currentnum-j-1].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells);
            if (j==0){
                p2x =avert.data[currentnum-1].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells);
                p2y =avert.data[currentnum-1].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells);}
            else{
                p2x =avert.data[currentnum-j].x-cloc.data[k].x+(checkright-checkleft)*sqrt(Numcells);
                p2y =avert.data[currentnum-j].y-cloc.data[k].y+(checkup-checkdown)*sqrt(Numcells);}
            dtheta=VertAngle(p1x,p1y,p2x,p2y);
        angle+= dtheta;
        }
    if(abs(angle)>M_PI){concel.data[k]=i;
        celcon.data[i].x+=conc.data[k];
        celcon.data[i].y+=1;
       // cout << k << endl;
       // if(i==6){cout << k << endl;}
    }
    }


};


//Calculates the angle difference between two verticies of polygon and a test point inside it
double VoronoiQuadraticEnergyWithConcTens::VertAngle(double p1x, double p1y, double p2x, double p2y)
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







