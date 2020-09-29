#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#define ENABLE_CUDA
#include "vertexQuadraticEnergy.h"
#include "selfPropelledCellVertexDynamics.h"
#include "brownianParticleDynamics.h"
#include "DatabaseNetCDFAVM.h"
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;


int main(int argc, char*argv[])
{
    int numpts = 600; //number of cells
    int USE_GPU = -1; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps    = 100000000; //number of time steps to run after initialization
    int initSteps = 1000000; //number of initialization steps
    Dscalar dt = 0.01; //the time step size
    Dscalar p0 = 3.5;  //the preferred perimeter
    Dscalar a0 = 1.0;  // the preferred area
    Dscalar v0 = 0.001;  // the self-propulsion
    //Dscalar T = 0.001;  // the self-propulsion
    Dscalar Dr = 1.0;  //the rotational diffusion constant of the cell directors
    int program_switch = 0; //various settings control output
    int p_id=1; //If you are running locally only 1 (not ensemble) than this is 0. This process id is for cluster runs.
    int Nvert = 2*numpts;
    int c;
    //int timeToT1 = 0;
    //int w;

    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            //case 'v': T = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
	      //Turn off q below and remove q from "n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:w:"list above
	      //and from baseRun.sh for analysis!
	  //case 'q': p_id = atoi(optarg); break;
            ///Removed w from "n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:w:"list above
            //case 'w': timeToT1 = atoi(optarg); break;
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

     clock_t t1,t2; //clocks for timing information
     bool reproducible = true; // if you want random numbers with a more random seed each run, set this to false
     //check to see if we should run on a GPU
     bool initializeGPU = true;
     if (USE_GPU >= 0)
        {
        bool gpu = chooseGPU(USE_GPU);
        if (!gpu) return 0;
        cudaSetDevice(USE_GPU);
        }
     else
        initializeGPU = false;

     printf("p_id=%i \n", p_id);
     bool runSPV = false;//setting this to true will relax the random cell positions to something more uniform before running vertex model dynamics
   
     //     EOMPtr spp = make_shared<selfPropelledCellVertexDynamics>(numpts,Nvert);
     shared_ptr<brownianParticleDynamics> bd = make_shared<brownianParticleDynamics>(Nvert);
     shared_ptr<VertexQuadraticEnergy> avm = make_shared<VertexQuadraticEnergy>(numpts,a0,p0,reproducible,runSPV);

     avm->setCellPreferencesUniform(a0,p0);
     avm->setModuliUniform(100.0,1.0);

     FILE *f;
     int rec;
     //
     //get the rec number. Once you have it for the first p_id you can quit the loop.///
     for(int i=0; i<p_id; i++){
     char dataname[1000000];
     //sprintf(dataname,"/home/elawsonk/Shear/output/test_N%i_p%.3f.nc",numpts,p0);
     sprintf(dataname,"/home/elawsonk/cellGPU-master/gaptest_N_200_ka_100.0000_dt_0.00010_a0_1.00_p0_%.4f_run_0.nc",p0);
     // sprintf(dataname,"/home/gerdemci/cellGPU-T1-ShearModulus/output/vertexBD_N%i_p%.3f_T%.6f_tT1%i_p_id%i.nc",numpts,p0,T,timeToT1,i);
       //sprintf(dataname,"/home/gerdemci/cellGPU-T1-ShearModulus/output/vertexMinimize_N%i_p%.3f_tT1%i_p_id%i.nc",numpts,p0,timeToT1,i);
       printf("%s\n",dataname);
       //Check if the file exists in the output folder. if it does then do the scan
       if ((f = fopen(dataname, "r")) == NULL)
	 {
	   printf("Error! opening file\n");
	   printf("%s\n",dataname);
	   return -1;
	 }else{
	 fclose(f);
	 AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::ReadOnly);
	 rec=ncdat.GetNumRecs();
	 printf("rec=%u\n",rec);
	 break;
       }
     }
     Dscalar shearmodulustotal=0.0;
     Dscalar shearmodulusList[p_id];
     for(int i=0; i<p_id; i++){
        char dataname[1000000];
	//sprintf(dataname,"/home/elawsonk/Shear/output/test_N%i_p%.3f.nc",numpts,p0);
    sprintf(dataname,"/home/elawsonk/cellGPU-master/gaptest_N_200_ka_100.0000_dt_0.00010_a0_1.00_p0_%.4f_run_0.nc",p0);
	//printf("%s\n",dataname);
        //Check if the file exists in the output folder. if it does then do the scan
        if ((f = fopen(dataname, "r")) == NULL)
            {
               printf("Error! opening file\n"); 
               printf("%s\n",dataname);
	       return -1;
        }else{
	        fclose(f);
            AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::ReadOnly);	
            rec=ncdat.GetNumRecs();
	    //printf("rec=%u\n",rec);
	    //Read the final state for the shear modulus calculation
            ncdat.ReadState(avm,rec-1,false);
	    Matrix2x2 d2E;
            //d2E=avm->d2Edridrj(0,0);
	    //printf("d2E.x11=%lf\n",d2E.x11);
	    //printf("d2E.x12=%lf\n",d2E.x12);
	    //printf("d2E.x21=%lf\n",d2E.x21);
	    //printf("d2E.x22=%lf\n",d2E.x22);
	
	    int DIM=2;
	    typedef Eigen::Matrix<double,Dynamic,Dynamic> MatrixX;
	    typedef Eigen::Matrix<double,Dynamic,1> VectorX;
	    //Initialize the dynamical matrix and set all values to zero
	    MatrixX dynamicalMatrix(Nvert*DIM,Nvert*DIM);
	    dynamicalMatrix.setZero();

	    // Fill out the dynamical matrix
	    for(int i = 0; i < Nvert; i++) {
	      // Only calculate values for the upper triangle of the matrix
	      for(int j = i + 1; j < Nvert; j++) {
		// Fill out the upper right square of values
		d2E=avm->d2Edridrj(i,j);
		dynamicalMatrix(i * DIM, j * DIM) = d2E.x11; //Upper-Left
		dynamicalMatrix(i * DIM, j * DIM + 1) = d2E.x12; //Upper-Right
		dynamicalMatrix(i * DIM + 1, j * DIM) = d2E.x21; //Lower-Left
		dynamicalMatrix(i * DIM + 1, j * DIM + 1) = d2E.x22; //Lower-Right

		// Fill out the lower left square of values by setting i -> j and j -> i
		dynamicalMatrix(j * DIM, i * DIM) = d2E.x11; //Upper-Left
		dynamicalMatrix(j * DIM, i * DIM + 1) = d2E.x21; //Upper-Right, NOTE: FLIPPED FROM ABOVE
		dynamicalMatrix(j * DIM + 1, i * DIM) = d2E.x12; //Lower-Left, NOTE: FLIPPED FROM ABOVE
		dynamicalMatrix(j * DIM + 1, i * DIM + 1) = d2E.x22; //Lower-Right
	      }
	    }

	    // Apply the sum rule Hii = -Sum(Hij)
            for(int i = 0; i < Nvert; i++) {
              for(int j = 0; j < Nvert; j++) {
		if(j!=i){
		 dynamicalMatrix(i * DIM, i * DIM) -= dynamicalMatrix(i * DIM, j * DIM); //Upper-Left       
		 dynamicalMatrix(i * DIM, i * DIM + 1) -= dynamicalMatrix(i * DIM, j * DIM + 1); //Upper-Right
		 dynamicalMatrix(i * DIM + 1, i * DIM) -= dynamicalMatrix(i * DIM + 1, j * DIM); //Lower-Left
		 dynamicalMatrix(i * DIM + 1, i * DIM + 1) -= dynamicalMatrix(i * DIM + 1, j * DIM + 1); //Lower-Right  
		}
	      }
	    }


	    //Initialize eigenvalues and eigenvectors
	    VectorX eigenvalues;
	    MatrixX eigenvectors;

	    SelfAdjointEigenSolver<MatrixX> eigensolver(dynamicalMatrix);
	    eigenvalues = eigensolver.eigenvalues();
	    eigenvectors = eigensolver.eigenvectors();

	    // cout << "Here is the dynamical matrix" << endl << dynamicalMatrix << endl << endl;
	    // cout << "The eigenvalues are:" << endl << eigenvalues << endl;
	    // cout << "The matrix of eigenvectors is:" << endl << eigenvectors << endl << endl;
	    // cout << "The first eigenvector is:"<< endl << eigenvectors.col(0) << endl;

            int positiveNonZeroEigenvalues=0;
            for(int mm=0; mm<2*Nvert; ++mm){
	      if( (eigenvalues(mm)>0) && (log(eigenvalues(mm))>log(1e-12)) )
		  positiveNonZeroEigenvalues++;
	    }
	    Dscalar shearmodulus;                                   
            if (positiveNonZeroEigenvalues==2*Nvert-2){
	      //Note that below shear modulus calculation is good for rigid only 
	      //as the loop over mm start from 2 excluding the two zero modes.                          
	      //The following is to avoid the shear modulus calculation for non-minimized states. 
	      //Ones the minimization is improved this needs to be updated.
	      Dscalar d2Edgammadrialpha[2*Nvert];
	      for(int ii=0; ii<Nvert; ++ii){
		d2Edgammadrialpha[2*ii]=avm->d2Edgammadrialpha(ii).x;
		d2Edgammadrialpha[2*ii+1]=avm->d2Edgammadrialpha(ii).y;
	      }
	      Dscalar sum=0;
	      for(int mm=2; mm<2*Nvert; ++mm){
		Dscalar dotproduct=0.0;
		for(int ii=0; ii<2*Nvert; ++ii){
		  dotproduct += (d2Edgammadrialpha[ii]*eigenvectors.col(mm)[ii]);
		}
		sum+=(1/eigenvalues(mm))*dotproduct*dotproduct;
	      }
	      Dscalar d2Edgamma2Total=0;
	      for(int ii=0; ii<numpts; ++ii){
		d2Edgamma2Total+=avm->d2Edgamma2(ii);
	      }
	      shearmodulus=(d2Edgamma2Total-sum)/numpts;
            }else{
	      shearmodulus=0.0;
	    }
	    //The following is to avoid the shear modulus calculation for non-minimized states. 
	    //Ones the minimization is improved, this needs to be changed.
            if(shearmodulus<0){
              shearmodulus=0.0;
            }
	    shearmodulusList[p_id]=shearmodulus;
	    printf("shearmodulus[%i]=%lf\n", i, shearmodulus);
	    shearmodulustotal+=shearmodulus;
           }

      }

     Dscalar shearModulusAverage;
     shearModulusAverage=shearmodulustotal / (double) p_id;

     Dscalar standardDeviationTotal;
     Dscalar standardDeviation;
     for (int jj=0; jj<p_id; ++jj){
       standardDeviationTotal += pow(shearmodulusList[jj]-shearModulusAverage,2);
     }
     if(p_id>1){
       standardDeviation=sqrt(standardDeviationTotal/(p_id-1));
     }else{
       standardDeviation=0.0;
     }

     FILE *fw;
     char datafilename2[10000];
     sprintf(datafilename2,"./output/shear_test.txt");
     fw = fopen( datafilename2, "a");
     if(fw == NULL){
       printf("Error!");   
       exit(1);             
     }

     fprintf( fw, "%.3lf %.9lf %.9lf %.9lf\n", p0, v0, shearModulusAverage, standardDeviation);
     fclose(fw);



    return 0;
};
