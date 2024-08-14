#include <iostream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <filesystem>
#include <random>
#include <vector>


#include <Kokkos_Core.hpp>


#include "../external/HighFive/include/highfive/highfive.hpp"
 

using real = float ; 

/*
 * for n = 256
const real Du = 0.15 ;
const real Dv = 0.081 ;
const real K = 0.064 ;
const real F = 0.050 ;
const real dt = 1.0 ;
*/

/*
// * for n = 64
const real Du = 0.15 ; 
const real Dv = 0.08 ; 
const real K = 0.064 ; 
const real F = 0.050 ; 
const real dt = 1.0 ;
*/


struct Setting {
	real Du = 0.15 ;
	real Dv = 0.08 ;
	real K = 0.064 ;
	real F = 0.0505 ;
	real dt = 1.0 ;

	int frames = 100 ;
	int nrepeat = 1000 ; 

	int nx = 64 ;
	int ny = nx ;
	int nz = nx ;
};


class HDF {
public:
	HighFive::File file;
	HighFive::DataSet dataset;
	std::vector<size_t> dims ; 
	std::vector<real> tmp ; 

	HDF(int nx, int ny , int nz, std::string dumpname) :
	       file(dumpname, HighFive::File::Overwrite),
	       dims {static_cast<size_t>(nx*ny*nz)},
	       tmp(nx*ny*nz)
	{	}
};


void save_to_hdf(HDF& hdf, const auto& h_view, auto nx, auto ny, auto nz, auto framename){
	using namespace HighFive ; 
        hdf.dataset = hdf.file.createDataSet<real>(framename, DataSpace(hdf.dims));
	
	for (int i = 0 ; i < nx ; i++){
		        for (int j = 0 ; j < ny ; j++){
				        for (int k = 0 ; k < nz ; k++){
						hdf.tmp[i*ny*nz + j*nz + k] = h_view(i,j,k);
					}
			}
	}
	hdf.dataset.write(hdf.tmp);
}


void save_to_file(const auto& view1, const auto& view2, auto nx, auto ny, auto nz, auto frame_number) {

	std::string directory = "frames";
	std::filesystem::create_directories(directory);

	std::string filename_u = "frames/frame_u_" + std::to_string(frame_number) + ".txt";
        std::string filename_v = "frames/frame_v_" + std::to_string(frame_number) + ".txt";

	std::ofstream file_u(filename_u);
        std::ofstream file_v(filename_v);

	if (!file_u.is_open() || !file_v.is_open()) {
		std::cerr << "Erreur lors de l'ouverture du fichier "  << std::endl;
	return;
	}

	// Enregistrer les particules dans le fichier
	file_u << nx << " " << ny << " " << nz << std::endl ; 
	file_v << nx << " " << ny << " " << nz << std::endl ; 

	for (int i = 0 ; i < nx ; i++) {
		for (int j = 0 ; j < ny ; j++){
			for (int k = 0 ; k < nz ; k++){
				file_u << view1(i,j,k) << " ";
				file_v << view2(i,j,k) << " ";
			}
		}
	}
	file_u.close();
	file_v.close();
}

int main( int argc, char** argv ){
	Setting setting ; 

  // Read command line arguments.
        for ( int i = 0; i < argc; i++ ) {
                if ( ( strcmp( argv[ i ], "-size" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
                        setting.nx = static_cast<int>(atoi( argv[++i]));
			setting.ny = setting.nx ; 
			setting.nz = setting.nx ; 
                printf( "  User nx is %d\n", setting.nx );
        }
        else if ( ( strcmp( argv[ i ], "-frames" ) == 0 ) || ( strcmp( argv[ i ], "-Frames" ) == 0 ) ) {
                setting.frames = static_cast<int>(atoi( argv[++i]));
                printf( "  User frames is %d\n", setting.frames );
        }
        else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
                setting.nrepeat = static_cast<int>(atoi( argv[ ++i ] ));
	}
        else if ( ( strcmp( argv[ i ], "-dt" ) == 0 ) || ( strcmp( argv[ i ], "-Dt" ) == 0 ) ) {
                setting.dt = static_cast<real>(atof( argv[++i]));
               printf( "  User dt is %f\n", setting.dt );
        }
        else if ( ( strcmp( argv[ i ], "-du") == 0 ) || ( strcmp( argv[ i ], "-DU" ) == 0 ) ) {
                setting.Du = static_cast<real>(atof( argv[++i]));
                printf( "  User Du is %f\n", setting.Du );
        }
        else if ( ( strcmp( argv[ i ], "-dv" ) == 0 ) || ( strcmp( argv[ i ], "-DV" ) == 0 ) ) {
                setting.Dv= static_cast<real>(atof( argv[++i]));
                printf( "  User Dv is %f\n", setting.Dv );
        }
        else if ( ( strcmp( argv[ i ], "-k" ) == 0 ) || ( strcmp( argv[ i ], "-K" ) == 0 ) ) {
                setting.K= static_cast<real>(atof( argv[++i]));
                printf( "  User K is %f\n", setting.K );
        }
        else if ( ( strcmp( argv[ i ], "-f" ) == 0 ) || ( strcmp( argv[ i ], "-F" ) == 0 ) ) {
                setting.F = static_cast<real>(atof( argv[++i]));
                printf( "  User f is %f\n", setting.F );
        }
	
        else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
                printf( "  NBody Options:\n" );
                printf( "  -size (-Size) <int>:    Size of the domain\n" );
                printf( "  -frames (-Frames) <int>:         number of saved frames\n" );
                printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
                printf( "  -dt (-Dt) <int>:       timestep\n" );
                printf( "  -du (-DU) <int>:       Diffusivité pour u \n" );
                printf( "  -dv (-DV) <int>:        Diffucivité pour v\n" );
                printf( "  -K (-K) <int>:         valeur K\n" );
                printf( "  -f (-Frames) <int>:      valeur f\n" );			
                printf( "  -help (-h):            print this message\n\n" );
                exit( 1 );
        }
}



	Kokkos::initialize( argc, argv );
	{
	// Allocate y, x vectors and Matrix A on device.
	typedef Kokkos::View<real***>  ViewVectorType;
	typedef Kokkos::View<Setting*> SettingType ;

        SettingType s("setting", 1);
        SettingType::HostMirror h_s = Kokkos::create_mirror_view (s);
	h_s[0] = setting; 
        int nx = setting.nx ;
        int ny = setting.ny ;
        int nz = setting.nz ;
	int frames = setting.frames ;
	int nrepeat = setting.nrepeat ;	

	ViewVectorType u("u", nx, ny, nz);
	ViewVectorType v("v", nx, ny, nz);

        ViewVectorType utmp("utmp", nx, ny, nz);
        ViewVectorType vtmp("vtmp", nx, ny, nz);

	// Create host mirrors of device views.
	ViewVectorType::HostMirror h_u = Kokkos::create_mirror_view( u ) ;
        ViewVectorType::HostMirror h_v = Kokkos::create_mirror_view( v ) ;

        
	Kokkos::deep_copy(s, h_s);


	// init views
	//
	Kokkos::Timer timer;
	Kokkos::parallel_for("init", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, { nx, ny, nz}),
			KOKKOS_LAMBDA (int i, int j, int k) {
		u(i,j,k) = 1.0; 
		v(i,j,k) = 0.0	;
		utmp(i,j,k) = 0.0;
		vtmp(i,j,k) = 0.0;
		if (i > 2*s(0).nx/5 && i < 3*s(0).nx/5 && j > 2*s(0).ny/5 && j < 3*s(0).ny/5 && k > 2*s(0).nz/5 && k < 3*s(0).nz/5){
                        v(i, j, k) = 0.5 ;
                        u(i, j, k) = 0.25 ;
		}
	});


	HDF hdf(nx, ny, nz, "dump") ; 
	save_to_hdf(hdf, h_v, nx, ny, nz, std::to_string(0));

	for (int frame = 0 ; frame < frames ; frame++){
		for (int repeat = 0 ; repeat < nrepeat ; repeat++){
		// update kernel 
			Kokkos::parallel_for("stencil", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}),
					KOKKOS_LAMBDA (int i, int j, int k) {
				real invdx = 1.00;//(nx*nx);
//				 7 points 3D stencil 
//				 future: use 27 points 3D laplacian stencil 
//

				// periodic BL
			int im = i-1 ; 
			int jm = j-1 ; 
			int km = k-1 ;
			int ip = i+1 ;
			int jp = j+1 ;
			int kp = k+1 ;
			
			if (i == 0) im = nx-1 ;
			if (i == nx-1) ip = 0 ; 
                        if (j == 0) jm = ny-1 ;
                        if (j == ny-1) jp = 0 ;
                        if (k == 0) km = nz-1 ;
                        if (k == nz-1) kp = 0 ;
			
			
                              real lap_u = u(ip, j, k) + u(im, j, k) + u(i, jp, k) + u(i, jm, k) 
			      		+ u(i, j, km) + u(i, j, kp) 
					- 6.0 * u(i, j, k);
                              real lap_v = v(ip, j, k) + v(im, j, k) + v(i, jp, k) + v(i, jm, k)
			     		+ v(i, j, km) + v(i, j, kp)
				       	- 6.0 * v(i,j, k);


				real du = s(0).Du * lap_u * invdx - u(i,j,k) * v(i,j,k) * v(i,j,k) + s(0).F * (1.0 - u(i,j,k));
		                real dv = s(0).Dv * lap_v * invdx + u(i,j,k) * v(i,j,k) * v(i,j,k) - (s(0).F + s(0).K) * v(i,j,k);		
		
				utmp(i,j,k) = s(0).dt * du ; 
				vtmp(i,j,k) = s(0).dt * dv ; 
			});



                        Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}),
                                        KOKKOS_LAMBDA (int i, int j, int k) {
                                u(i,j,k) += utmp(i,j,k);
				v(i,j,k) += vtmp(i,j,k);
                        });
			
		}
		// copy on host and save with save_to_file
		Kokkos::deep_copy(h_u, u);
		Kokkos::deep_copy(h_v, v);
//		save_to_file(h_u, h_v, nx, ny, nz, frame);
	        save_to_hdf(hdf, h_v, nx, ny, nz, std::to_string(frame+1));
	}

	Kokkos::fence();
	double time = timer.seconds();
	std::cout << "Elapsed time : " << time << std::endl ; 
	}
	Kokkos::finalize();

	return 0;
}

