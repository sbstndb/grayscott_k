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


const real Du = 0.16 ; 
const real Dv = 0.08 ; 
const real K = 0.065 ; 
const real F = 0.050 ; 
const real dt = 1.0 ;

const int frames = 10; 
const int nrepeat = 100; 

const int nx = 64 ; 
const int ny = nx ; 
const int nz = nx ; 

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
	{
                std::cout << "coucou0" << std::endl ;		
		dataset = file.createDataSet<real>("default", HighFive::DataSpace(dims));
		std::cout << "coucou" << std::endl ; 
	}
};


void configure_hdf(HDF& hdf, auto nx, auto ny, auto nz){
        using namespace HighFive ;
//        hdf.dataset = hdf.file.createDataSet<real>("filename", DataSpace(hdf.dims));

}

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

	Kokkos::initialize( argc, argv );
	{
	// Allocate y, x vectors and Matrix A on device.
	typedef Kokkos::View<real***>  ViewVectorType;
	ViewVectorType u("u", nx, ny, nz);
	ViewVectorType v("v", nx, ny, nz);

        ViewVectorType utmp("utmp", nx, ny, nz);
        ViewVectorType vtmp("vtmp", nx, ny, nz);


	// Create host mirrors of device views.
	ViewVectorType::HostMirror h_u = Kokkos::create_mirror_view( u ) ;
        ViewVectorType::HostMirror h_v = Kokkos::create_mirror_view( v ) ;

	// init views
	//
	Kokkos::Timer timer;
	Kokkos::parallel_for("init", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx, ny,nz}),
			KOKKOS_LAMBDA (int i, int j, int k) {
		u(i,j,k) = 1.0; 
		v(i,j,k) = 0.0	;
		utmp(i,j,k) = 0.0;
		vtmp(i,j,k) = 0.0;
		if (i > 2*nx/5 && i < 3*nx/5 && j > 2*ny/5 && j < 3*ny/5 && k > 2*nz/5 && k < 3*nz/5){
                        v(i, j, k) = 0.5 ;
                        u(i, j, k) = 0.25 ;
		}
	});

	HDF hdf(nx, ny, nz, "dump") ; 
	save_to_hdf(hdf, h_v, nx, ny, nz, std::to_string(0));

	for (int frame = 0 ; frame < frames ; frame++){
		for (int repeat = 0 ; repeat < nrepeat ; repeat++){
		// update kernel 
			Kokkos::parallel_for("stencil", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1},{nx-1, ny-1, nz-1}),
					KOKKOS_LAMBDA (int i, int j, int k) {
				real invdx = 1.00;//(nx*nx);
//				 7 points 3D stencil 
//				 future: use 27 points 3D laplacian stencil 
                              real lap_u = u(i+1, j, k) + u(i-1, j, k) + u(i, j+1, k) + u(i, j-1, k) 
			      		+ u(i, j, k-1) + u(i, j, k+1) 
					- 6.0 * u(i, j, k);
                              real lap_v = v(i+1, j, k) + v(i-1, j, k) + v(i, j+1, k) + v(i, j-1, k)
			     		+ v(i, j, k-1) + v(i, j, k+1)
				       	- 6.0 * v(i,j, k);


				real du = Du * lap_u * invdx - u(i,j,k) * v(i,j,k) * v(i,j,k) + F * (1.0 - u(i,j,k));
		                real dv = Dv * lap_v * invdx + u(i,j,k) * v(i,j,k) * v(i,j,k) - (F + K) * v(i,j,k);		
		
				utmp(i,j,k) = dt * du ; 
				vtmp(i,j,k) = dt * dv ; 
			});
                        Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1},{nx-1, ny-1, nz-1}),
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

