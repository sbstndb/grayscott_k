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

#include <Kokkos_Core.hpp>

using real = double ; 


const real Du = 0.16 ; 
const real Dv = 0.08 ; 
const real k = 0.065 ; 
const real F = 0.035 ; 
const real dt = 1.0 ;

const int frames = 100; 
const int nrepeat = 1000; 

const int nx = 256 ; 
const int ny = nx ; 



void save_to_file(const std::string& filename, int frame, auto view1, auto view2, auto nx, auto ny){

}


void save_to_file(const auto& view1, const auto& view2, auto nx, auto ny, auto frame_number) {

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
	file_u << nx << " " << ny << std::endl ; 
	file_v << nx << " " << ny << std::endl ; 

	for (int i = 0 ; i < nx ; i++) {
		for (int j = 0 ; j < ny ; j++){
			file_u << view1(i,j) << " ";
			file_v << view2(i,j) << " ";
		}
		file_u << std::endl ; 
		file_v << std::endl ;
	}
	file_u.close();
	file_v.close();
}

int main( int argc, char** argv ){

	Kokkos::initialize( argc, argv );
	{
	// Allocate y, x vectors and Matrix A on device.
	typedef Kokkos::View<real**>  ViewVectorType;
	ViewVectorType u("u", nx, ny);
	ViewVectorType v("v", nx, ny);

        ViewVectorType utmp("utmp", nx, ny);
        ViewVectorType vtmp("vtmp", nx, ny);


	// Create host mirrors of device views.
	ViewVectorType::HostMirror h_u = Kokkos::create_mirror_view( u ) ;
        ViewVectorType::HostMirror h_v = Kokkos::create_mirror_view( v ) ;

	// init views
	//
	Kokkos::Timer timer;
	Kokkos::parallel_for("init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, ny}),
			KOKKOS_LAMBDA (int i, int j) {
		u(i,j) = 1.0; 
		v(i,j) = 0.0	;
		utmp(i,j) = 0.0;
		vtmp(i,j) = 0.0;
		if (i > 2*nx/5 && i < 3*nx/5 && j > 2*ny/5 && j < 3*ny/5){
                        v(i, j) = 0.5 ;
                        u(i, j) = 0.25 ;
		}
	});

	for (int frame = 0 ; frame < frames ; frame++){
		for (int repeat = 0 ; repeat < nrepeat ; repeat++){
		// update kernel 
			Kokkos::parallel_for("stencil", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{nx-1, ny-1}),
					KOKKOS_LAMBDA (int i, int j) {
				real invdx = 1.00;//(nx*nx);
//				real lap_u = 0.2 * (u(i+1, j) + u(i-1, j) + u(i, j+1) + u(i, j-1))
//					+ 0.05 * (u(i-1, j-1) + u(i-1, j+1) + u(i+1, j-1) + u(i+1, j+1))
//			       		- 1.0 * u(i,j);
//		                real lap_v = 0.2 * (v(i+1, j) + v(i-1, j) + v(i, j+1) + v(i, j-1)) 
//                                        + 0.05 * (v(i-1, j-1) + v(i-1, j+1) + v(i+1, j-1) + v(i+1, j+1))
//                                        - 1.0 * v(i,j);

                              real lap_u = u(i+1, j) + u(i-1, j) + u(i, j+1) + u(i, j-1) - 4.0 * u(i,j);
                              real lap_v = v(i+1, j) + v(i-1, j) + v(i, j+1) + v(i, j-1) - 4.0 * v(i,j);


				real du = Du * lap_u * invdx - u(i,j) * v(i,j) * v(i,j) + F * (1.0 - u(i,j));
		                real dv = Dv * lap_v * invdx + u(i,j) * v(i,j) * v(i,j) - (F + k) * v(i,j);		
		
				utmp(i,j) = dt * du ; 
				vtmp(i,j) = dt * dv ; 
			});
                        Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{nx-1, ny-1}),
                                        KOKKOS_LAMBDA (int i, int j) {
                                u(i,j) += utmp(i,j);
				v(i,j) += vtmp(i,j);
                        });
			
		}
		// copy on host and save with save_to_file
		Kokkos::deep_copy(h_u, u);
		Kokkos::deep_copy(h_v, v);
		save_to_file(h_u, h_v, nx, ny, frame);
	}

	Kokkos::fence();
	double time = timer.seconds();
	std::cout << "Elapsed time : " << time << std::endl ; 
	}
	Kokkos::finalize();

	return 0;
}

