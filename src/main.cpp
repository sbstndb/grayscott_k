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

const double Du = 0.5 ; 
const double Dv = 0.3 ; 
const double k = 0.05 ; 
const double F = 0.03 ; 
const double dt = 0.1 ;

const int frames = 10 ; 
const int nrepeat = 10 ; 

const int nx = 4000 ; 
const int ny = 4000 ; 

void save_to_file(const std::string& filename, int frame, auto view1, auto view2, auto nx, auto ny){

}


int main( int argc, char** argv ){

	Kokkos::initialize( argc, argv );
	{
	// Allocate y, x vectors and Matrix A on device.
	typedef Kokkos::View<double**>  ViewVectorType;
	ViewVectorType u("u", nx, ny);
	ViewVectorType v("v", nx, ny);
	// Create host mirrors of device views.
	ViewVectorType::HostMirror h_u = Kokkos::create_mirror_view( u ) ;
        ViewVectorType::HostMirror h_v = Kokkos::create_mirror_view( v ) ;

	// init views
	//
	Kokkos::Timer timer;
	Kokkos::parallel_for("init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, ny}),
			KOKKOS_LAMBDA (int i, int j) {
		u(i,j) = 1.0 ; 
		v(i,j) = 0.0 ;
		if (i == nx/2 && j == ny/2){
			u(nx/2, ny/2) /=2.0 ; 
		}
	});

	for (int frame = 0 ; frame < frames ; frame++){
		for (int repeat = 0 ; repeat < nrepeat ; repeat++){
		// update kernel 
			Kokkos::parallel_for("stencil", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{nx-1, ny-1}),
					KOKKOS_LAMBDA (int i, int j) {
				double lap_u = u(i+1, j) + u(i-1, j) + u(i, j+1) + u(i, j-1) - 4.0 * u(i,j);
		                double lap_v = v(i+1, j) + v(i-1, j) + v(i, j+1) + v(i, j-1) - 4.0 * v(i,j);
		
				double du = Du * lap_u - u(i,j) * v(i,j) * v(i,j) + F * (1.0 - u(i,j));
		                double dv = Dv * lap_v + u(i,j) * v(i,j) * v(i,j) - (F + k) * v(i,j);		
		
				u(i,j) += dt * du ; 
				v(i,j) += dt * dv ; 
			});
		}
		// copy on host and save with save_to_file
		Kokkos::deep_copy(h_u, u);
		Kokkos::deep_copy(h_v, v);
	}

	Kokkos::fence();
	double time = timer.seconds();
	std::cout << "Elapsed time : " << time << std::endl ; 
	}
	Kokkos::finalize();

	return 0;
}

