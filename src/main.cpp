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




int main( int argc, char** argv ){

	Kokkos::initialize( argc, argv );
	{
	// Allocate y, x vectors and Matrix A on device.
	int N = 1024 ; 
	typedef Kokkos::View<double*>  ViewVectorType;
	ViewVectorType d("d", N);
	// Create host mirrors of device views.
	ViewVectorType::HostMirror h_d = Kokkos::create_mirror_view( d ) ;

	}
	Kokkos::finalize();
	return 0;
}
