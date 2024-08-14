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

class GrayScott {
public : 




	enum class BoundaryCondition {
		Periodic
	};
	enum class InitializationType {
		Random, 
		CentralBlock,
		Full
	};
	enum class StencilType {
		SevenPoint,
		TwentySevenPoint
	};


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
		bool is3D = true ; 
		GrayScott::BoundaryCondition bc = BoundaryCondition::Periodic; 
		GrayScott::InitializationType initType = InitializationType::CentralBlock;
		GrayScott::StencilType stencilType = StencilType::SevenPoint;
	};


        Setting setting ;
        int nx, ny, nz;
        Kokkos::View<real***>u, v, utmp, vtmp;
        Kokkos::View<real***>::HostMirror h_u, h_v;
        Kokkos::View<GrayScott::Setting*> s ;
        Kokkos::View<GrayScott::Setting*>::HostMirror h_s;




	GrayScott(Setting& setting) : setting(setting) {
		nx = setting.nx ; 
		ny = setting.ny ; 
		nz = setting.is3D ? setting.nz : 1 ; 

                std::cout << "allocate?" << std::endl ;
		allocate();
                std::cout << "initialize?" << std::endl ;
		initialize() ; 
                std::cout << "initialize!" << std::endl ;		
	}

	void allocate(){
                u = Kokkos::View<real***>("u", nx, ny, nz);
                v = Kokkos::View<real***>("v", nx, ny, nz);
                h_u = Kokkos::create_mirror_view (u);
                h_v = Kokkos::create_mirror_view (v);

		std::cout << " size of u : " << u.size() << std::endl ; 
		std::cout << " size of h_u:" << h_u.size() << std::endl ; 

                std::cout << " size of v : " << v.size() << std::endl ;
                std::cout << " size of v_u:" << h_v.size() << std::endl ;


                s = Kokkos::View<GrayScott::Setting*>("s",1);
                h_s = Kokkos::create_mirror_view (s);


                std::cout << " size of s : " << s.size() << std::endl ;
                std::cout << " size of h_s:" << h_s.size() << std::endl ;


                Kokkos::deep_copy(s, h_s);
		Kokkos::deep_copy(u, h_u);
		Kokkos::deep_copy(v, h_v);

                utmp = Kokkos::View<real***>("utmp", nx, ny, nz);
                vtmp = Kokkos::View<real***>("vtmp", nx, ny, nz);
	}


	void run(){
		HDF hdf(nx, ny, nz, "dump.hdf5");
		deep_copy(h_v, v);
		std::cout << " save hdf 0?" << std::endl ; 		
                std::cout << " save hdf 0!" << std::endl ;
                std::cout << " size of hview before hdf : " << h_v.size() << std::endl ;		
		save_to_hdf(hdf, h_v, std::to_string(0));
		for (int frame = 0 ; frame < setting.frames; frame++){
			for (int repeat = 0 ; repeat < setting.nrepeat; repeat++){
				std::cout << "frame - repeat : " << frame << " " << repeat << std::endl ; 
				perform_time_step() ; 
			}
			Kokkos::deep_copy(h_u, u);
			Kokkos::deep_copy(h_v, v);
			save_to_hdf(hdf, h_v, std::to_string(frame+1));
		}
	}

	void initialize() {
		switch(setting.initType) {
			case InitializationType::Random : 
				initialize_random();
			case InitializationType::CentralBlock:
				initialize_central_block();
			case InitializationType::Full:
				initialize_full();
		}	
	}

	void initialize_random() {
	        Kokkos::parallel_for("init_random", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
	            KOKKOS_LAMBDA(int i, int j, int k) {
//	                u(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
//	                v(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
	            });
	}
	void initialize_central_block() {
	        Kokkos::parallel_for("init_central_block", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
	            KOKKOS_LAMBDA(int i, int j, int k) {
	                u(i, j, k) = 1.0f;
	                v(i, j, k) = 0.0f;
	
	                if (i > 2 * s(0).nx / 5 && i < 3 * s(0).nx / 5 &&
	                    j > 2 * s(0).ny / 5 && j < 3 * s(0).ny / 5 &&
	                    k > 2 * s(0).nz / 5 && k < 3 * s(0).nz / 5) {
	                    v(i, j, k) = 0.5f;
	                    u(i, j, k) = 0.25f;
	                }
	            });
	}
	void initialize_full() {
		Kokkos::parallel_for("init_full", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
				KOKKOS_LAMBDA(int i, int j, int k) {
			u(i, j, k) = 1.0f;
			v(i, j, k) = 0.0f;
		});
	}

	void perform_time_step() {

                switch(setting.stencilType) {
                        case StencilType::SevenPoint :
                                stencil_seven_point();
                        case StencilType::TwentySevenPoint:
                                stencil_twenty_seven_point();
                }

		Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
				KOKKOS_LAMBDA(int i, int j, int k) {
			u(i, j, k) += utmp(i, j, k);
			v(i, j, k) += vtmp(i, j, k);
		});
	}

	void stencil_seven_point() {
		Kokkos::parallel_for("stencil_7", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
			KOKKOS_LAMBDA(int i, int j, int k) {
			int im = (i == 0) ? s(0).nx - 1 : i - 1;
			int ip = (i == s(0).nx - 1) ? 0 : i + 1;
			int jm = (j == 0) ? s(0).ny - 1 : j - 1;
			int jp = (j == s(0).ny - 1) ? 0 : j + 1;
			int km = (k == 0) ? s(0).nz - 1 : k - 1;
			int kp = (k == s(0).nz - 1) ? 0 : k + 1;

			real lap_u = u(ip, j, k) + u(im, j, k) + u(i, jp, k) + u(i, jm, k) +
				u(i, j, km) + u(i, j, kp) - 6.0f * u(i, j, k);
			real lap_v = v(ip, j, k) + v(im, j, k) + v(i, jp, k) + v(i, jm, k) +
				v(i, j, km) + v(i, j, kp) - 6.0f * v(i, j, k);

			real du = s(0).Du * lap_u - u(i, j, k) * v(i, j, k) * v(i, j, k) + s(0).F * (1.0f - u(i, j, k));
			real dv = s(0).Dv * lap_v + u(i, j, k) * v(i, j, k) * v(i, j, k) - (s(0).F + s(0).K) * v(i, j, k);

			utmp(i, j, k) = s(0).dt * du;
			vtmp(i, j, k) = s(0).dt * dv;
            });	
	}
	void stencil_twenty_seven_point(){
		//todo
	}

	void save_to_hdf(HDF& hdf, const auto & h_view, const std::string filename){
		HighFive::DataSet dataset = hdf.file.createDataSet<real>(filename, HighFive::DataSpace(hdf.dims));

		std::cout << " nx ny nz : " << nx << " " << ny << " " << nz << std::endl ; 
		std::cout << " size of hview in hdf : " << h_view.size() << std::endl ;
		for (int i = 0 ; i < nx ; i++){
	                        for (int j = 0 ; j < ny ; j++){
	                                        for (int k = 0 ; k < nz ; k++){
	                                                hdf.tmp[i*ny*nz + j*nz + k] = h_view(i,j,k);
	                                        }
	                        }
	        }
	        hdf.dataset.write(hdf.tmp);
	}

};


int main(int argc, char ** argv){
	Kokkos::initialize(argc, argv);
	

		GrayScott::Setting setting ; 
		setting.is3D = true ; 
		setting.initType = GrayScott::InitializationType::CentralBlock;
		setting.stencilType = GrayScott::StencilType::SevenPoint;
		setting.bc = GrayScott::BoundaryCondition::Periodic;
		
		GrayScott simulation(setting); 
		std::cout << "simulation init" << std::endl ;
		simulation.run() ; 
	
	Kokkos::finalize() ; 
	return 0 ; 
}


