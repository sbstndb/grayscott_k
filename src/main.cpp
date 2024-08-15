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
using View3D = Kokkos::View<real***> ;

// --> add this to a default parameter selector :-)
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


class HDF {
/**
 *	@brief HDF that enable saving the several flattened fields host_v in one file
 *	@todo 
 */
public:
	HighFive::File file;
	HighFive::DataSet dataset;
	std::vector<size_t> dims ; 
	std::vector<real> tmp ; 

	HDF(int nx, int ny , int nz, std::string dumpname) :
	       file(dumpname, HighFive::File::Overwrite),
	       dims {static_cast<size_t>(nx*ny*nz)},
	       tmp(nx*ny*nz)	{	}
};

class GrayScott {
/**
 *	@brief Main class of the project
 *	@todo Implement 2D only simulation
 *	@todo Implement different stencil versions
 *	@todo Implement different initializations
 *	@todo Add parser
 *	@todo Add predefined Setting configuration helping the user to reproduce beautiful cases
 *	@todo finetune the kokkos kernels
 *	@todo work on the Functors : refactor them ? 
 */	
public : 
	enum class BoundaryCondition {
		Periodic,
		NoPeriodic
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
/**
 *	@brief struct that define the simulation parameters
 *	@todo allow non cubic parameter (nx != ny f.e)
 *	@todo allow modifying it with a CLI parser or a predefined configuration file
 */		
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
        View3D u, v, utmp, vtmp;
        View3D::HostMirror h_u, h_v;

	GrayScott(Setting& setting) : setting(setting) {
		nx = setting.nx ; 
		ny = setting.ny ; 
		nz = setting.is3D ? setting.nz : 1 ; 

		allocate();
		initialize() ; 
	}

	void allocate(){
/**
 *	@brief Specification and allocation of the several Kokkos Views and mirror Views
 *	@todo Add exception !
 */
                u = View3D("u", nx, ny, nz);
                v = View3D("v", nx, ny, nz);
                h_u = Kokkos::create_mirror_view (u);
                h_v = Kokkos::create_mirror_view (v);

		Kokkos::deep_copy(u, h_u);
		Kokkos::deep_copy(v, h_v);

                utmp = View3D("utmp", nx, ny, nz);
                vtmp = View3D("vtmp", nx, ny, nz);
	}


	void run(){
/**
 *	@brief Main nested loop that call the Kokkos kernel and save the v field in an HDF file
 *	@todo Add timer
 *	@todo enable asynchronous execution of the kernel and the save_to_hdf function
 */
		HDF hdf = HDF(nx, ny, nz, "dump");
		Kokkos::deep_copy(h_v, v);
		Kokkos::fence() ; 
		save_to_hdf(hdf, h_v, std::to_string(0));
		
		for (int frame = 0 ; frame < setting.frames; frame++){
			for (int repeat = 0 ; repeat < setting.nrepeat; repeat++){
				perform_time_step() ; 
			}
			Kokkos::deep_copy(h_u, u);
			Kokkos::deep_copy(h_v, v);
			save_to_hdf(hdf, h_v, std::to_string(frame+1));
		}
	}

	struct FunctorCentralBlock {
/**
 *	@brief Functor that initialize the fields u and v with a central block perturbation
 *	@todo Enable defining the size and the location of the blok perturbation
 */
		View3D u, v, utmp, vtmp ; 
		int nx, ny, nz ; 
		FunctorCentralBlock(View3D u, View3D v, View3D utmp, View3D vtmp,
			       int nx, int ny, int nz) : 
			u(u), v(v), utmp(utmp), vtmp(vtmp),
			nx(nx), ny(ny), nz(nz) {}

                KOKKOS_INLINE_FUNCTION
                void operator()(int i, int j, int k) const {
                	u(i,j,k) = 1.0;
	                v(i,j,k) = 0.0  ;
	                utmp(i,j,k) = 0.0;
	                vtmp(i,j,k) = 0.0;
	                if (i > 2*nx/5 && i < 3*nx/5 && j > 2*ny/5 && j < 3*ny/5 && k > 2*nz/5 && k < 3*nz/5){
	                        v(i, j, k) = 0.5 ;
	                        u(i, j, k) = 0.25 ;
	
	                }
                }
	};

	struct FunctorStencilSevenPoint {
/**
 *	@brief Functor that implements the 7c-3D stencil kernel with perdiodic boundary condition
 *	@todo select with or without periodic boundary condition
 *	@todo separate the laplacian computation (the stencil part) and the du/dv computation that are independants ! then only select the stencil code with each possible variant
 */
                View3D u, v, utmp, vtmp ;
                Setting s ;
                FunctorStencilSevenPoint(View3D u, View3D v,View3D utmp,View3D vtmp,
                               Setting s) :
                        u(u), v(v), utmp(utmp), vtmp(vtmp),
                        s(s) {}

		KOKKOS_INLINE_FUNCTION
		void operator()(int i, int j, int k) const {
			int im = (i == 0) ? s.nx - 1 : i - 1;
                        int ip = (i == s.nx - 1) ? 0 : i + 1;
                        int jm = (j == 0) ? s.ny - 1 : j - 1;
                        int jp = (j == s.ny - 1) ? 0 : j + 1;
                        int km = (k == 0) ? s.nz - 1 : k - 1;
                        int kp = (k == s.nz - 1) ? 0 : k + 1;
                        real lap_u = u(ip, j, k) + u(im, j, k) + u(i, jp, k) + u(i, jm, k) +
                                u(i, j, km) + u(i, j, kp) - 6.0f * u(i, j, k);
                        real lap_v = v(ip, j, k) + v(im, j, k) + v(i, jp, k) + v(i, jm, k) +
                                v(i, j, km) + v(i, j, kp) - 6.0f * v(i, j, k);
                        real du = s.Du * lap_u - u(i, j, k) * v(i, j, k) * v(i, j, k) + s.F * (1.0f - u(i, j, k));
                        real dv = s.Dv * lap_v + u(i, j, k) * v(i, j, k) * v(i, j, k) - (s.F + s.K) * v(i, j, k);
                        utmp(i, j, k) = s.dt * du;
                        vtmp(i, j, k) = s.dt * dv;
		}		
	};

	
        struct FunctorUpdate {
/**
 *	@brief Functor that update the main fields u and v with the temporary utmp and vtmp solutions given by the stencil kernel
 */		
                View3D u, v, utmp, vtmp ;
                int nx, ny, nz ;
                FunctorUpdate(View3D u, View3D v, View3D utmp, View3D vtmp,
                               int nx, int ny, int nz) :
                        u(u), v(v), utmp(utmp), vtmp(vtmp),
                        nx(nx), ny(ny), nz(nz) {}

                KOKKOS_INLINE_FUNCTION
                void operator()(int i, int j, int k) const {
                        u(i,j,k) += utmp(i,j,k) ;
                        v(i,j,k) += vtmp(i,j,k) ;
                }
        };




	void initialize_random() {
/**
 *	@brief Call the Kokkos kernel that initialize with random values
 *	@todo Implement the dedicated Functor
 */		
	        Kokkos::parallel_for("init_random", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
	            KOKKOS_LAMBDA(int i, int j, int k) {
//	                u(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
//	                v(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
	            });
	}

	void initialize_central_block() {
/**
 *	@brief Call the Kokkos kernel that initialize with the central block policy
 *	@todo Usage of hierarchical parallelism
 */		
		FunctorCentralBlock functor(u, v, utmp, vtmp, nx,ny,nz) ; 
		Kokkos::parallel_for("test", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

	}

        void update() {
/**
 *	@brief Call the Kokkos kernel that update the main fields u and v with the newly updated  utmp and vtmp fields
 */		
                FunctorUpdate functor(u, v, utmp, vtmp, nx,ny,nz) ;
                Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

        }



	void perform_time_step() {
/**
 *	@brief Select the Kokkos stencil kernel mode 
 *	@todo Enable 2D and 3D configurations
 */
                switch(setting.stencilType) {
                        case StencilType::SevenPoint :
                                stencil_seven_point();
                        case StencilType::TwentySevenPoint:
                                stencil_twenty_seven_point();
                }
		update();
	}

        void initialize() {
/**
 *	@brief Select the field initialization mode
 *	@todo use switch case and implement the full and random versions
 */

		
		if (setting.initType == InitializationType::CentralBlock){
			initialize_central_block();
		}
		else if (setting.initType == InitializationType::Full){
                        //initialize_full();
                }
                if (setting.initType == InitializationType::Random){
                        initialize_random();
                }

        }


	void stencil_seven_point() {
/**
 *	@brief Call the seven point stencil Kokkos kernel
 *	@todo Add hierarchical paralellism policy
 */		
                FunctorStencilSevenPoint functor(u, v, utmp, vtmp, setting) ;
                Kokkos::parallel_for("test3", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

	}
	void stencil_twenty_seven_point(){
/**
 *	@brief Call the twenty seven points stencil kokkos kernel
 *	@todo Implement it !!
 */		
	}

	void save_to_hdf(HDF& hdf,  Kokkos::View<real***>::HostMirror& h_view, const std::string filename){
/**
 *	@brief save a flattened Kokkos Host View in the already created HDF file 
 */		
		hdf.dataset = hdf.file.createDataSet<real>(filename, HighFive::DataSpace(hdf.dims));

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

	std::cout << "-- GrayScott Generator --"<< std::endl ; 

	GrayScott::Setting setting ;
	setting.is3D = true ;
	setting.initType = GrayScott::InitializationType::CentralBlock;
	setting.stencilType = GrayScott::StencilType::SevenPoint;	
	setting.bc = GrayScott::BoundaryCondition::Periodic;

	// Here : add parser that modify GrayScott::Setting parameters
	// Here : add printf of the modified / unmodified parameters if wanted 

	Kokkos::initialize(argc, argv);
	{		
		GrayScott simulation(setting);
		simulation.allocate();
		simulation.initialize() ; 
		simulation.run() ; 
	}
	Kokkos::finalize() ; 
	return 0 ; 
}


