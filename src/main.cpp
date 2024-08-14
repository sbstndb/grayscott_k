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


class HDF {
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
	        int nrepeat = 100 ;
	
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

		allocate();
		initialize() ; 
	}

	void allocate(){

                u = Kokkos::View<real***>("u", nx, ny, nz);
                v = Kokkos::View<real***>("v", nx, ny, nz);
                h_u = Kokkos::create_mirror_view (u);
                h_v = Kokkos::create_mirror_view (v);

                s =  Kokkos::View<GrayScott::Setting*>("s",1);
                h_s = Kokkos::create_mirror_view (s);

                Kokkos::deep_copy(s, h_s);
		Kokkos::deep_copy(u, h_u);
		Kokkos::deep_copy(v, h_v);

                utmp = Kokkos::View<real***>("utmp", nx, ny, nz);
                vtmp = Kokkos::View<real***>("vtmp", nx, ny, nz);
	}


	void run(){
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
		Kokkos::View<real***> u, v, utmp, vtmp ; 
		int nx, ny, nz ; 
		FunctorCentralBlock(Kokkos::View<real***> u ,Kokkos::View<real***> v, Kokkos::View<real***> utmp,Kokkos::View<real***> vtmp,
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
                Kokkos::View<real***> u, v, utmp, vtmp ;
                Setting s ;
                FunctorStencilSevenPoint(Kokkos::View<real***> u ,Kokkos::View<real***> v, Kokkos::View<real***> utmp,Kokkos::View<real***> vtmp,
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
                Kokkos::View<real***> u, v, utmp, vtmp ;
                int nx, ny, nz ;
                FunctorUpdate(Kokkos::View<real***> u ,Kokkos::View<real***> v, Kokkos::View<real***> utmp,Kokkos::View<real***> vtmp,
                               int nx, int ny, int nz) :
                        u(u), v(v), utmp(utmp), vtmp(vtmp),
                        nx(nx), ny(ny), nz(nz) {}

                KOKKOS_INLINE_FUNCTION
                void operator()(int i, int j, int k) const {
                        u(i,j,k) +=utmp(i,j,k);
                        v(i,j,k) += vtmp(i,j,k) ;
                }
        };




	void initialize_random() {
	        Kokkos::parallel_for("init_random", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
	            KOKKOS_LAMBDA(int i, int j, int k) {
//	                u(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
//	                v(i, j, k) = static_cast<real>(rand()) / RAND_MAX;
	            });
	}

	void initialize_central_block() {
		FunctorCentralBlock functor(u, v, utmp, vtmp, nx,ny,nz) ; 
		Kokkos::parallel_for("test", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

	}

        void update() {
                FunctorUpdate functor(u, v, utmp, vtmp, nx,ny,nz) ;
                Kokkos::parallel_for("update", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

        }



	void perform_time_step() {

                switch(setting.stencilType) {
                        case StencilType::SevenPoint :
                                stencil_seven_point();
                        case StencilType::TwentySevenPoint:
                                stencil_twenty_seven_point();
                }
		update();
		// update here 
	}

        void initialize() {
		if (setting.initType == InitializationType::CentralBlock){
			initialize_central_block();
		}
		else if (setting.initType == InitializationType::Full){
			std::cout << "ighjfouhjd" << std::endl ; 
                        //initialize_full();
                }
                if (setting.initType == InitializationType::Random){
                        initialize_random();
                }

        }


	void stencil_seven_point() {
                FunctorStencilSevenPoint functor(u, v, utmp, vtmp, setting) ;
                Kokkos::parallel_for("test3", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx, ny, nz}), functor);

	}
	void stencil_twenty_seven_point(){
		//todo
	}

	void save_to_hdf(HDF& hdf,  Kokkos::View<real***>::HostMirror& h_view, const std::string filename){
		hdf.dataset = hdf.file.createDataSet<real>(filename, HighFive::DataSpace(hdf.dims));

		for (int i = 0 ; i < nx ; i++){
	                        for (int j = 0 ; j < ny ; j++){
	                                        for (int k = 0 ; k < nz ; k++){
	                                                hdf.tmp[i*ny*nz + j*nz + k] = h_view(i,j,k);
                                                        //hdf.tmp[i*ny*nz + j*nz + k] = 0.0;							
	                                        }
	                        }
	        }
		std::cout << "here" << std::endl ; 
	        hdf.dataset.write(hdf.tmp);
                std::cout << "not here" << std::endl ;

	}

};


using View = Kokkos::View<real*>;

class Test {
public:
	View a = View("a", 10);
	
};

class Test2 {
public:
	View a ; 
	View::HostMirror b ; 
};


class Test3 {
public:
	View a ; 
	View::HostMirror b ; 
	void init(int n){
		a = View("a", n) ; 
		b = create_mirror_view(a) ; 
	}
	void c(){
		Kokkos::deep_copy(b, a) ; 
	}
};

int main(int argc, char ** argv){
	Kokkos::initialize(argc, argv);
	{	



	
		GrayScott::Setting setting ; 
		setting.is3D = true ; 
		setting.initType = GrayScott::InitializationType::CentralBlock;
		setting.stencilType = GrayScott::StencilType::SevenPoint;
		setting.bc = GrayScott::BoundaryCondition::Periodic;
		
		GrayScott simulation(setting);
		simulation.allocate();
		simulation.initialize() ; 
		simulation.run() ; 
	
	}
	Kokkos::finalize() ; 
	return 0 ; 
}


