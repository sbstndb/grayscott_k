import h5py
import numpy as np
import pyvista as pv



# Ouvrir le fichier HDF5
with h5py.File("dump", 'r') as file:
    # Déterminer le nombre de frames en lisant tous les datasets
    frame_keys = sorted([key for key in file.keys()], key=lambda x: int(x))
    num_frames = len(frame_keys)
    print(f"Number of frames: {num_frames}")

    # Déterminer les dimensions du tableau 3D à partir de la première frame
    first_frame_data = file[frame_keys[0]][:]
    total_elements = first_frame_data.size
    cube_dim = int(np.cbrt(total_elements))  # Calculer la racine cubique du nombre total d'éléments
    nx, ny, nz = cube_dim, cube_dim, cube_dim
    print(f"Determined 3D dimensions: {nx}x{ny}x{nz}")

    # Lire la dernière frame et la reshaper en cube 3D
#    last_frame = file[frame_keys[-1]][:].reshape((nx, ny, nz))
    last_frame = file[frame_keys[-1]][:]


grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
ugrid = grid.cast_to_unstructured_grid()


ncells = ugrid.n_cells
#ugrid.cell_data['values'] = np.random.rand(ncells,1)
ugrid.cell_data['values'] = last_frame

threshold = ugrid.threshold([0.2, 1])



pl = pv.Plotter()

#" rotation camera
#pl.camera.azimuth += 360 / 10
#pl.camera.azimuth %= 360
#pl.camera.azimuth +=0.0

outline = grid.outline()
pl.add_mesh(outline)
#pl.add_mesh(threshold)
pl.add_volume(threshold)
pl.camera.azimuth += 50.0
pl.show()


