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
    last_frame = file[frame_keys[-1]][:].reshape((nx, ny, nz))

# Dimensions du grid
nx, ny, nz = 64, 64, 64

grid = pv.ImageData(dimensions=(nx, ny, nz));

# Définir les dimensions du grid
grid.dimensions = np.array(last_frame.shape) + 1

# Les points d'origine
grid.origin = (0, 0, 0)  # Origine

# La distance entre les points
grid.spacing = (1/nx, 1/ny, 1/nz)  # Espacement (1, 1, 1)

# Ajouter les données de scalaires au grid
#grid.cell_data["values"] = last_frame.flatten(order="F")  # Ordre Fortran pour PyVista
grid["values"] = last_frame.flatten(order="F")
# Créer une visualisation 3D avec transparence
opacity = np.linspace(0, 1, 256)  # Définir un mappage d'opacité
p = pv.Plotter()
p.add_volume(grid, cmap="viridis", opacity=opacity)
p.show()

