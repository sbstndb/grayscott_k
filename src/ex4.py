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



# Création des coordonnées pour chaque point
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
x, y, z = np.meshgrid(x, y, z)

# Fusionner les coordonnées pour créer un tableau de points 3D
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

# Créer un StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Ajouter les données scalaires
grid["values"] = last_frame.flatten(order="F")

# Créer une visualisation 3D avec transparence

opacity = np.zeros(256)
opacity[50:150] = np.linspace(0.2, 1.0, 100)  # Progressivement plus opaque au milieu

opacity="sigmoid"



p = pv.Plotter()
p.add_volume(grid, scalars="values", cmap="viridis", opacity=0.8)
p.show()

