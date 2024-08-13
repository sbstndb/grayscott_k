import pyvista as pv
import numpy as np

# Dimensions du tableau 3D
nx, ny, nz = 50, 50, 50

# Exemple : Création d'un tableau 3D aléatoire (remplacez ceci par vos propres données)
data = np.random.random((nx, ny, nz))

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
grid["data"] = data.flatten()

# Créer un plotter
plotter = pv.Plotter()

# Ajouter la visualisation du champ en utilisant add_volume
plotter.add_volume(grid, scalars="data", opacity="sigmoid", cmap="viridis")

# Afficher la visualisation
plotter.show()

