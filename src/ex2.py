import pyvista as pv
import numpy as np

# Taille du maillage
nx, ny, nz = 50, 50, 50

# Création d'une grille rectangulaire
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
z = np.linspace(-10, 10, nz)
x, y, z = np.meshgrid(x, y, z)

# Création d'un champ scalaire 3D, ici une gaussienne 3D
sigma = 5.0
field = np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

# Conversion des grilles en un tableau de points pour StructuredGrid
points = np.c_[x.ravel(), y.ravel(), z.ravel()]

# Créer le maillage StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Ajouter les données scalaires
grid["field"] = field.ravel()

# Créer un plotter
plotter = pv.Plotter()

# Ajouter la visualisation du champ en utilisant add_volume
plotter.add_volume(grid, scalars="field", opacity="sigmoid", cmap="viridis")

# Afficher la visualisation
plotter.show()

