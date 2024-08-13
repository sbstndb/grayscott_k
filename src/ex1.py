import pyvista as pv
grid = pv.ImageData(dimensions=(9, 9, 9))
grid['scalars'] = -grid.x
pl = pv.Plotter()
_ = pl.add_volume(grid, opacity='linear')
pl.show()
