import h5py
import numpy as np
import pyvista as pv
import imageio


# Ouvrir le fichier HDF5
with h5py.File("dump", 'r') as file:
    frame_keys = sorted([key for key in file.keys()], key=lambda x: int(x))[0:-1]# 0 --> champ constant
    num_frames = len(frame_keys)
    print(f"Number of frames: {num_frames}")

    first_frame_data = file[frame_keys[0]][:]
    total_elements = first_frame_data.size
    cube_dim = int(np.cbrt(total_elements))  
    nx, ny, nz = cube_dim, cube_dim, cube_dim
    print(f"Determined 3D dimensions: {nx}x{ny}x{nz}")

    last_frame = file[frame_keys[-1]][:].reshape((nx, ny, nz))

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    x, y, z = np.meshgrid(x, y, z)
    
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (nx, ny, nz)
    
    p = pv.Plotter(off_screen=True)
    
    gif = []
    for i, key in enumerate(frame_keys):
        print(i)
        frame = file[key][:].reshape((nx, ny, nz))
        grid["values"] = frame.flatten(order="F")
    
        p.add_volume(grid, scalars="values", opacity_unit_distance=0.1)
        #p.update()
##        p.show()
#        p.save_graphic("g.svg")
        img=p.screenshot()
        gif.append(img)
        p.clear()

    p.show()
    imageio.mimsave('GS_3D.gif', gif, fps=5)
    
