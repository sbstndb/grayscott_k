import h5py
import numpy as np
import pyvista as pv
import imageio


# Ouvrir le fichier HDF5
with h5py.File("dump", 'r') as file:
    frame_keys = sorted([key for key in file.keys()], key=lambda x: int(x))[2:-1]# 0 --> champ constant
    num_frames = len(frame_keys)
    print(f"Number of frames: {num_frames}")

    first_frame_data = file[frame_keys[0]][:]
    total_elements = first_frame_data.size
    cube_dim = int(np.cbrt(total_elements))  
    nx, ny, nz = cube_dim, cube_dim, cube_dim
    print(f"Determined 3D dimensions: {nx}x{ny}x{nz}")

    last_frame = file[frame_keys[-1]][:]

    p = pv.Plotter(off_screen=True)
    
    gif = []
    for i, key in enumerate(frame_keys):
        print(i)
        frame = file[key][:]
   
        grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
        ugrid =  grid.cast_to_unstructured_grid()
        ugrid.cell_data['values'] = frame
        threshold = ugrid.threshold([0.2,1])
        outline = grid.outline()
        p.add_mesh(outline)
#        p.add_mesh(threshold)
        p.add_volume(threshold, opacity="linear", ambient=0.2, shade=True)

        #p.update()
##        p.show()
#        p.save_graphic("g.svg")
        p.camera.azimuth += 2
        img=p.screenshot()
        gif.append(img)
        p.clear()

    p.show()
    imageio.mimsave('GS_3D.gif', gif, fps=60)
    
