import h5py
import numpy as np
import pyvista as pv
import imageio

import cmocean

import os as os
from PIL import Image


pv.global_theme.allow_empty_mesh = True

def vec_to_mesh(frame, nx, ny, nz):
        grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
        ugrid =  grid.cast_to_unstructured_grid()
        ugrid.cell_data['values'] = frame
        return ugrid

def apply_threshold(mesh, value):
    return mesh.threshold([value,1])



def apply_light_cubemap(pl):
    from pyvista import examples
    cubemap = examples.download_sky_box_cube_map()
    pl.set_environment_texture(cubemap)

def apply_light_custom(pl):
    color = 'white'
    color = "ffff99"
    light = pv.Light((-2, 2, 0), (0, 0, 0), color)
    pl.add_light(light)
    light = pv.Light((2, 0, 0), (0, 0, 0), color)
    pl.add_light(light)
    light = pv.Light((0, 0, 10), (0, 0, 0), color)
    pl.add_light(light)
    
def extract_smooth_surface(mesh, iter, pb):
    surf = threshold.extract_geometry()
    # Smooth the surface
    smooth = surf.smooth_taubin(n_iter=iter, pass_band=pb)    #50 et 0.05
    return smooth

def add_mesh_custom(pl, mesh):
    pl.add_mesh(mesh, show_scalar_bar=False, color="ffffcc", smooth_shading=True, pbr=True, metallic=0.25, roughness=0.4, diffuse=0.2)



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

    
    gif = []
    for i, key in enumerate(frame_keys):
        print(i)
        frame = file[key][:]

        ugrid = vec_to_mesh(frame, nx, ny, nz)
        threshold = apply_threshold(ugrid, 0.1)

        outline = ugrid.outline()


        smooth = extract_smooth_surface(threshold, 50, 0.05)
        p = pv.Plotter(off_screen=True)

        p.enable_eye_dome_lighting()

        p.set_background("black")


        p.add_mesh(outline)

        add_mesh_custom(p,smooth)
        p.view_isometric()
        
        p.camera.azimuth += 2*i

        apply_light_cubemap(p)
#        apply_light_custom(p)
        folder = "img/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = "img"
        filenames = folder  + filename+"{}.png".format(i)
        p.screenshot(filenames)
        gif.append(Image.open(filenames))
        p.clear()
        p.close()

imageio.mimsave('GS_3D.gif', gif, fps=20)

