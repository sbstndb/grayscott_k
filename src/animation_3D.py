import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ouvrir le fichier HDF5
with h5py.File("dump", 'r') as file:
    # Spécifiez les dimensions du cube (ex. 64x64x64)
    nx, ny, nz = 64, 64, 64

    # Lire les 10 premières frames et les reshaper en cubes 3D
    frames = []
    for i in range(10):
        data = file[str(i)][:]  # Lire les données de la frame i
        cube = data.reshape((nx, ny, nz))  # Reshape en 3D
        frames.append(cube)

# Choisissez l'axe et le plan médian pour l'animation (0 = x, 1 = y, 2 = z)
axis = 2  # Choix de l'axe pour le plan médian

# Extraction du plan médian
if axis == 0:
    plane_frames = [frame[nx // 2, :, :] for frame in frames]
elif axis == 1:
    plane_frames = [frame[:, ny // 2, :] for frame in frames]
else:
    plane_frames = [frame[:, :, nz // 2] for frame in frames]

# Création de l'animation
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(frame, cmap='viridis')
    ax.set_title('Plan médian sur l\'axe {}'.format(['X', 'Y', 'Z'][axis]))

ani = animation.FuncAnimation(fig, update, frames=plane_frames, interval=200, repeat=True)

# Afficher l'animation
plt.show()

