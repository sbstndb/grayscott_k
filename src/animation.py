import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_frame(filename):
    with open(filename, 'r') as f:
        # Lire la première ligne pour obtenir nx et ny
        first_line = f.readline().strip()
        nx, ny = map(int, first_line.split())
        
        # Charger les données en sautant une ligne à chaque "ny" valeurs
        data = np.loadtxt(f).reshape((nx, ny))
    
    return data

def get_frame_filenames(folder):
    files = []
    for file in sorted(os.listdir(folder)):
        if file.startswith('frame_u_') and file.endswith('.txt'):
            files.append(os.path.join(folder, file))
    return files

def create_animation(filenames):
    # Charger la première frame pour initialiser la figure
    first_frame = load_frame(filenames[0])
    nx, ny = first_frame.shape
    
    fig, ax = plt.subplots()
    cax = ax.imshow(first_frame, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cax)
    
    def update(frame):
        data = load_frame(frame)
        cax.set_array(data)
        return cax,

    ani = animation.FuncAnimation(fig, update, frames=filenames, blit=True)
    plt.show()

def main():
    folder = 'frames'
    filenames = get_frame_filenames(folder)
    
    if not filenames:
        print(f"No files found in {folder}")
        return
    
    print(f"Found {len(filenames)} files.")
    create_animation(filenames)

if __name__ == '__main__':
    main()

